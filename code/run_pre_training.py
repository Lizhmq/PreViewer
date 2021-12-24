import os
import torch
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from utils import convert_examples_to_features
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import TextDataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loaders(args, tokenizer, pool):
    def fn(examples):
        feats = pool.map(convert_examples_to_features, [(example, tokenizer, args) for example in examples])
        # feats = [convert_examples_to_features((example, tokenizer, args)) for example in examples]
        return feats
    num_train_optimization_steps = 0
    data_tuples = []
    data_list = [os.path.join(args.train_path, f"{lang}_gen.jsonl") for lang in args.langs]
    for data_file in data_list:
        dataset = TextDataset(tokenizer, pool, args, data_file)
        # sampler = DistributedSampler(dataset)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=fn)
        num_train_optimization_steps += args.num_train_epochs * len(dataloader) * args.gradient_accumulation_steps
        dataloader = cycle(dataloader)
        data_tuples.append((dataset, sampler, dataloader))
    return data_tuples, num_train_optimization_steps


def main(local_rank, args):
    ip = os.environ["MASTER_IP"]
    port = os.environ["MASTER_PORT"]
    hosts = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    gpus = torch.cuda.device_count()
    args.n_gpu = gpus
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=hosts * gpus,
        rank=rank * gpus + local_rank,
    )
    torch.cuda.set_device(local_rank)

    t0 = time.time()
    # set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    if os.path.exists("{}/checkpoints/pytorch_model.bin".format(args.output_dir)):
        model.load_state_dict(
            torch.load("{}/checkpoints/pytorch_model.bin".format(args.output_dir))
        )
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank)
    pool = multiprocessing.Pool(args.cpu_count)

    # data_list = ["data_{}.json".format(i) for i in range(5)]
    data_tuples, num_train_optimization_steps = get_loaders(args, tokenizer, pool)
    args.num_train_optimization_steps = num_train_optimization_steps
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    args.warmup_steps = num_train_optimization_steps * 0.1
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_train_optimization_steps,
    )

    if os.path.exists("{}/checkpoints/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/checkpoints/scheduler.pt".format(args.output_dir),
                map_location="cpu",
            )
        )

    global_step = 0
    global_epoch = 0
    save_steps = 1000

    probs = [len(d) for (d, _, _) in data_tuples]
    probs = [x / sum(probs) for x in probs]
    probs = [x ** 0.7 for x in probs]
    probs = [x / sum(probs) for x in probs]

    per_epoch_steps = args.num_train_optimization_steps // args.num_train_epochs
    for _, _, dataloader in data_tuples:
        dataloader.sampler.set_epoch(global_epoch)
    model.train()
    while True:
        global_step += 1
        dataset, sampler, dataloader = np.random.choice(data_tuples, 1, p=probs)[0]
        if global_step % per_epoch_steps == 0:
            global_epoch += 1
            if global_epoch > args.num_train_epochs:
                break
            for _, _, dataloader in data_tuples:
                dataloader.sampler.set_epoch(global_epoch)
        
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        examples = next(dataloader)
        tuple_examples = [(ex, tokenizer, args) for ex in examples]
        features = pool.map(convert_examples_to_features, tuple_examples)
        source_ids = torch.tensor(
            [ex.input_ids for ex in features], dtype=torch.long
        ).to(local_rank)
        # ///////////////
        target_ids = torch.tensor(
            [ex.target_ids for ex in features], dtype=torch.long
        ).to(local_rank)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        outputs = model(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask,
        )
        loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()

        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            if rank == 0 and local_rank == 0 and global_step % 100 == 0:
                train_loss = round(
                    tr_loss
                    * args.gradient_accumulation_steps
                    / (nb_tr_steps + 1),
                    4,
                )
                logger.info(
                    "step {}/{}: Train loss {}".format(
                        global_step,
                        num_train_optimization_steps,
                        round(train_loss, 3),
                    )
                )

        if rank == 0 and local_rank == 0 and global_step % save_steps == 0:
            # Save the checkpoint for each epoch
            output_dir = os.path.join(args.output_dir, "checkpoints")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            config.save_pretrained(output_dir)
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
            torch.save(
                optimizer.state_dict(),
                output_optimizer_file,
                _use_new_zipfile_serialization=False,
            )
            output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
            torch.save(
                scheduler.state_dict(),
                output_scheduler_file,
                _use_new_zipfile_serialization=False,
            )
            logger.info(
                "Save the {}-step model and optimizer into {}".format(
                    global_step, output_dir
                )
            )
            time.sleep(5)
        global_step += 1

    if rank == 0 and local_rank == 0:
        # Save the final checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        config.save_pretrained(output_dir)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
        torch.save(
            optimizer.state_dict(),
            output_optimizer_file,
            _use_new_zipfile_serialization=False,
        )
        output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
        torch.save(
            scheduler.state_dict(),
            output_scheduler_file,
            _use_new_zipfile_serialization=False,
        )
        logger.info("Save the trained model and optimizer into {}".format(output_dir))
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.train_path = ''
    args.langs = [""]
    # args.model_type = "codet5"
    # args.train_filename = "../data/pre-training/"
    # args.num_train_epochs = 10
    # args.learning_rate = 5e-5
    # args.tokenizer_name = "roberta-base"
    # args.tokenizer_path = "../tokenizer/salesforce"
    # args.model_name_or_path = "../pretrained_models/codet5_base"
    # args.output_dir = "../editing_models/editing_base"
    # args.train_batch_size = 5
    # args.max_source_length = 512
    # args.max_target_length = 256
    args.cpu_count = multiprocessing.cpu_count()
    logger.info(args)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
