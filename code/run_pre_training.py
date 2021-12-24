import os
import torch
import logging
import argparse
import random
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


def get_loaders(data_list, args, tokenizer, pool):
    def fn(examples):
        feats = pool.map(convert_examples_to_features, [(example, tokenizer, args) for example in examples])
        # feats = [convert_examples_to_features((example, tokenizer, args)) for example in examples]
        return feats
    random.shuffle(data_list)       # this will shuffle data chunks
    for data_file in data_list:
        dataset = TextDataset(tokenizer, pool, args, data_file)
        sampler = DistributedSampler(dataset)
        # sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=fn)
        dataloader = cycle(dataloader)
        yield dataset, sampler, dataloader

def save_model(model, optimizer, scheduler, output_dir, config):
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


def main(local_rank, args):
    LOCAL = os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    if LOCAL:
        # means local here
        os.environ["MASTER_IP"] = "*"
        os.environ["MASTER_PORT"] = "22333"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
    gpus = torch.cuda.device_count()
    args.n_gpu = gpus
    ip = os.environ["MASTER_IP"]
    port = os.environ["MASTER_PORT"]
    hosts = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
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

    data_list = [os.path.join(args.train_path, f"{lang}_gen.jsonl") for lang in args.langs]
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
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
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
    save_steps = args.save_steps

    for epoch in range(1, args.train_epochs + 1):
        data_tuples = get_loaders(data_list, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        chunknum = 0
        for _, _, dataloader in data_tuples:
            logger.info(f"Start chunk {chunknum}")
            chunknum += 1
            for step, examples in enumerate(dataloader):
                source_ids = torch.tensor(
                    [ex.source_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                source_labels = torch.tensor(
                    [ex.source_labels for ex in examples], dtype=torch.long
                ).to(local_rank)
                target_ids = torch.tensor(
                    [ex.target_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                source_mask = source_ids.ne(tokenizer.pad_id)
                target_mask = target_ids.ne(tokenizer.pad_id)

                loss = model(
                    input_ids=source_ids,
                    input_labels=source_labels,
                    decoder_input_ids=target_ids,
                    attention_mask=source_mask,
                    decoder_attention_mask=target_mask,
                )

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
                    if rank == 0 and local_rank == 0 and global_step % args.log_steps == 0:
                        train_loss = round(
                            tr_loss
                            * args.gradient_accumulation_steps
                            / (nb_tr_steps + 1),
                            4,
                        )
                        logger.info(
                            "step {}/{}: Train loss {}".format(
                                global_step,
                                args.train_steps,
                                round(train_loss, 3),
                            )
                        )
                    if global_step == args.train_steps:
                        output_dir = os.path.join(args.output_dir, "checkpoints" + str(global_step))
                        save_model(model, optimizer, scheduler, output_dir, config)
                        logger.info(f"Reach max steps {args.train_steps}.")
                        time.sleep(5)
                        return

                if rank == 0 and local_rank == 0 and global_step % save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step))
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(
                        "Save the {}-step model and optimizer into {}".format(
                            global_step, output_dir
                        )
                    )
                    time.sleep(5)
    if rank == 0 and local_rank == 0:
        # Save the final checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoints")
        save_model(model, optimizer, scheduler, output_dir, config)
        logger.info("Save the trained model and optimizer into {}".format(output_dir))
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    args.langs = ["ruby"]
    logger.info(args)
    # main(0, args)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
