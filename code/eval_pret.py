import os
import torch
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import ConcatDataset
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


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    # add concat dataset
    dataset = TextDataset(tokenizer, pool, args, data_file)
    # sampler = DistributedSampler(dataset)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.train_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    # load model
    _, model, tokenizer = build_or_load_gen_model(args)
    # load last model
    if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)):
        model.load_state_dict(
            torch.load("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
        )
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank)
    pool = multiprocessing.Pool(args.cpu_count)

    data_file = args.eval_file

    global_step = 0
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    tr_loss = 0
    with torch.no_grad():
        for step, examples in enumerate(dataloader, 1):
            if step == 1:
                ex = examples[0]
                logger.info(f"batch size: {len(examples)}")
                logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                # logger.info(f"example label: {tokenizer.convert_ids_to_tokens(ex.source_labels)}")
                logger.info(f"example target: {tokenizer.convert_ids_to_tokens(ex.target_ids)}")
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
            tr_loss += loss.item()
            global_step += 1
            train_loss = round(
                tr_loss / global_step,
                4,
            )
            logger.info(
                "step {}: loss {}, ppl {}".format(
                    global_step,
                    round(train_loss, 3),
                    round(np.exp(train_loss), 3),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Evaluation finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
