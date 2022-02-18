import os
import torch
import logging
import argparse
import numpy as np
import multiprocessing
import time, json
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import TextDataset
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluator.bleu import _bleu
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    # add concat dataset
    dataset = TextDataset(tokenizer, pool, args, data_file)
    # sampler = DistributedSampler(dataset)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_ppl_epoch(args, eval_dataloader, model, tokenizer):
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    local_rank = 0
    with torch.no_grad():
        for step, examples in enumerate(eval_dataloader, 1):
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
                encoder_loss=False
            )
            eval_loss += loss.item()
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_dataloader, model, tokenizer):
    # if dist.get_rank() == 0:
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    bleu = 0.0
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    gold_src, gold_tgt = [], []
    kk = 0
    for step, examples in tqdm(enumerate(eval_dataloader)):
        kk += 1
        if kk == 6:
            break
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        gold_src.extend(tokenizer.decode(ex.source_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ex in examples)
        gold_tgt.extend(tokenizer.decode(ex.target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ex in examples)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                            attention_mask=source_mask,
                            use_cache=True,
                            num_beams=args.beam_size,
                            early_stopping=True,
                            max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
        ex_ids.extend(ids)
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.load_model_path, "test.output")
    gold_fn = os.path.join(args.load_model_path, "test.gold")
    src_fn = os.path.join(args.load_model_path, "test.src")

    # gold_src, gold_tgt = [], []
    # with open(args.eval_file, "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         js = json.loads(line)
    #         if "msg" not in js or len(js["msg"]) == 0:
    #             continue
    #         src = js["patch"].replace("\n", "\\n")
    #         tgt = js["msg"].replace("\n", " ")
    #         gold_src.append(src)
    #         gold_tgt.append(tgt)
    gold_src = [gold_src[i] for i in ex_ids]
    gold_tgt = [gold_tgt[i] for i in ex_ids]
    logger.info(f"Gold len: {len(gold_src)}")
    logger.info(f"Pred len: {len(pred_nls)}")
    dev_accs = []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, src, tgt in zip(pred_nls, gold_src, gold_tgt):
            dev_accs.append(pred_nl.strip() == tgt.strip())
            f.write(pred_nl.strip().replace("\n", " ") + '\n')
            f1.write(tgt.strip() + '\n')
            f2.write(src.strip() + '\n')
    bleu = round(_bleu(gold_fn, output_fn), 2)
    em = np.mean(dev_accs) * 100
    result = {'em': em, 'bleu': bleu}
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result


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
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)

    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    eval_bleu_epoch(args, dataloader, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Evaluation finished.")
