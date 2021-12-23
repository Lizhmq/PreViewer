# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import hf_env
hf_env.set_env('202105')
import hfai
import os
import torch
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

work_dir = '/ceph-jd/pub/jupyter/lizhuo/notebooks/examples/Editing-main'

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    # if dist.get_rank() == 0:
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for step, batch in enumerate(eval_dataloader):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.module.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=(args.task == 'summarize' or args.task == 'summarize_intent'),
                                        max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
        if step % 100 == 0:
            logger.info("Eval bleu: {}/{}".format(step, len(eval_dataloader)))

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize', 'summarize_intent']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize' or args.task == 'summarize_intent':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task == 'concode':
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        em = np.mean(dev_accs) * 100
        result = {'em': em, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main(args):
    t0 = time.time()
    config, model, tokenizer = build_or_load_gen_model(args)
    if os.path.exists('{}/checkpoint-last/pytorch_model.bin'.format(args.output_dir)):
        model.load_state_dict(torch.load('{}/checkpoint-last/pytorch_model.bin'.format(args.output_dir), map_location='cpu'))
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
        # model = model.module
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    # if args.local_rank in [-1, 0]:
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        args.warmup_steps = num_train_optimization_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        if os.path.exists('{}/checkpoint-last/optimizer.pt'.format(args.output_dir)): 
            optimizer.load_state_dict(torch.load('{}/checkpoint-last/optimizer.pt'.format(args.output_dir), map_location='cpu'))
            scheduler.load_state_dict(torch.load('{}/checkpoint-last/scheduler.pt'.format(args.output_dir), map_location='cpu'))
            
        # Start training
        train_example_num = len(train_data)
        if args.local_rank in [-1, 0]:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_example_num)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
            logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        save_steps = len(train_dataloader) // 5
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            # if args.local_rank in [-1, 0]:
            #     bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            # else:
            #     bar = train_dataloader
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            # train_dataloader.sampler.set_epoch(cur_epoch)
            model.train()
            for step, batch in enumerate(train_dataloader):
                if global_step < hfai.get_whole_life_state():
                    global_step += 1
                    continue
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
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
                    if args.local_rank in [-1, 0] and global_step % save_steps == 0:
                        train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                        logger.info("step {}/{}: Train loss {}".format(
                            global_step, num_train_optimization_steps, round(train_loss, 3)))
                
                if global_step % save_steps == 0 or hfai.receive_suspend_command():
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_optimizer_file = os.path.join(last_output_dir, "optimizer.pt")
                    torch.save(optimizer.state_dict(), output_optimizer_file, _use_new_zipfile_serialization=False)
                    output_scheduler_file = os.path.join(last_output_dir, "scheduler.pt")
                    torch.save(scheduler.state_dict(), output_scheduler_file, _use_new_zipfile_serialization=False)
                    hfai.set_whole_life_state(global_step)
                    time.sleep(5)
                    logger.info("Save the last model into %s", output_model_file)
                    
            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                # if args.save_last_checkpoints or hfai.receive_suspend_command():
                #     last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                #     if not os.path.exists(last_output_dir):
                #         os.makedirs(last_output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)
                #     output_optimizer_file = os.path.join(last_output_dir, "optimizer.pt")
                #     torch.save(optimizer.state_dict(), output_optimizer_file, _use_new_zipfile_serialization=False)
                #     output_scheduler_file = os.path.join(last_output_dir, "scheduler.pt")
                #     torch.save(scheduler.state_dict(), output_scheduler_file, _use_new_zipfile_serialization=False)
                #     hfai.set_whole_life_state(global_step)
                #     time.sleep(5)
                #     logger.info("Save the last model into %s", output_model_file)
                    

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize', 'summarize_intent']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
            logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu', 'best-ppl']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.module.load_state_dict(torch.load(file))
            # model = torch.load(file)
            # if isinstance(model, torch.nn.DataParallel):
                # model = model.module
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


def get_args_by_task_model(task, sub_task, model_tag):
    if task == 'translate':
        # java-cs: Read 10300 examples, avg src len: 13, avg trg len: 15, max src len: 136, max trg len: 118
        # [TOKENIZE] avg src len: 45, avg trg len: 56, max src len: 391, max trg len: 404
        src_len = 320
        trg_len = 256
        epoch = 100
        patience = 5
    elif task == 'summarize':
        # ruby: Read 24927 examples, avg src len: 66, avg trg len: 12, max src len: 501, max trg len: 146
        # [TOKENIZE] avg src len: 100, avg trg len: 13, max src len: 1250, max trg len: 161
        # Python: Read 251820 examples, avg src len: 100, avg trg len: 11, max src len: 512, max trg len: 222
        # [TOKENIZE] avg src len: 142, avg trg len: 12, max src len: 2016, max trg len: 245
        # Javascript: Read 58025 examples, avg src len: 114, avg trg len: 11, max src len: 512, max trg len: 165
        # [TOKENIZE] avg src len: 136, avg trg len: 12, max src len: 3016, max trg len: 177
        src_len = 256 
        trg_len = 128
        epoch = 15
        patience = 2
    elif task == 'summarize_intent':
        src_len = 384 # original: 256, 256(input code)+128(retrieval comment)=384
        trg_len = 128
        epoch = 15
        patience = 2
    elif task == 'refine':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'small':
            src_len = 130 
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 240 
            trg_len = 240
        epoch = 50
        patience = 5
    elif task == 'refine_intent': # add by lijia
        if sub_task == 'small':
            src_len = 180 # 50(commit msg) + 130(input code) = 180
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 290 # 240 + 50
            trg_len = 240
        epoch = 50
        patience = 5
    elif task == 'concode':
        # Read 100000 examples, avg src len: 71, avg trg len: 26, max src len: 567, max trg len: 140
        # [TOKENIZE] avg src len: 213, avg trg len: 33, max src len: 2246, max trg len: 264
        src_len = 320
        trg_len = 150
        epoch = 30
        patience = 3
    elif task == 'defect':
        # Read 21854 examples, avg src len: 187, avg trg len: 1, max src len: 12195, max trg len: 1
        # [TOKENIZE] avg src len: 597, avg trg len: 1, max src len: 41447, max trg len: 1
        src_len = 512
        trg_len = 3
        epoch = 10
        patience = 2
    elif task == 'clone':
        # Read 901028 examples, avg src len: 120, avg trg len: 123, max src len: 5270, max trg len: 5270
        # [TOKENIZE] avg src len: 318, avg trg len: 323, max src len: 15111, max trg len: 15111
        src_len = 400
        trg_len = 400
        epoch = 2
        patience = 1

    if 'codet5_small' in model_tag or 'editing_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'summarize_intent' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = 64
    else:
        bs = 32
        if task == 'translate':
            bs = 25
        elif task == 'summarize' or task == 'summarize_intent':
            bs = 48
    lr = 5
    if task == 'concode':
        lr = 10
    elif task == 'defect':
        lr = 2
    return bs, lr, src_len, trg_len, patience, epoch

def get_args(args, arg_dict):
    if arg_dict['data_num'] == -1:
        data_tag = 'all'
    else:
        data_tag = str(args.data_num)
        arg_dict['epoch'] = 1
    
    full_model_tag = '{}_{}_lr{}_bs{}_src{}_trg{}_pat{}_e{}'.format(
        arg_dict['model_tag'], data_tag, arg_dict['lr'], arg_dict['batch_size'], 
        arg_dict['src_len'], arg_dict['trg_len'], arg_dict['patience'], arg_dict['epoch'])
    
    if arg_dict['sub_task'] == 'none':
        output_dir = '{}/saved_models/{}/{}'.format(work_dir, arg_dict['task'], full_model_tag)
    else:
        output_dir = '{}/saved_models/{}/{}/{}'.format(work_dir, arg_dict['task'], arg_dict['sub_task'], full_model_tag)
    
    cache_dir = '{}/cache_data'.format(output_dir)
    res_dir = '{}/prediction'.format(output_dir)
    log = '{}/train.log'.format(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    model_type = 'codet5'
    tokenizer = 'roberta-base'
    tokenizer_path = '{}/tokenizer/salesforce'.format(work_dir)

    if arg_dict['model_tag'] == 'codet5_base':
        model_path = '{}/pretrained_models/codet5_base'.format(work_dir)
    elif arg_dict['model_tag'] == 'editing_small':
        model_path = '{}/editing_models/editing_small/checkpoints'.format(work_dir)
    elif arg_dict['model_tag'] == 'editing_base':
        model_path = '{}/editing_models/editing_base/checkpoints'.format(work_dir)

    args.do_train = True
    args.do_eval = True
    args.do_test = True
    args.do_eval_bleu = True
    args.save_last_checkpoints = True
    args.always_save_model = True
    
    args.task = arg_dict['task']
    args.sub_task = arg_dict['sub_task']
    args.model_type = model_type
    args.data_num = arg_dict['data_num']

    args.num_train_epochs = arg_dict['epoch']
    args.warmup_steps = 1000
    args.learning_rate = arg_dict['lr'] * 1e-5
    args.patience = arg_dict['patience']
    args.tokenizer_name = tokenizer
    args.tokenizer_path = tokenizer_path
    args.model_name_or_path = model_path
    args.output_dir = output_dir
    args.summary_dir = arg_dict['summary_dir']
    args.data_dir = '{}/data'.format(work_dir)
    args.cache_path = cache_dir
    args.res_dir = res_dir
    args.res_fn = arg_dict['res_fn']
    args.train_batch_size = arg_dict['batch_size']
    args.eval_batch_size = arg_dict['batch_size'] // 2
    args.max_source_length = arg_dict['src_len']
    args.max_target_length = arg_dict['trg_len']

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    task = 'summarize_intent'
    sub_task = 'java' # 'none'
    model_tag = 'editing_base'
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(task, sub_task, model_tag)
    
    arg_dict = {'task':task, 'sub_task':sub_task, 'model_tag':model_tag, 'data_num':-1, 'batch_size':bs,
                'lr':lr, 'src_len':src_len, 'trg_len':trg_len, 'patience':patience, 'epoch':epoch,
                'summary_dir':'tensorboard', 'res_fn':'{}/{}_{}'.format(work_dir, task, model_tag)}
    
    args = get_args(args, arg_dict)
    
    logger.info(args)
    set_seed(args)
    set_dist(args)
    main(args)

