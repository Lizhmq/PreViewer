import os
import argparse
# import hf_env
# hf_env.set_env('202105')
# import hfai

def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, gradient_accumulation_steps):
    cmd_str = 'bash exp_with_args.sh %s %s %s %s %d %d %d %d %d %d %d %d %s %s %s %d' % \
              (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
               warmup, model_dir, summary_dir, res_fn, gradient_accumulation_steps)
    return cmd_str


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
            src_len = 130 # original: 130, 50(commit msg) + 130(input code) = 180
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 240 # original: 240
            trg_len = 240
        epoch = 50
        patience = 5
    elif task == 'refine_intent': # add by lijia
        if sub_task == 'small':
            src_len = 180 # original: 130, 50(commit msg) + 130(input code) = 180
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 290 # original: 240
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

    if 'codet5_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = 64 # 64
    else:
        bs = 32 # original: 32, refine small valid in 16GB: 16
        if task == 'translate':
            bs = 25
        elif task == 'summarize':
            bs = 48 # original: 48, valid in 16GB: 8
    lr = 5
    if task == 'concode':
        lr = 10
    elif task == 'defect':
        lr = 2
    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag), 
                      gradient_accumulation_steps=args.gradient_accumulation_steps)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task == 'summarize':
        sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
    elif task == 'translate':
        sub_tasks = ['java-cs', 'cs-java']
    elif task in ['refine', 'refine_intent']:
        sub_tasks = ['small', 'medium']
    elif task in ['concode', 'defect', 'clone']:
        sub_tasks = ['none']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_small',
                        choices=['roberta', 'codebert', 'codet5_small', 'codet5_base', 'editing_small', 'editing_base'])
    parser.add_argument("--task", type=str, default='summarize', choices=['summarize', 'concode', 'translate',
                                                                          'refine', 'defect', 'clone', 'refine_intent'])
    parser.add_argument("--sub_task", type=str, default='none')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=str, default='1', help='index of the gpu to use in a cluster')
    parser.add_argument("--gradient_accumulation_steps", type=int, default='1')
    
    args = parser.parse_args()
    
    args.model_tag = 'editing_base'
    args.task = 'refine'
    args.sub_task = 'medium'
    args.gpu = '0,1,2,3,4,5,6,7'
    
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    assert args.sub_task in get_sub_tasks(args.task)
    
    run_one_exp(args)
