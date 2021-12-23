# --nproc_per_node GPU的数量
# --train_filename 训练数据的路径
# --train_batch_size 每个GPU上的batch size，在16GB的V100上最大是5
work_dir="/mnt/lzzz/PreViewer"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch --nproc_per_node 4 ../run_pre_training.py  \
  --model_type codet5  --train_filename ${work_dir}/data/pre-training/ \
  --num_train_epochs 10 --learning_rate 2e-5  \
  --tokenizer_name=roberta-base --tokenizer_path=${work_dir}/tokenizer/salesforce \
  --model_name_or_path=${work_dir}/pretrained_models/codet5_base --output_dir ${work_dir}/editing_models/codet5_base \
  --train_batch_size 1 --max_source_length 512 --max_target_length 256

