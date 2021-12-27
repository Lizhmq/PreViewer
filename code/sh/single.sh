work_dir="/home/v-zhuoli1/wspace/PreViewer"

MASTER_ADDR=localhost
MASTER_PORT=29500
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
# NCCL_SOCKET_IFNAME=ib0
NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0 RANK=0 PER_NODE_GPU=1 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=2950  \
python ../run_pre_training.py  \
    --model_type t5 \
    --add_lang_ids \
    --train_epochs 10 \
    --config_name t5-base \
    --tokenizer_path t5-base \
    --model_name_or_path t5-base \
    --output_dir ${work_dir}/saved_models \
    --train_path ${work_dir}/../../lzzz/processed \
    --max_source_length 512 \
    --max_target_length 256 \
    --train_batch_size 6 \
    --learning_rate 2e-5 \
    --mask_rate 0.15 \
    --save_steps 50 \
    --log_steps 5 \
    --train_steps 1000000 \
    --seed 2233 \
    --from_scratch \
    --debug
