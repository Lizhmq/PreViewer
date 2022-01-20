# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

# MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
# MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
# RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
# PER_NODE_GPU=8 && echo PER_NODE_GPU: ${PER_NODE_GPU}
# WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
# # NCCL_SOCKET_IFNAME=ib0
# NCCL_DEBUG=INFO

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
# NCCL_SOCKET_IFNAME=ib0
NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${WORLD_SIZE} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_gendiff.py  \
  --model_type t5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name t5-base \
  --tokenizer_path t5-base \
  --model_name_or_path t5-base \
  --output_dir ${mnt_dir}/PreViewer/saved_models \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/saved_models/checkpoints-170000 \
  --train_path ${mnt_dir}/processed \
  --eval_file ${mnt_dir}/processed/chunk_8.jsonl \
  --no_cls_head \
  --max_source_length 512 \
  --max_target_length 256 \
  --train_batch_size 6 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2 \
  --mask_rate 0.15 \
  --save_steps 8000 \
  --log_steps 100 \
  --train_steps 150000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
