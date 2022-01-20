# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NODES=1 && echo NODES: ${NODES}
# NCCL_SOCKET_IFNAME=ib0
NCCL_DEBUG=INFO

# MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
# MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
# RANK=0 && echo RANK: ${RANK}
# PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
# WORLD_SIZE=4 && echo WORLD_SIZE: ${WORLD_SIZE}
# NODES=1 && echo NODES: ${NODES}
# # NCCL_SOCKET_IFNAME=ib0
# NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_msggen.py  \
  --model_type t5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/pretrained_models/t5 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/t5 \
  --model_name_or_path ${mnt_dir}/PreViewer/pretrained_models/t5 \
  --output_dir ${mnt_dir}/PreViewer/saved_models_msggen_origt5 \
  --train_path ${mnt_dir}/processed \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 6 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 10000 \
  --log_steps 100 \
  --train_steps 300000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/saved_models_t5/checkpoints-170000 \

# 3e-5  4 * 6 * 3