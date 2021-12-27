# batch size 6 for 16 GB GPU

work_dir="/mnt/lzzz/PreViewer"

MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=2 && echo WORLD_SIZE: ${WORLD_SIZE}
# NCCL_SOCKET_IFNAME=ib0
NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${WORLD_SIZE} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_pre_training.py  \
  --model_type t5 \
  --add_lang_ids \
  --train_epochs 100 \
  --config_name t5-base \
  --tokenizer_path t5-base \
  --model_name_or_path t5-base \
  --output_dir ${work_dir}/saved_models \
  --train_path ${work_dir}/../../lzzz/processed \
  --max_source_length 512 \
  --max_target_length 256 \
  --train_batch_size 6 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 4
  --mask_rate 0.15 \
  --save_steps 5000 \
  --log_steps 100 \
  --train_steps 1000000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --from_scratch
