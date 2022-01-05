# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
# NCCL_SOCKET_IFNAME=ib0
NCCL_DEBUG=INFO

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${WORLD_SIZE} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../eval_pret.py  \
RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=23333 \
python ../eval_pret.py \
  --model_type t5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name t5-base \
  --tokenizer_path t5-base \
  --model_name_or_path t5-base \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/saved_models/checkpoints-80000 \
  --output_dir not_used \
  --eval_file ${mnt_dir}/processed/chunk_8.jsonl \
  --max_source_length 512 \
  --max_target_length 256 \
  --eval_batch_size 6 \
  --log_steps 100 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \



# 50000: step 700: loss 0.848, ppl 2.334
# 80000: step 700: loss 0.828, ppl 2.289
# 100000: step 700: loss 0.795, ppl 2.214
# 115000: step 700: loss 0.803, ppl 2.232