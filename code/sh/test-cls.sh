# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

# MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
# MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
# RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
# PER_NODE_GPU=2 && echo PER_NODE_GPU: ${PER_NODE_GPU}
# NODES=1 && echo NODES: ${NODES}
# NCCL_DEBUG=INFO

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_cls_codet5_simple_3e-4/checkpoints-3600-0.619 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_codet5_simple_3e-4/checkpoints-3600-0.619 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/Processor/data/cls-test.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_codet5_simple_3e-4/checkpoints-3600-0.619 \
  --raw_input

# ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
# ${mnt_dir}/Tufano/pytorch \
# ${mnt_dir}/PreViewer/saved_models_cls_codet5_shuai_1e-3_dp0.3/checkpoints-3600-0.689 \
# ${mnt_dir}/PreViewer/saved_models_cls_codet5_shuai/checkpoints-4000-0.6830665813060179 \
# ${mnt_dir}/PreViewer/saved_models_cls_codet5/checkpoints-4000-0.6872599231754162 \
# ${mnt_dir}/PreViewer/saved_models_cls_codet5_simple_3e-4/checkpoints-3600-0.619 \
# ${mnt_dir}/PreViewer/saved_models_cls_tufano/checkpoints-4000-0.6743277848911652 \
