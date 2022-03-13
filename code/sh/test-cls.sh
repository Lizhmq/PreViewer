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


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_nodec/checkpoints-3600-0.729 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_nodec/checkpoints-3600-0.729 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_nodec/checkpoints-3600-0.729 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/cls-test.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_noenc/checkpoints-3600-0.737 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_noenc/checkpoints-3600-0.737 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_noenc/checkpoints-3600-0.737 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/cls-test.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_nomsg/checkpoints-3600-0.736 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_nomsg/checkpoints-3600-0.736 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_nomsg/checkpoints-3600-0.736 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/cls-test.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233



# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs/checkpoints-1200-0.724 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs/checkpoints-1200-0.724 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs/checkpoints-1200-0.724 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-cs.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs_multi/checkpoints-800-0.731 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs_multi/checkpoints-800-0.731 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs_multi/checkpoints-800-0.731 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-cs.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233



# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_java/checkpoints-2800-0.748 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_java/checkpoints-2800-0.748 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_java/checkpoints-2800-0.748 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-java.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_java_multi/checkpoints-2800-0.752 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_java_multi/checkpoints-2800-0.752 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_java_multi/checkpoints-2800-0.752 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-java.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233




# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb/checkpoints-2000-0.792 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb/checkpoints-2000-0.792 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb/checkpoints-2000-0.792 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-rb.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb_multi/checkpoints-4400-0.802 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb_multi/checkpoints-4400-0.802 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb_multi/checkpoints-4400-0.802 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/cls-test-rb.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 16 \
#   --mask_rate 0.15 \
#   --save_steps 4000 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/Processor/newdata/cls-test-rb.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/Processor/newdata/cls-test-java.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233


python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_cls_shuai/checkpoints-3600-0.740 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/Processor/newdata/cls-test-cs.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233