# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

# MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
# MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
# RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
# PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
# NODES=1 && echo NODES: ${NODES}
# NCCL_DEBUG=INFO

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

# echo -e "import nltk\nnltk.download('punkt')" > tmp.py
# python tmp.py

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/msg-test.jsonl \
#   --max_source_length 300 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input


# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-23400-4.74 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-23400-4.74 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-23400-4.74 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/msg-test.jsonl \
#   --max_source_length 300 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input


python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
  --model_type t5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_gen_tufano_newdata/checkpoints-48600-4.33 \
  --tokenizer_path ${mnt_dir}/Tufano/pytorch/TokenizerModel.model \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_newdata/checkpoints-48600-4.33 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_newdata/checkpoints-48600-4.33 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/Processor/newdata/msg-test.jsonl \
  --max_source_length 300 \
  --max_target_length 128 \
  --eval_batch_size 12 \
  --mask_rate 0.15 \
  --save_steps 1800 \
  --beam_size 10 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_codet5_newdata/checkpoints-25200-4.89 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_codet5_newdata/checkpoints-25200-4.89 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_codet5_newdata/checkpoints-25200-4.89 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/msg-test-s.jsonl \
#   --max_source_length 300 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_newdata/checkpoints-12600-5.34 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/msg-test.jsonl \
#   --max_source_length 300 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-28800-4.69 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-28800-4.69 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_scr/checkpoints-28800-4.69 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/newdata/msg-test.jsonl \
#   --max_source_length 300 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input

# ${mnt_dir}/PreViewer/saved_models_gen_codet5_osimp/checkpoints-14400-5.58 \
# ${mnt_dir}/PreViewer/saved_models_gen_shuai_new/checkpoints-3600-5.43 \
# ${mnt_dir}/PreViewer/saved_models_gen_codet5_simple_new/checkpoints-32400-5.2 \
# ${mnt_dir}/PreViewer/saved_models_gen_codet5_simple_new/checkpoints-30600-5.2 \
# ${mnt_dir}/PreViewer/saved_models_gen_codet5_2/checkpoints-3600-6.88 \
# ${mnt_dir}/PreViewer/saved_models_gen_shuai_2/checkpoints-5400-7.07 \

  # --model_type t5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_gen_tufano_newdata/checkpoints-16200-4.99 \
  # --tokenizer_path ${mnt_dir}/Tufano/pytorch/TokenizerModel.model \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_simple_new/checkpoints-10800-7.34 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_simple_new/checkpoints-10800-7.34 \

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type codet5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_codet5_simple/checkpoints-18000-4.14 \
#   --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_codet5_simple/checkpoints-18000-4.14 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_codet5_simple/checkpoints-18000-4.14 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/msg-test.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input
  

# python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_gen.py  \
#   --model_type t5 \
#   --add_lang_ids \
#   --train_epochs 30 \
#   --config_name ${mnt_dir}/PreViewer/saved_models_gen_tufano_simple/checkpoints-32400-4.34 \
#   --tokenizer_path ${mnt_dir}/Tufano/pytorch/TokenizerModel.model \
#   --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_simple/checkpoints-32400-4.34 \
#   --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_tufano_simple/checkpoints-32400-4.34 \
#   --output_dir ${mnt_dir}/PreViewer/empty \
#   --eval_file ${mnt_dir}/Processor/data/msg-test.jsonl \
#   --max_source_length 512 \
#   --max_target_length 128 \
#   --eval_batch_size 12 \
#   --mask_rate 0.15 \
#   --save_steps 1800 \
#   --beam_size 10 \
#   --log_steps 100 \
#   --train_steps 120000 \
#   --gpu_per_node=${PER_NODE_GPU} \
#   --node_index=${RANK} \
#   --seed 2233 \
#   --raw_input

