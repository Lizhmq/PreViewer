# batch size 12 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

MASTER_HOST=${MASTER_IP} && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=${MASTER_PORT} && echo MASTER_PORT: ${MASTER_PORT}
RANK=${OMPI_COMM_WORLD_RANK} && echo RANK: ${RANK}
PER_NODE_GPU=2 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

# MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
# MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
# RANK=0 && echo RANK: ${RANK}
# PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
# WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
# NODES=1 && echo NODES: ${NODES}
# NCCL_DEBUG=INFO

echo -e "import nltk\nnltk.download('punkt')" > tmp.py
python tmp.py

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_cls.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --output_dir ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs_multi \
  --train_filename ${mnt_dir}/Processor/newdata/cls-train-cs.jsonl \
  --dev_filename ${mnt_dir}/Processor/newdata/cls-valid-cs.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 12 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 400 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  # --raw_input

  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_cs/checkpoints-8000-6.06 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_cs/checkpoints-8000-6.06 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_cs/checkpoints-8000-6.06 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs \
  # --train_filename ${mnt_dir}/Processor/newdata/cls-train-cs.jsonl \
  # --dev_filename ${mnt_dir}/Processor/newdata/cls-valid-cs.jsonl \

  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_java/checkpoints-17000-5.71 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_java/checkpoints-17000-5.71 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_java/checkpoints-17000-5.71 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_cls_shuai_java \
  # --train_filename ${mnt_dir}/Processor/newdata/cls-train-java.jsonl \
  # --dev_filename ${mnt_dir}/Processor/newdata/cls-valid-java.jsonl \


  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_rb/checkpoints-7000-5.99 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_rb/checkpoints-7000-5.99 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_shuai/monolingual/save_codet5_rb/checkpoints-7000-5.99 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_cls_shuai_rb \
  # --train_filename ${mnt_dir}/Processor/newdata/cls-train-rb.jsonl \
  # --dev_filename ${mnt_dir}/Processor/newdata/cls-valid-rb.jsonl \


  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_cls_shuai_cs_multi \