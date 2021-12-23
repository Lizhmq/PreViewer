WORKDIR="/ceph-jd/pub/jupyter/lizhuo/notebooks/examples/Editing-main"
export PYTHONPATH=${WORKDIR}/code

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}
GRAD_ACC_STEPS=${16}


if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${WORKDIR}/${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${WORKDIR}/${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=roberta-base
  MODEL_PATH=${WORKDIR}/pretrained_models/codet5_small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=roberta-base
  MODEL_PATH=${WORKDIR}/pretrained_models/codet5_base
elif [[ $MODEL_TAG == editing_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=roberta-base
  MODEL_PATH=${WORKDIR}/editing_models/editing_small/checkpoints
elif [[ $MODEL_TAG == editing_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=roberta-base
  MODEL_PATH=${WORKDIR}/editing_models/editing_base/checkpoints
fi

if [[ ${TASK} == 'clone' ]]; then
  RUN_FN=../run_clone.py
else
  RUN_FN=../run_gen.py
fi


CUDA_VISIBLE_DEVICES=${GPU} \
  python3 ${RUN_FN} ${MULTI_TASK_AUG} \
  --do_test --do_train --do_eval --do_eval_bleu --save_last_checkpoints --always_save_model \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER} --tokenizer_path=${WORKDIR}/tokenizer/salesforce \
  --model_name_or_path=${MODEL_PATH} --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --data_dir ${WORKDIR}/data  --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS}
  2>&1 | tee ${LOG}
