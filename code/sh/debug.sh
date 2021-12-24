work_dir="/home/v-zhuoli1/wspace/PreViewer"


CUDA_VISIBLE_DEVICES=0 \
  python ../run_pre_training.py \
    --model_type codet5 \
    --add_lang_ids \
    --train_epochs 10 \
    --config_name Salesforce/codet5-base \
    --tokenizer_path Salesforce/codet5-base \
    --model_name_or_path Salesforce/codet5-base \
    --output_dir ${work_dir}/saved_models \
    --train_path ${work_dir}/../../lzzz/processed \
    --max_source_length 512 \
    --max_target_length 256 \
    --train_batch_size 4 \
    --learning_rate 2e-5 \
    --mask_rate 0.15 \
    --save_steps 50 \
    --log_steps 5 \
    --train_steps 200 \
    --seed 2233 \
    # --from_scratch \

