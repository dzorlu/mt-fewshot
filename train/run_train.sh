
export WANDB_PROJECT=mt-fewshot

deepspeed ./train.py \
    --model_name_or_path yuchenlin/BART0pp \
    --deepspeed t0ds-config.json \
    --do_eval \
    --do_train \
    --train_file ../data/train_response_generation.json \
    --validation_file ../data/test_response_generation.json \
    --text_column prompt \
    --target_column output \
    --output_dir /hdd/mt_few/run2 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_checkpointing \
    --learning_rate 5e-05 \
    --predict_with_generate \
    --save_total_limit 10\
    --evaluation_strategy steps\
    --num_train_epochs 1\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 2000\
    --eval_steps 2000\
    --logging_steps 25 \
    --fp16 \
    --report_to wandb


#
#
# --model_name_or_path bigscience/T0_3B \
#--gradient_accumulation_steps 36\
# deepspeed ./train.py \
# python ./train.py \
#     