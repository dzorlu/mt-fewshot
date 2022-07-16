
export WANDB_PROJECT=mt-fewshot
#deepspeed ./train.py \
./train.py \
    --model_name_or_path bigscience/T0_3B \
    --do_eval \
    --train_file ../data/train_answer_generation.json \
    --validation_file ../data/test_answer_generation.json \
    --text_column prompt \
    --target_column output \
    --output_dir output \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps 36\
    --gradient_checkpointing \
    --learning_rate 5e-05 \
    --predict_with_generate \
    --save_total_limit 10\
    --max_train_samples 2 \
    --max_eval_samples 2 \
    --evaluation_strategy steps\
    --num_train_epochs 1\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 200\
    --eval_steps 200\
    --logging_steps 25 \
    --report_to wandb

##--do_train \
#--deepspeed ../t0ds-config.json \
#--deepspeed t0ds-config.json \
# --fp16\