python train.py \
    --train_file data/train.jsonl \
    --model_name google/mt5-small \
    --output_dir saved_model \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --accumulation_steps 2 \
    --max_input_length 256 \
    --max_output_length 64 \
    --seed 42 \
    --num_beams 6 \
    --early_stopping \
    --repetition_penalty 1.2 \
    --no_repeat_ngram_size 2

    # --do_sample \
    # --top_k 50 \
    # --top_p 0.95 \
    # --temperature 0.7 \

