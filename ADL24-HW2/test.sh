python test.py \
    --test_file data/public.jsonl \
    --output_dir saved_model_sample \
    --output_file submission.jsonl \
    --batch_size 1 \
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

# python eval.py -r data/public.jsonl -s saved_model/submission.jsonl
