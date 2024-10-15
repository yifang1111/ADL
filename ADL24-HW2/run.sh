python test.py \
    --test_file "${1}" \
    --output_dir saved_model \
    --output_file "${2}" \
    --batch_size 1 \
    --max_input_length 256 \
    --max_output_length 64 \
    --seed 42 \
    --num_beams 6 \
    --early_stopping \
    --repetition_penalty 1.2 \
    --no_repeat_ngram_size 2

# bash ./run.sh data/public.jsonl submission.jsonl
# python eval.py -r data/public.jsonl -s saved_model/submission.jsonl
