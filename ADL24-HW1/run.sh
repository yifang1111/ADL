python test_mc.py \
  --model_name_or_path mc_macbert \
  --pad_to_max_length \
  --max_seq_length 512 \
  --output_dir mc_macbert \
  --context_file "${1}" \
  --test_file "${2}"

python convert_test.py --test_file "${2}" --mc_ckpt "mc_macbert" --output_file "new_test_macbert.json"

python test_qa.py \
  --model_name_or_path qa_macbert \
  --pad_to_max_length \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir qa_macbert \
  --context_file "${1}" \
  --test_file "new_test_macbert.json" \
  --output_file "${3}" \

# bash run.sh context.json test.json qa_macbert/prediction.csv
