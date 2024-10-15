python train_mc.py \
  --model_name_or_path hfl/chinese-macbert-base \
  --pad_to_max_length \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 2 \
  --output_dir mc_macbert