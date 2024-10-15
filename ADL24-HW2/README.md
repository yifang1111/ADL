# NTU ADL 2024 Fall HW2
Train a T5 model for Chinese News Summarization

## Dataset
[download link](https://drive.google.com/drive/folders/1PMa25MwIVWTRhUtkWTfBFgqbqmGAxG2-?usp=drive_link)  
[backup download link](https://drive.google.com/drive/folders/1vXG_upPXnPTBhN7ewgA0HS-LxQLj8Zl1?usp=drive_link)

## Environment
```shell
make
conda activate adl-hw2-2024
pip install -r requirements.txt
pip install -e tw_rouge # install tw_rouge for evaluation
```

## Fix T5 FP16 Training (Optional)
```shell
git clone https://github.com/huggingface/transformers.git
git checkout t5-fp16-no-nans
cd transformers
pip install -e .
```

## Download models, tokenizers
```shell
bash download.sh
```

## Training
```shell
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
```

## Inference
```shell
# usage: bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
bash run.sh data/public.jsonl submission.jsonl
```

## Evaluation
```shell
# usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]
python eval.py -r data/public.jsonl -s saved_model/submission.jsonl
```