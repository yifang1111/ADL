import json
import random
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate titles using the trained T5 model.")
    parser.add_argument('--test_file', type=str, default='test.jsonl', help='Path to the test data file.')
    parser.add_argument('--output_dir', type=str, default='saved_model', help='Directory of the saved trained model.')
    parser.add_argument('--output_file', type=str, default='submission.jsonl', help='Path to save the predictions.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generation.')
    parser.add_argument('--max_input_length', type=int, default=256, help='Maximum input sequence length.')
    parser.add_argument('--max_output_length', type=int, default=64, help='Maximum output sequence length.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling; if not set, greedy decoding or beam search is used.')
    parser.add_argument('--early_stopping', action='store_true', help='Stop when at least num_beams sentences are finished per batch.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling (top-p).')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature.')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty.')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='No repeat n-gram size.')

    return parser.parse_args()

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=256, prefix="summarize: "):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.prefix = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = self.prefix + item['maintext']
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'id': item['id']
        }

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_data = load_data(args.test_file)

    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataset = NewsDataset(test_data, tokenizer, max_input_length=args.max_input_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Titles"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sample_id = batch['id']

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_output_length,
                num_beams=args.num_beams if not args.do_sample else 1,
                do_sample=args.do_sample,
                top_k=args.top_k if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                temperature=args.temperature if args.do_sample else None,
                early_stopping=args.early_stopping,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )

            for i in range(len(sample_id)):
                generated_title = tokenizer.decode(
                    generated_ids[i],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                predictions.append({
                    'title': generated_title,
                    'id': sample_id[i]
                })

    with open(os.path.join(args.output_dir, args.output_file), 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"Predictions saved to {os.path.join(args.output_dir, args.output_file)}")

if __name__ == '__main__':
    main()
