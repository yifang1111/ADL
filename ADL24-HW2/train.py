import json
import random
import argparse
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from tqdm import tqdm
# from tw_rouge import get_rouge
from rouge import Rouge 
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR 


def parse_args():
    parser = argparse.ArgumentParser(description="Train a T5 model for title generation.")
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='Path to the training data file.')
    parser.add_argument('--model_name', type=str, default='t5-small', help='Model name or path.')
    parser.add_argument('--output_dir', type=str, default='saved_model', help='Directory to save the trained model.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps.')
    parser.add_argument('--max_input_length', type=int, default=256, help='Maximum input sequence length.')
    parser.add_argument('--max_output_length', type=int, default=64, help='Maximum output sequence length.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Ratio of validation set.')
    parser.add_argument('--step_size', type=int, default=2, help='Step size for learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Multiplicative factor for learning rate decay.')

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

def plot_learning_curves(num_epochs, train_losses, val_losses, rouge_scores, output_dir):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rouge_scores, label='Validation ROUGE-L F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE-L F1 Score (%)')
    plt.title('Validation ROUGE-L F1 Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'rouge_curve.png'))
    plt.close()

class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=256, max_output_length=64, prefix="summarize: "):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
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
        if 'title' in item:
            output_text = item['title']
            outputs = self.tokenizer.encode_plus(
                output_text,
                max_length=self.max_output_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': outputs['input_ids'].squeeze(),
                'title': output_text
            }
        else:
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

    data = load_data(args.train_file)
    train_data, valid_data = train_test_split(data, test_size=args.valid_ratio, random_state=args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = NewsDataset(train_data, tokenizer, max_input_length=args.max_input_length, max_output_length=args.max_output_length)
    valid_dataset = NewsDataset(valid_data, tokenizer, max_input_length=args.max_input_length, max_output_length=args.max_output_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_losses = []
    val_losses = []
    rouge_scores = []
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = GradScaler()
    rouge = Rouge()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

             # Mixed precision training
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / args.accumulation_steps  # Normalize loss

            # Scales loss for mixed precision training
            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                # Optimizer step with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (i + 1)})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step()

        torch.cuda.empty_cache() 

        model.eval()
        val_loss = 0
        references = []
        hypotheses = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_loss += loss.item()

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

                for i in range(len(labels)):
                    ref = batch['title'][i]
                    hyp = tokenizer.decode(
                        generated_ids[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    if not hyp.strip():
                        print(f"Warning: Empty hypothesis generated at index {i}")
                        print(f"Input text: {tokenizer.decode(input_ids[i], skip_special_tokens=True)}")
                        print(f"Reference title: {ref}")
                    else:
                        references.append(ref)
                        hypotheses.append(hyp)

        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        rouge_score = rouge.get_scores(hypotheses, references, avg=True)
        # rouge_score = get_rouge(hypotheses, references)
        rouge_scores.append(rouge_score['rouge-l']['f'] * 100)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation ROUGE-L F1 Score: {rouge_score['rouge-l']['f'] * 100:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Best model saved with validation loss {best_val_loss}")

    plot_learning_curves(args.num_epochs, train_losses, val_losses, rouge_scores, args.output_dir)

if __name__ == '__main__':
    main()
