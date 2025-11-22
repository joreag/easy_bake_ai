import json
import torch
import torch.nn as nn
import argparse
import os
import time
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests

# Boilerplate to find 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.hcts_transformer_architecture import HCTS_Transformer
from src.pascal_guided_transformer import PascalGuidedTransformer

class GroundingDataset(Dataset):
    def __init__(self, dataset_filepath, vocab_map, max_length=256):
        self.vocab = vocab_map; self.max_length = max_length
        self.pad_idx = self.vocab.get('[PAD]', 0); self.sos_idx = self.vocab.get('[CLS]', 1); self.eos_idx = self.vocab.get('[SEP]', 2)
        try:
            with open(dataset_filepath, 'r', encoding='utf-8') as f: self.pairs = [json.loads(line) for line in f]
        except Exception as e: print(f"[ERROR] Failed to load dataset: {e}"); self.pairs = []
    def __len__(self): return len(self.pairs)
    def _tokenize(self, text):
        token_ids = [self.vocab.get(char, self.vocab.get('[UNK]',0)) for char in str(text).lower()]
        token_ids = [self.sos_idx] + token_ids + [self.eos_idx]
        padding_len = self.max_length - len(token_ids)
        token_ids += [self.pad_idx] * max(0, padding_len)
        return torch.tensor(token_ids[:self.max_length])
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return self._tokenize(pair['question']), self._tokenize(pair['answer'])

def train_model(config: dict):
    print("--- Starting Training Sub-process ---")
    with open(config['vocab_path'], 'r') as f: vocab = json.load(f)
    VOCAB_SIZE, PAD_IDX = len(vocab), vocab['[PAD]']
    
    dataset = GroundingDataset(config['dataset_path'], vocab, config['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Training on: {device.upper()}")

    # --- ARCHITECTURE SELECTION ---
    arch_type = config.get('arch_type', 'standard')
    
    if arch_type == 'pascal':
        print("   -> Architecture: Pascal-Guided Transformer (Model Z)")
        # Define the Hierarchy (Syntax -> Semantic -> Reasoning)
        # We split the total layers requested across the 3 stacks
        enc_per_stack = max(1, config['num_encoder_layers'] // 3)
        dec_per_stack = max(1, config['num_decoder_layers'] // 3)
        
        stack_config = [
            {'name': 'Syntax', 'encoder_layers': enc_per_stack, 'decoder_layers': dec_per_stack},
            {'name': 'Semantic', 'encoder_layers': enc_per_stack, 'decoder_layers': dec_per_stack},
            {'name': 'Reasoning', 'encoder_layers': enc_per_stack, 'decoder_layers': dec_per_stack},
        ]
        
        model = PascalGuidedTransformer(
            vocab_size=VOCAB_SIZE, d_model=config['d_model'], nhead=config['nhead'],
            stack_config=stack_config, dim_feedforward=config['dim_feedforward'], 
            dropout=0.1, pad_idx=PAD_IDX
        )
    else:
        print("   -> Architecture: Standard HCTS-Transformer v1")
        model = HCTS_Transformer(
            vocab_size=VOCAB_SIZE, d_model=config['d_model'], nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'], num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'], pad_idx=PAD_IDX
        )

    model.to(device)
    print(f"   -> Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    callback_url = config.get('callback_url')
    callback_failed = False
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train(); total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")
        
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_y = tgt[:, :-1], tgt[:, 1:]
            optimizer.zero_grad()
            prediction = model(src, tgt_input)
            loss = criterion(prediction.view(-1, VOCAB_SIZE), tgt_y.reshape(-1))
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            
            current_avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({'Loss': f"{current_avg_loss:.4f}"})

            # --- TELEMETRY BLOCK ---
            if callback_url and not callback_failed:
                try:
                    # Safe extraction of speed (it/s)
                    speed = pbar.format_dict.get('rate')
                    if speed is None: speed = 0.0
                    
                    payload = {
                        "epoch": epoch + 1, 
                        "total_epochs": config['epochs'],
                        "loss": float(current_avg_loss), # Send avg loss, not instantaneous
                        "speed": float(speed)
                    }
                    # Send EVERY step (removed % 10 restriction) for fluid UI
                    requests.post(callback_url, json=payload, timeout=0.1)
                except requests.exceptions.RequestException:
                    # Fails silently to avoid spamming logs, but disables future attempts
                    print("\n[WARNING] Dashboard connection failed. Telemetry disabled.")
                    callback_failed = True

        print(f"Epoch [{epoch+1}/{config['epochs']}] Complete, Loss: {total_loss / len(dataloader):.4f}")

    print(f"\n--- Training Complete in {(time.time() - start_time) / 60:.2f} minutes ---")
    
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    final_model_path = config['output_model']
    torch.save({'model_state_dict': model_state, 'config': config}, final_model_path)
    print(f"Educated HCTS model saved to '{final_model_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a foundational HCTS-Transformer model.")
    parser.add_argument('--dataset-path', required=True); parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-model', required=True)
    parser.add_argument('--callback-url', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4); parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=512); parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6); parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=2048)
    parser.add_argument('--arch-type', type=str, default='standard') 
    args = parser.parse_args()
    train_model(vars(args))