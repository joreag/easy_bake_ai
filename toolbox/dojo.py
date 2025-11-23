import torch
import torch.nn as nn
import argparse
import json
import os
import sys
import re
import random
from tqdm import tqdm

# --- Pathing Setup ---
TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLBOX_DIR)
sys.path.append(PROJECT_ROOT)

# --- Architecture Imports ---
from src.hcts_transformer_architecture import HCTS_Transformer
from src.pascal_guided_transformer import PascalGuidedTransformer

class HCTSDojo:
    def __init__(self, model_path, dataset_path, vocab_path, remedial_lr=5e-5, remedial_epochs=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.remedial_lr = remedial_lr
        self.remedial_epochs = remedial_epochs
        
        print("\n" + "#"*70 + "\n###      HCTS DOJO: Active Reinforcement Laboratory      ###\n" + "#"*70 + "\n")

        # 1. Load Vocab
        print(f"--- Loading Vocabulary: {vocab_path}")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_idx = self.vocab.get('[PAD]', 0)
        self.sos_idx = self.vocab.get('[CLS]', 1)
        self.eos_idx = self.vocab.get('[SEP]', 2)

        # 2. Load Checkpoint & Config
        print(f"--- Loading Brain: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # 3. Instantiate Correct Architecture
        self.model = self._build_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # 4. Optimizer for Remedial Lessons
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.remedial_lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        # 5. Load Dataset
        self.dataset = self._load_dataset(dataset_path)
        print(f"--- Dojo Initialized. {len(self.dataset)} Flashcards loaded. ---")

    def _build_model(self, config):
        """Reconstructs the exact model used in the build."""
        arch_type = config.get('arch_type', 'standard')
        vocab_size = len(self.vocab)
        
        # Fallbacks for older builds that might miss keys
        d_model = int(config.get('d_model', 512))
        nhead = int(config.get('nhead', 8))
        enc_layers = int(config.get('num_encoder_layers', 6))
        dec_layers = int(config.get('num_decoder_layers', 6))
        dim_ff = int(config.get('dim_feedforward', 2048))

        if arch_type == 'pascal':
            print("   -> Architecture Detected: Pascal-Guided (Model Z)")
            # Reconstruct stack config
            enc_per = max(1, enc_layers // 3)
            dec_per = max(1, dec_layers // 3)
            stack_config = [
                {'name': 'Syntax', 'encoder_layers': enc_per, 'decoder_layers': dec_per},
                {'name': 'Semantic', 'encoder_layers': enc_per, 'decoder_layers': dec_per},
                {'name': 'Reasoning', 'encoder_layers': enc_per, 'decoder_layers': dec_per},
            ]
            return PascalGuidedTransformer(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                stack_config=stack_config, dim_feedforward=dim_ff, dropout=0.1, pad_idx=self.pad_idx
            )
        else:
            print("   -> Architecture Detected: Standard HCTS")
            return HCTS_Transformer(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
                dim_feedforward=dim_ff, pad_idx=self.pad_idx
            )

    def _load_dataset(self, path):
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Normalize keys
                        q = item.get('question') or item.get('prompt')
                        a = item.get('answer') or item.get('output')
                        if q and a:
                            data.append({'question': q, 'answer': a})
        except Exception as e:
            sys.exit(f"FATAL: Could not load flashcards: {e}")
        return data

    def _tokenize(self, text):
        token_ids = [self.vocab.get(c, self.vocab.get('[UNK]', 0)) for c in str(text).lower()]
        token_ids = [self.sos_idx] + token_ids + [self.eos_idx]
        # Pad to max length from config or default 256
        max_len = int(self.config.get('max_seq_length', 256))
        
        # Create tensor directly
        tensor = torch.full((1, max_len), self.pad_idx, dtype=torch.long, device=self.device)
        length = min(len(token_ids), max_len)
        tensor[0, :length] = torch.tensor(token_ids[:length], device=self.device)
        return tensor

    def _generate(self, src):
        self.model.eval()
        tgt = torch.tensor([[self.sos_idx]], device=self.device)
        
        # Simple greedy generation
        for _ in range(150):
            with torch.no_grad():
                logits = self.model(src, tgt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break
        
        # Decode
        tokens = [self.inv_vocab.get(i, '') for i in tgt.squeeze().tolist()]
        # Remove special tokens
        return "".join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']])

    def _remedial_lesson(self, question, answer):
        """Runs a mini training loop on just this one card."""
        print(f"      >>> Initiating Cognitive Jolt (Remedial Training)...")
        self.model.train()
        
        src = self._tokenize(question)
        tgt = self._tokenize(answer)
        
        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        
        pbar = tqdm(range(self.remedial_epochs), desc="      Learning", leave=False)
        for _ in pbar:
            self.optimizer.zero_grad()
            pred = self.model(src, tgt_input)
            loss = self.criterion(pred.reshape(-1, len(self.vocab)), tgt_y.reshape(-1))
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    def _normalize(self, text):
        return re.sub(r'[\W_]+', '', text.lower())

    def begin_training(self):
        cards = self.dataset
        pass_num = 0
        
        while cards:
            pass_num += 1
            print(f"\n=== PASS {pass_num} | Cards Remaining: {len(cards)} ===")
            failed_cards = []
            random.shuffle(cards)
            
            for i, card in enumerate(cards):
                q, a = card['question'], card['answer']
                
                # Check Knowledge
                src = self._tokenize(q)
                response = self._generate(src)
                
                # Compare
                if self._normalize(response) != self._normalize(a):
                    print(f"\n[FAIL] Card #{i+1}")
                    print(f"   Q: {q}")
                    print(f"   Expected: {a}")
                    print(f"   Got:      {response}")
                    
                    # Fix
                    self._remedial_lesson(q, a)
                    failed_cards.append(card)
                else:
                    # Optional: Print success dots to show progress
                    print(".", end="", flush=True)

            if not failed_cards:
                print(f"\n\n*** MASTERY ACHIEVED in {pass_num} passes! ***")
                break
            
            cards = failed_cards
            print(f"\n\n--- Saving Progress to Checkpoint ---")
            torch.save({
                'model_state_dict': self.model.state_dict(), 
                'config': self.config
            }, 'dojo_checkpoint.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HCTS Reinforcement Dojo")
    parser.add_argument('--build', required=True, help="Name of the build folder (e.g. 'my_first_forge')")
    parser.add_argument('--dataset', help="Optional: Path to specific jsonl file. Defaults to build's dataset.")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=5, help="Remedial epochs per failed card")
    
    args = parser.parse_args()
    
    # Construct paths based on Build Name
    build_path = os.path.join(PROJECT_ROOT, 'builds', args.build)
    model_path = os.path.join(build_path, 'model.pth')
    vocab_path = os.path.join(build_path, 'vocab.json')
    
    # Default to the dataset generated during the build if none provided
    dataset_path = args.dataset if args.dataset else os.path.join(build_path, 'grounding_dataset.jsonl')
    
    if not os.path.exists(model_path):
        sys.exit(f"Build not found at {build_path}")

    dojo = HCTSDojo(
        model_path=model_path,
        dataset_path=dataset_path,
        vocab_path=vocab_path,
        remedial_lr=args.lr,
        remedial_epochs=args.epochs
    )
    dojo.begin_training()