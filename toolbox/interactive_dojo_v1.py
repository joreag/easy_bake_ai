import torch
import torch.nn as nn
import argparse
import json
import os
import sys
import re
from torch.utils.data import DataLoader, Dataset

# --- Boilerplate ---
TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLBOX_DIR)
sys.path.append(PROJECT_ROOT)

from src.hcts_transformer_architecture import HCTS_Transformer
from src.pascal_guided_transformer import PascalGuidedTransformer

class SimpleDataset(Dataset):
    def __init__(self, data, vocab, max_len=256):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.get('[PAD]', 0)
        self.sos_idx = vocab.get('[CLS]', 1)
        self.eos_idx = vocab.get('[SEP]', 2)

    def __len__(self): return len(self.data)
    
    def _tokenize(self, text):
        ids = [self.vocab.get(c, self.vocab.get('[UNK]', 0)) for c in str(text).lower()]
        ids = [self.sos_idx] + ids + [self.eos_idx]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        else: ids += [self.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self._tokenize(item['input']), self._tokenize(item['output'])

class InteractiveDojoV1:
    def __init__(self, model_path, vocab_path, lr=1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n" + "#"*60)
        print("###    HCTS INTERACTIVE DOJO (Standard Edition)      ###")
        print("#"*60 + "\n")
        
        self.model_path = model_path
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            self.inv_vocab = {i: c for c, i in self.vocab.items()}
            
        self.pad_idx = self.vocab.get('[PAD]', 0)
        self.sos_idx = self.vocab.get('[CLS]', 1)
        self.eos_idx = self.vocab.get('[SEP]', 2)
        
        print(f"--- Loading Brain from '{model_path}' ---")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Architecture Factory
        arch = config.get('arch_type', 'standard')
        vocab_size = len(self.vocab)
        d_model = int(config.get('d_model', 512))
        nhead = int(config.get('nhead', 8))
        dim_ff = int(config.get('dim_feedforward', 2048))
        
        if arch == 'pascal':
            print("   -> Type: Pascal-Guided (Model Z)")
            enc_layers = int(config.get('num_encoder_layers', 6))
            dec_layers = int(config.get('num_decoder_layers', 6))
            # Reconstruct stack config simply
            per_stack = max(1, enc_layers // 3)
            stack_config = [
                {'name': 'Syn', 'encoder_layers': per_stack, 'decoder_layers': max(1, dec_layers//3)},
                {'name': 'Sem', 'encoder_layers': per_stack, 'decoder_layers': max(1, dec_layers//3)},
                {'name': 'Reas', 'encoder_layers': per_stack, 'decoder_layers': max(1, dec_layers//3)},
            ]
            self.model = PascalGuidedTransformer(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                stack_config=stack_config, dim_feedforward=dim_ff, dropout=0.1, pad_idx=self.pad_idx
            )
        else:
            print("   -> Type: Standard HCTS v1")
            self.model = HCTS_Transformer(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                num_encoder_layers=int(config.get('num_encoder_layers', 6)),
                num_decoder_layers=int(config.get('num_decoder_layers', 6)),
                dim_feedforward=dim_ff, pad_idx=self.pad_idx
            )
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        print("--- Ready. ---")

    def _generate(self, text):
        self.model.eval()
        tokens = [self.vocab.get(c, 0) for c in text.lower()]
        src = torch.tensor([self.sos_idx] + tokens + [self.eos_idx], device=self.device).unsqueeze(0)
        
        if hasattr(self.model, 'generate'):
            out = self.model.generate(src, self.sos_idx, self.eos_idx, max_len=128)
            ids = out.squeeze().tolist()
        else:
            # Manual Greedy loop for standard models
            tgt = torch.tensor([[self.sos_idx]], device=self.device)
            for _ in range(128):
                logits = self.model(src, tgt)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == self.eos_idx: break
            ids = tgt.squeeze().tolist()
            
        chars = [self.inv_vocab.get(i, '') for i in ids if i not in [self.sos_idx, self.eos_idx, self.pad_idx]]
        return "".join(chars)

    def _teach(self, q, a):
        print(f"\n   >>> TEACHING: '{q}' -> '{a}'")
        self.model.train()
        dataset = SimpleDataset([{'input': q, 'output': a}], self.vocab)
        loader = DataLoader(dataset, batch_size=1)
        
        for _ in range(5): # 5 micro-epochs
            for src, tgt in loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                self.optimizer.zero_grad()
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                preds = self.model(src, tgt_input)
                loss = self.criterion(preds.reshape(-1, len(self.vocab)), tgt_output.reshape(-1))
                loss.backward()
                self.optimizer.step()
        print("   >>> Learned.")
        torch.save({'model_state_dict': self.model.state_dict(), 'config': self.model_path}, self.model_path)

    def chat(self):
        print("\nType 'exit' to quit. Type 'teach' to correct last answer.")
        last_q = ""
        while True:
            try:
                user = input("\nYOU > ")
                if user.lower() in ['exit', 'quit']: break
                if user.lower() == 'teach' and last_q:
                    ans = input("   Correct Answer > ")
                    self._teach(last_q, ans)
                    continue
                
                resp = self._generate(user)
                print(f"AI  > {resp}")
                last_q = user
            except KeyboardInterrupt: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    args = parser.parse_args()
    
    # Auto-find paths
    build_dir = os.path.join(PROJECT_ROOT, 'builds', args.build)
    InteractiveDojoV1(
        os.path.join(build_dir, 'model.pth'),
        os.path.join(build_dir, 'vocab.json')
    ).chat()