import sys
import os
import argparse
import torch
import json

# --- PATHING MAGIC ---
# Allow importing from 'src' which is one level up
TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLBOX_DIR)
sys.path.append(PROJECT_ROOT)

from src.hcts_transformer_architecture import HCTS_Transformer

def load_model_and_vocab(build_name):
    build_dir = os.path.join(PROJECT_ROOT, 'builds', build_name)
    model_path = os.path.join(build_dir, 'model.pth')
    vocab_path = os.path.join(build_dir, 'vocab.json')

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"FATAL: Could not find build '{build_name}'. Check your 'builds' folder.")
        sys.exit(1)

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Load Checkpoint to get config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Instantiate Model
    model = HCTS_Transformer(
        vocab_size=len(vocab),
        d_model=int(config.get('d_model', 512)),
        nhead=int(config.get('nhead', 8)),
        num_encoder_layers=int(config.get('num_encoder_layers', 6)),
        num_decoder_layers=int(config.get('num_decoder_layers', 6)),
        dim_feedforward=int(config.get('dim_feedforward', 2048)),
        pad_idx=vocab.get('[PAD]', 0)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, device

def generate_response(model, vocab, text, device, max_len=150):
    rev_vocab = {v: k for k, v in vocab.items()}
    
    # Tokenize
    token_ids = [vocab.get(char, vocab.get('[UNK]', 0)) for char in text.lower()]
    token_ids = [vocab.get('[CLS]', 1)] + token_ids + [vocab.get('[SEP]', 2)]
    src = torch.tensor(token_ids).unsqueeze(0).to(device)
    
    # Generate
    sos_idx = vocab.get('[CLS]', 1)
    eos_idx = vocab.get('[SEP]', 2)
    
    output_ids = model.generate(src, sos_idx, eos_idx, max_len)
    
    # Decode
    output_tokens = []
    for idx in output_ids.squeeze(0).cpu().numpy():
        if idx == eos_idx: break
        if idx not in [sos_idx, vocab.get('[PAD]', 0)]:
            output_tokens.append(rev_vocab.get(idx, ''))
            
    return "".join(output_tokens)

def main():
    parser = argparse.ArgumentParser(description="Chat with your HCTS Model.")
    parser.add_argument('--build', default='my_first_forge', help="Name of the folder in 'builds/'")
    args = parser.parse_args()

    print(f"--- Loading AI Brain from build: '{args.build}' ---")
    model, vocab, device = load_model_and_vocab(args.build)
    
    print("\n" + "="*50)
    print(" HCTS v1 INFERENCE ENGINE ONLINE")
    print(" Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("YOU > ")
            if user_input.lower() in ['exit', 'quit']: break
            
            response = generate_response(model, vocab, user_input, device)
            print(f"AI  > {response}\n")
            
        except KeyboardInterrupt:
            break
    print("\nSession Terminated.")

if __name__ == '__main__':
    main()