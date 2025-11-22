import sys
import os
import json
import torch
from sentence_transformers import util

TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLBOX_DIR)
sys.path.append(PROJECT_ROOT)

from src.hcts_transformer_architecture import HCTS_Transformer
from src.query_engine import KnowledgeGraphQueryEngine

class LatentSpaceMapper:
    def __init__(self, build_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        build_dir = os.path.join(PROJECT_ROOT, 'builds', build_name)
        
        vocab_path = os.path.join(build_dir, 'vocab.json')
        model_path = os.path.join(build_dir, 'model.pth')
        graph_path = os.path.join(build_dir, 'knowledge_graph.pkl')

        if not os.path.exists(model_path):
            sys.exit(f"Build '{build_name}' not found.")

        # Load Vocab
        with open(vocab_path, 'r') as f: self.vocab = json.load(f)
        
        # Load Graph
        print("Loading Knowledge Graph...")
        self.query_engine = KnowledgeGraphQueryEngine(graph_path)

        # Load Model
        print("Loading Neural Network...")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        self.model = HCTS_Transformer(
            vocab_size=len(self.vocab),
            d_model=int(config.get('d_model', 512)),
            nhead=int(config.get('nhead', 8)),
            num_encoder_layers=int(config.get('num_encoder_layers', 6)),
            num_decoder_layers=int(config.get('num_decoder_layers', 6)),
            dim_feedforward=int(config.get('dim_feedforward', 2048)),
            pad_idx=self.vocab.get('[PAD]', 0)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.embeddings = {}
        self.concept_names = []

    def _tokenize(self, text):
        token_ids = [self.vocab.get(char, self.vocab.get('[UNK]', 0)) for char in text.lower()]
        token_ids = [self.vocab.get('[CLS]', 1)] + token_ids + [self.vocab.get('[SEP]', 2)]
        return torch.tensor(token_ids).unsqueeze(0).to(self.device)

    def map_all_concepts(self):
        print("\n[SCANNING] Generating latent space vectors for all concepts...")
        all_concepts = self.query_engine.get_all_canonical_names()
        
        with torch.no_grad():
            for name in all_concepts:
                # We probe the encoder with the concept question
                question = f"what is {name.lower()}?"
                src = self._tokenize(question)
                try:
                    # Extract the semantic vector from the Encoder
                    embedding = self.model.encode(src).mean(dim=1).squeeze(0)
                    self.embeddings[name] = embedding
                    self.concept_names.append(name)
                except Exception:
                    pass
        print(f"  -> Mapped {len(self.embeddings)} concepts.")

    def analyze_all(self, top_k=5):
        print("\n" + "#"*60)
        print("###  LATENT SPACE ASSOCIATIONS (The AI's Mental Map)  ###")
        print("#"*60)
        
        # Limit output to 20 random concepts to keep it readable, or sort alphabetically
        for name in sorted(self.concept_names)[:20]: 
            target_emb = self.embeddings[name]
            similarities = []
            for other_name, other_emb in self.embeddings.items():
                if other_name != name:
                    sim = util.cos_sim(target_emb, other_emb).item()
                    similarities.append((other_name, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nConcept: '{name}' is associated with:")
            for i, (res_name, sim) in enumerate(similarities[:top_k]):
                print(f"   {i+1}. {res_name} ({sim:.3f})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='my_first_forge')
    args = parser.parse_args()
    
    scanner = LatentSpaceMapper(args.build)
    scanner.map_all_concepts()
    scanner.analyze_all()