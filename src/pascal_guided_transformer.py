import torch
import torch.nn as nn
import math
from typing import List, Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TransformerStackBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
    
    def forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer(src, tgt, 
                                src_mask=None, tgt_mask=tgt_mask, 
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)

class TwistMatrix(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.transform(x))

class PascalGuidedTransformer(nn.Module):
    """
    The "Model Z": Hierarchical stacks connected by Twist Matrices.
    """
    def __init__(self, vocab_size: int, d_model: int, stack_config: List[Dict], nhead: int, dim_feedforward: int, dropout: float, pad_idx: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer_blocks = nn.ModuleList()
        for config in stack_config:
            self.transformer_blocks.append(
                TransformerStackBlock(
                    d_model=d_model, nhead=nhead,
                    num_encoder_layers=config['encoder_layers'],
                    num_decoder_layers=config['decoder_layers'],
                    dim_feedforward=dim_feedforward, dropout=dropout
                )
            )
        
        self.twist_matrices = nn.ModuleList([TwistMatrix(d_model) for _ in range(len(self.transformer_blocks) - 1)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        current_src = src_emb
        current_tgt = tgt_emb
        
        final_output = None
        
        for i, block in enumerate(self.transformer_blocks):
            output = block(current_src, current_tgt,
                           tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)
            
            if i < len(self.transformer_blocks) - 1:
                twist = self.twist_matrices[i]
                current_tgt = twist(output)
                
                # Twist memory from encoder for next block
                memory = block.transformer.encoder(current_src, src_key_padding_mask=src_key_padding_mask)
                current_src = twist(memory)
            else:
                final_output = output

        return self.fc_out(final_output)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 100) -> torch.Tensor:
        # Basic greedy generation wrapper for the hierarchical model
        self.eval()
        device = src.device
        
        # Pre-calculate encoder steps for efficiency is hard here due to twists, 
        # so we run the full forward pass iteratively (simpler for v1)
        tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.forward(src, tgt)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos_idx: break
            
        self.train()
        return tgt