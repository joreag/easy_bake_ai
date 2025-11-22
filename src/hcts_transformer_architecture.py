import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # ... (This class is correct and unchanged) ...
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

class HCTS_Transformer(nn.Module):
    # ... (This class is excellent and production-ready, no changes needed to its logic) ...
    def __init__(self, vocab_size: int, d_model: int = 384, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1, pad_idx: int = 0, **kwargs):
        super().__init__()
        self.d_model = d_model; self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout); self.pad_idx = pad_idx
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = (src == self.pad_idx)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = (src == self.pad_idx)
        memory = self.encode(src)
        output = self.decode(tgt, memory, src_key_padding_mask)
        return self.fc_out(output)
    @torch.no_grad()
    def generate(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 100) -> torch.Tensor:
        self.eval(); device = src.device
        memory = self.encode(src); src_key_padding_mask = (src == self.pad_idx)
        tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            output = self.decode(tgt, memory, src_key_padding_mask)
            prob = self.fc_out(output[:, -1])
            _, next_word_idx = torch.max(prob, dim=1); next_word_idx = next_word_idx.item()
            tgt = torch.cat([tgt, torch.tensor([[next_word_idx]], device=device)], dim=1)
            if next_word_idx == eos_idx: break
        self.train()
        return tgt