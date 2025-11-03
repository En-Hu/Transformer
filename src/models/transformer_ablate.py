import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# -------------------- 位置编码 --------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(1), :])


# -------------------- 注意力模块 --------------------
def scaled_dot_attn(q, k, v, attn_mask: Optional[torch.Tensor] = None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / np.sqrt(d_k)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须是n_heads的整数倍"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in ["q", "k", "v", "o"]])
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        B, Tq, _ = q.shape
        q, k, v = [p(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2) for p, x in zip(self.proj[:3], [q, k, v])]
        out = scaled_dot_attn(q, k, v, attn_mask.unsqueeze(1) if attn_mask is not None else None)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.drop(self.proj[3](out))


# -------------------- 前馈层 --------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------- 编码器层（可控归一化/FFN） --------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_layer_norm: bool = True, use_feedforward: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_feedforward = use_feedforward

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout) if use_feedforward else nn.Identity()

    def forward(self, x, src_mask=None):
        if self.use_layer_norm:
            x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)
            x = x + self.ffn(self.ln2(x))
        else:
            x = x + self.self_attn(x, x, x, src_mask)
            x = x + self.ffn(x)
        return x


# -------------------- 解码器层（可控归一化/FFN） --------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_layer_norm: bool = True, use_feedforward: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_feedforward = use_feedforward

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout) if use_feedforward else nn.Identity()

    def forward(self, x, mem, tgt_mask=None, src_mask=None):
        if self.use_layer_norm:
            x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), tgt_mask)
            x = x + self.cross_attn(self.ln2(x), self.ln2(mem), self.ln2(mem), src_mask)
            x = x + self.ffn(self.ln3(x))
        else:
            x = x + self.self_attn(x, x, x, tgt_mask)
            x = x + self.cross_attn(x, mem, mem, src_mask)
            x = x + self.ffn(x)
        return x


# -------------------- 因果掩码 --------------------
def subsequent_mask(sz: int, device: torch.device):
    return torch.tril(torch.ones(sz, sz, device=device, dtype=torch.long)).unsqueeze(0)


# -------------------- 主模型 --------------------
class TransformerSeq2SeqAblate(nn.Module):
    def __init__(self, args, vocab_size: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.max_src_len = args.max_input_len
        self.max_tgt_len = args.max_target_len

        self.use_pos_encoding = getattr(args, "use_pos_encoding", True)
        self.use_layer_norm = getattr(args, "use_layer_norm", True)
        self.use_feedforward = getattr(args, "use_feedforward", True)

        # 词嵌入 + 可选位置编码
        self.tok_embed = nn.Embedding(vocab_size, args.d_model, padding_idx=pad_id)
        if self.use_pos_encoding:
            self.pos_enc = SinusoidalPositionalEncoding(
                d_model=args.d_model,
                max_len=max(args.max_input_len, args.max_target_len),
                dropout=args.dropout
            )
        else:
            self.pos_enc = nn.Identity()

        # 编码器 & 解码器
        self.encoder = nn.ModuleList([
            EncoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                         self.use_layer_norm, self.use_feedforward)
            for _ in range(args.n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                         self.use_layer_norm, self.use_feedforward)
            for _ in range(args.n_layers)
        ])

        # 输出层（权重共享）
        self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def encode(self, src_ids, src_mask):
        x = self.pos_enc(self.tok_embed(src_ids))
        src_mask = src_mask.unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_ids, mem, tgt_mask, src_mask):
        x = self.pos_enc(self.tok_embed(tgt_ids))
        B, Tt = tgt_ids.shape
        causal_mask = subsequent_mask(Tt, x.device)
        tgt_mask = tgt_mask.unsqueeze(1).long() & causal_mask
        src_mask = src_mask.unsqueeze(1).expand(-1, Tt, -1)
        for layer in self.decoder:
            x = layer(x, mem, tgt_mask, src_mask)
        return x

    def forward(self, src_ids, src_mask, decoder_input, decoder_mask):
        mem = self.encode(src_ids, src_mask)
        dec_out = self.decode(decoder_input, mem, decoder_mask, src_mask)
        return self.lm_head(dec_out)

    @torch.no_grad()
    def greedy_generate(self, src_ids, src_mask, bos_id, eos_id, args):
        B, device = src_ids.size(0), src_ids.device
        mem = self.encode(src_ids, src_mask)
        gen_seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        decoder_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(args.max_target_len - 1):
            dec_out = self.decode(gen_seq, mem, decoder_mask, src_mask)
            logits = self.lm_head(dec_out[:, -1, :])
            if gen_seq.size(1) < args.min_gen_len:
                logits[:, eos_id] = -1e9
            next_token = logits.argmax(dim=-1, keepdim=True)
            gen_seq = torch.cat([gen_seq, next_token], dim=1)
            decoder_mask = torch.cat([decoder_mask, torch.ones((B, 1), device=device)], dim=1)
            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return gen_seq
