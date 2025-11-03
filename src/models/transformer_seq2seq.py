import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# -------------------- 5. Transformer模型（优化设计，删除冗余归一化） --------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(1), :])


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
        # 简化线性层定义
        self.proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in ["q", "k", "v", "o"]])
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        B, Tq, _ = q.shape
        # 多头拆分：(B, T, D) → (B, H, T, Dk)
        q, k, v = [p(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2) for p, x in zip(self.proj[:3], [q, k, v])]
        # 注意力计算
        out = scaled_dot_attn(q, k, v, attn_mask.unsqueeze(1) if attn_mask is not None else None)
        # 多头合并：(B, H, T, Dk) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.drop(self.proj[3](out))


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


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # Pre-Norm结构：归一化 → 注意力 → 残差连接
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, mem, tgt_mask=None, src_mask=None):
        # Pre-Norm结构：自注意力 → 交叉注意力 → FFN
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), tgt_mask)
        x = x + self.cross_attn(self.ln2(x), self.ln2(mem), self.ln2(mem), src_mask)
        x = x + self.ffn(self.ln3(x))
        return x


def subsequent_mask(sz: int, device: torch.device):
    """生成整数型的因果掩码（0/1）"""
    return torch.tril(torch.ones(sz, sz, device=device, dtype=torch.long)).unsqueeze(0)  # 新增 dtype=torch.long

class TransformerSeq2Seq(nn.Module):
    def __init__(self, args, vocab_size: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.max_src_len = args.max_input_len
        self.max_tgt_len = args.max_target_len

        # 嵌入层（词嵌入+位置编码）
        self.tok_embed = nn.Embedding(vocab_size, args.d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=args.d_model,
            max_len=max(args.max_input_len, args.max_target_len),
            dropout=args.dropout
        )

        # 编码器/解码器栈
        self.encoder = nn.ModuleList([
            EncoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout)
            for _ in range(args.n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout)
            for _ in range(args.n_layers)
        ])

        # 输出层（权重共享）
        self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def encode(self, src_ids, src_mask):
        """编码器前向传播"""
        x = self.pos_enc(self.tok_embed(src_ids))
        # 扩展掩码维度：(B, T) → (B, 1, T)（适配多头注意力）
        src_mask = src_mask.unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_ids, mem, tgt_mask, src_mask):
        x = self.pos_enc(self.tok_embed(tgt_ids))
        B, Tt = tgt_ids.shape
        # 因果掩码（已为整数型）
        causal_mask = subsequent_mask(Tt, x.device)
        # 将 tgt_mask 转为整数型，再与因果掩码按位与
        tgt_mask = tgt_mask.unsqueeze(1).long() & causal_mask  # 新增 .long()
        src_mask = src_mask.unsqueeze(1).expand(-1, Tt, -1)

        for layer in self.decoder:
            x = layer(x, mem, tgt_mask, src_mask)
        return x

    def forward(self, src_ids, src_mask, decoder_input, decoder_mask):
        """训练时前向传播（需输入解码器输入）"""
        mem = self.encode(src_ids, src_mask)
        dec_out = self.decode(decoder_input, mem, decoder_mask, src_mask)
        return self.lm_head(dec_out)

    @torch.no_grad()
    def greedy_generate(self, src_ids, src_mask, bos_id, eos_id, args):
        """贪婪解码（简化变量命名，优化逻辑）"""
        B, device = src_ids.size(0), src_ids.device
        mem = self.encode(src_ids, src_mask)

        # 初始化生成序列（B, 1）：以BOS开头
        gen_seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        decoder_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(args.max_target_len - 1):
            # 解码当前步
            dec_out = self.decode(gen_seq, mem, decoder_mask, src_mask)
            logits = self.lm_head(dec_out[:, -1, :])  # (B, vocab_size)

            # 禁止过早生成EOS（小于min_gen_len时）
            if gen_seq.size(1) < args.min_gen_len:
                logits[:, eos_id] = -1e9

            # 禁止重复n-gram
            if args.no_repeat_ngram > 0:
                for b in range(B):
                    seq = gen_seq[b]
                    if len(seq) < args.no_repeat_ngram:
                        continue
                    # 获取当前n-gram前缀对应的禁止token
                    prefix = tuple(seq[-(args.no_repeat_ngram-1):].tolist())
                    banned_tokens = set()
                    for i in range(len(seq) - args.no_repeat_ngram + 1):
                        if tuple(seq[i:i+args.no_repeat_ngram-1].tolist()) == prefix:
                            banned_tokens.add(int(seq[i+args.no_repeat_ngram-1]))
                    if banned_tokens:
                        logits[b][list(banned_tokens)] = -1e9

            # 选择概率最大的token
            next_token = logits.argmax(dim=-1, keepdim=True)
            gen_seq = torch.cat([gen_seq, next_token], dim=1)
            decoder_mask = torch.cat([decoder_mask, torch.ones((B, 1), device=device)], dim=1)

            # 标记完成的样本
            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return gen_seq
