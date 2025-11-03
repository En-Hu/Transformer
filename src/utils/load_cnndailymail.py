import os
import re
import numpy as np
import pandas as pd
from glob import glob
from datasets import load_dataset
from torch.utils.data import DataLoader

from collections import Counter

# -------------------- 3. Tokenizer --------------------
class CNNDailyMailTokenizer:
    def __init__(self, vocab_path, data_dir, vocab_size=30000):
        self.vocab_path = vocab_path
        self.data_dir = data_dir
        self.vocab_size = vocab_size

        # 不存在词表则自动构建
        if not os.path.exists(vocab_path):
            print(f"[Tokenizer] 生成词表: {vocab_path}")
            self._build_vocab()

        # 加载词表并定义特殊符号ID
        self.vocab = self._load_vocab()
        self.pad_id = self.vocab["<pad>"]
        self.cls_id = self.vocab["<cls>"]
        self.unk_id = self.vocab["<unk>"]
        self.eos_id = self.vocab["<eos>"]
        print(f"[Tokenizer] 特殊ID: pad={self.pad_id}, bos={self.cls_id}, eos={self.eos_id}")

    def _load_vocab(self):
        """加载词表（简化读取逻辑）"""
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            return {line.strip().split()[0]: int(line.strip().split()[1]) for line in f}

    def _tokenize_str(self, text):
        """基础文本分词"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.strip().split()

    def _build_vocab(self):
        """用glob可靠获取训练文件，简化词表构建"""
        train_files = glob(os.path.join(self.data_dir, "train-*.parquet"))
        if not train_files:
            raise FileNotFoundError(f"[Tokenizer] 未找到训练文件: {self.data_dir}/train-*.parquet")

        # 统计词频
        counter = Counter()
        for path in train_files:
            print(f"[Tokenizer] 处理文件: {path}")
            df = pd.read_parquet(path)
            for text in df["article"].astype(str).tolist() + df["highlights"].astype(str).tolist():
                counter.update(self._tokenize_str(text))

        # 保存词表
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            # 先写特殊符号
            f.write("<pad> 0\n<cls> 1\n<unk> 2\n<eos> 3\n")
            # 再写高频词
            for idx, (tok, _) in enumerate(counter.most_common(self.vocab_size - 4), start=4):
                f.write(f"{tok} {idx}\n")

    def tokenize(self, text, max_len=128, add_bos=False, add_eos=False):
        """统一tokenize逻辑（截断+填充）"""
        tokens = self._tokenize_str(text)
        token_ids = []

        # 添加BOS/EOS
        if add_bos:
            token_ids.append(self.cls_id)
        token_ids += [self.vocab.get(tok, self.unk_id) for tok in tokens]
        if add_eos:
            token_ids.append(self.eos_id)

        # 截断或填充到max_len
        if len(token_ids) < max_len:
            token_ids += [self.pad_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        return token_ids


# -------------------- 4. 数据加载（保留datasets库实现，删除自定义Dataset） --------------------
def load_data(args, tokenizer):
    """统一数据加载逻辑（用datasets库，高效批量处理）"""
    # 加载Parquet数据集
    data_files = {
        "train": glob(os.path.join(args.data_dir, "train-*.parquet")),
        "validation": os.path.join(args.data_dir, "validation-00000-of-00001.parquet"),
        "test": os.path.join(args.data_dir, "test-00000-of-00001.parquet")
    }
    ds = load_dataset("parquet", data_files=data_files)

    # 数据集抽样
    def sample_ds(ds, ratio, split_name):
        if ratio < 1.0:
            sample_num = max(1, int(len(ds) * ratio))
            ds = ds.select(np.random.choice(len(ds), sample_num, replace=False))
            print(f"[Data] {split_name} 抽样后: {len(ds)} 样本")
        return ds

    ds["train"] = sample_ds(ds["train"], args.sample_ratio, "Train")
    ds["validation"] = sample_ds(ds["validation"], args.sample_ratio, "Validation")
    ds["test"] = sample_ds(ds["test"], args.sample_ratio, "Test")

    # 批量预处理（统一输入格式）
    def preprocess_batch(batch):
        input_ids, attention_mask, labels, decoder_mask = [], [], [], []
        for art, summ in zip(batch["article"], batch["highlights"]):
            # 源文本：不加BOS/EOS；目标文本：只加EOS
            src_ids = tokenizer.tokenize(art, args.max_input_len, add_bos=False, add_eos=False)
            tgt_ids = tokenizer.tokenize(summ, args.max_target_len, add_bos=False, add_eos=True)

            # 构建注意力掩码（1=有效，0=PAD）
            src_mask = [1 if id != tokenizer.pad_id else 0 for id in src_ids]
            tgt_mask = [1 if id != tokenizer.pad_id else 0 for id in tgt_ids]

            input_ids.append(src_ids)
            attention_mask.append(src_mask)
            labels.append(tgt_ids)
            decoder_mask.append(tgt_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_mask": decoder_mask
        }

    # 批量预处理并转换为Torch格式
    ds = ds.map(preprocess_batch, batched=True, batch_size=512)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "decoder_mask"])

    # 构建DataLoader
    return {
        "train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(ds["validation"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False)
    }