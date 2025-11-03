import os
import csv
import time
import glob
import numpy as np
import torch

from models import TransformerSeq2Seq, TransformerSeq2SeqAblate
from utils.load_cnndailymail import CNNDailyMailTokenizer, load_data
from utils.parameter import parse_args
from utils.rouge import ids_to_text, _tokenize, rouge_n, rouge_l
from utils.seed import set_seed


def find_all_checkpoints(search_dirs):
    ckpts = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # 匹配形如 */best_summ.pt 的文件
        for path in glob.glob(os.path.join(d, "**", "best_summ.pt"), recursive=True):
            ckpts.append(os.path.abspath(path))
    return sorted(ckpts)


@torch.no_grad()
def evaluate_on_test(model, data_loader, inv_vocab, tokenizer, args, device, max_samples=None):
    model.eval()
    r1_list, r2_list, rl_list = [], [], []
    n_samples = 0
    t0 = time.time()

    for batch in data_loader:
        src_ids = batch["input_ids"].to(device)
        src_mask = batch["attention_mask"].to(device)
        gold_ids = batch["labels"].to(device)

        pred_ids = model.greedy_generate(src_ids, src_mask, tokenizer.cls_id, tokenizer.eos_id, args)

        for pred, gold in zip(pred_ids, gold_ids):
            hyp_text = ids_to_text(pred, inv_vocab)
            ref_text = ids_to_text(gold, inv_vocab)
            hyp_tok, ref_tok = _tokenize(hyp_text), _tokenize(ref_text)
            r1_list.append(rouge_n(ref_tok, hyp_tok, n=1))
            r2_list.append(rouge_n(ref_tok, hyp_tok, n=2))
            rl_list.append(rouge_l(ref_tok, hyp_tok))
            n_samples += 1
            if max_samples is not None and n_samples >= max_samples:
                break
        if max_samples is not None and n_samples >= max_samples:
            break

    elapsed = time.time() - t0
    avg_r1 = float(np.mean(r1_list)) if r1_list else 0.0
    avg_r2 = float(np.mean(r2_list)) if r2_list else 0.0
    avg_rl = float(np.mean(rl_list)) if rl_list else 0.0
    return {
        "R1": avg_r1,
        "R2": avg_r2,
        "RL": avg_rl,
        "num_samples": n_samples,
        "time_sec": elapsed,
        "samples_per_sec": (n_samples / elapsed) if elapsed > 0 else 0.0,
    }


def load_tokenizer_and_data(args):
    tokenizer = CNNDailyMailTokenizer(
        vocab_path=os.path.join(args.data_dir, "vocab.txt"),
        data_dir=args.data_dir
    )
    data_loaders = load_data(args, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 1
    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    return tokenizer, data_loaders, vocab_size, inv_vocab


# ===== 新增：模型参数统计 =====
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model_from_ckpt(ckpt_path, vocab_size, tokenizer_pad_id, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "args" not in ckpt or "model_state" not in ckpt:
        raise RuntimeError(f"[ERROR] 检查点缺少必要字段: {ckpt_path}")

    saved_args = ckpt["args"]
    tokenizer_info = ckpt.get("tokenizer_info", {})
    pad_id = tokenizer_info.get("pad_id", tokenizer_pad_id)

    # 根据路径判断模型类型
    ckpt_path_lower = ckpt_path.lower()
    if "ablate" in ckpt_path_lower:
        ModelClass = TransformerSeq2SeqAblate
        model_name = "TransformerSeq2SeqAblate"
    else:
        ModelClass = TransformerSeq2Seq
        model_name = "TransformerSeq2Seq"

    model = ModelClass(saved_args, vocab_size, pad_id).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, saved_args, model_name


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def main():
    """
    用法：
    python test_all.py \
        --data_dir data/CNNDM \
        --device cuda:0 \
        --batch_size 16 \
        --max_epochs 1  # 不用于测试，但 parse_args 需要
    """
    args = parse_args()
    set_seed(args.seed)

    # 选择设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Test] 使用设备: {device}")

    # 载入分词器与数据
    tokenizer, data_loaders, vocab_size, inv_vocab = load_tokenizer_and_data(args)
    test_loader = data_loaders["test"]

    # 查找所有 checkpoint
    search_dirs = [
        "results/base",
        "results/ablate",
        "results/ablate2",
        "results/search_hyperparams",
    ]
    ckpts = find_all_checkpoints(search_dirs)
    if not ckpts:
        print("[Test] 未找到任何 best_summ.pt，请确认结果目录。")
        return
    print(f"[Test] 共发现 {len(ckpts)} 个模型：")
    for p in ckpts:
        print("    -", p)

    # 汇总 CSV
    summary_csv = os.path.join("results", "test_summary.csv")
    ensure_dir(summary_csv)
    write_header = not os.path.exists(summary_csv)

    # 最大评测样本数：优先使用 args.test_eval_samples，其次回退 args.val_eval_samples，否则 None 全量
    max_samples = getattr(args, "test_eval_samples", None)
    if max_samples is None:
        max_samples = getattr(args, "val_eval_samples", None)

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "ckpt_path", "model_type",
                "params_total", "params_trainable", "params_total_M", "params_trainable_M",  # 新增列
                "R1", "R2", "RL",
                "num_samples", "time_sec", "samples_per_sec"
            ])

        # 逐个评测
        for ckpt_path in ckpts:
            try:
                model, saved_args, model_name = build_model_from_ckpt(
                    ckpt_path, vocab_size, tokenizer.pad_id, device
                )

                # ===== 参数统计并打印 =====
                params_total, params_trainable = count_parameters(model)
                params_total_M = params_total / 1e6
                params_trainable_M = params_trainable / 1e6
                print(f"[Model] 参数统计 —— total: {params_total:,}  trainable: {params_trainable:,} "
                      f"({params_total_M:.3f}M / {params_trainable_M:.3f}M)")

                print(f"\n[Test] 评测模型: {ckpt_path}  [{model_name}]")
                metrics = evaluate_on_test(
                    model=model,
                    data_loader=test_loader,
                    inv_vocab=inv_vocab,
                    tokenizer=tokenizer,
                    args=saved_args,           # 使用保存时的生成超参
                    device=device,
                    max_samples=max_samples
                )
                print(f"[Result] R1={metrics['R1']:.4f}  R2={metrics['R2']:.4f}  RL={metrics['RL']:.4f}  "
                      f"samples={metrics['num_samples']}  time={metrics['time_sec']:.2f}s")

                writer.writerow([
                    ckpt_path, model_name,
                    params_total, params_trainable, f"{params_total_M:.3f}", f"{params_trainable_M:.3f}",
                    f"{metrics['R1']:.6f}", f"{metrics['R2']:.6f}", f"{metrics['RL']:.6f}",
                    metrics["num_samples"], f"{metrics['time_sec']:.4f}", f"{metrics['samples_per_sec']:.2f}"
                ])

            except Exception as e:
                print(f"[WARN] 评测失败，跳过: {ckpt_path}\n       原因: {repr(e)}")
                # 失败项也记录，便于排查
                writer.writerow([
                    ckpt_path, "ERROR",
                    "", "", "", "",  # 参数统计占位
                    "", "", "",
                    0, "", ""
                ])

    print(f"\n✅ 全部评测完成！结果已写入 {summary_csv}")


if __name__ == "__main__":
    main()
