import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import TransformerSeq2Seq
from models import TransformerSeq2SeqAblate
from utils.load_cnndailymail import CNNDailyMailTokenizer, load_data
from utils.parameter import parse_args
from utils.plot import plot_and_save
from utils.rouge import ids_to_text, _tokenize, rouge_n, rouge_l
from utils.seed import set_seed


def main():
    args = parse_args()
    set_seed(args.seed)

    # ===============================
    # 自动识别运行模式并创建结果路径
    # ===============================
    base_root = "results/base"

    if hasattr(args, "save_dir") and args.save_dir is not None:
        # 从外部传入，例如 ablation / search 模式
        save_dir = args.save_dir
        if "ablate" in save_dir:
            base_root = "results/ablate"
        elif "search_hyperparams" in save_dir:
            base_root = "results/search_hyperparams"
    else:
        # 本地单次运行模式
        run_name = f"run_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}"
        save_dir = os.path.join(base_root, run_name)
        args.save_dir = save_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"[Main] 结果目录: {save_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] 使用设备: {device}")

    # ===============================
    # 初始化 Tokenizer & 数据
    # ===============================
    tokenizer = CNNDailyMailTokenizer(
        vocab_path=os.path.join(args.data_dir, "vocab.txt"),
        data_dir=args.data_dir
    )
    data_loaders = load_data(args, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 1
    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # ===============================
    # 初始化模型与优化器
    # ===============================
    if "ablate" in args.save_dir:
        print("[Main] 使用消融实验模型 TransformerSeq2SeqAblate ✅")
        model = TransformerSeq2SeqAblate(args, vocab_size, tokenizer.pad_id).to(device)
    else:
        print("[Main] 使用标准模型 TransformerSeq2Seq ✅")
        model = TransformerSeq2Seq(args, vocab_size, tokenizer.pad_id).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数总数: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # ===============================
    # 初始化日志文件
    # ===============================
    csv_path = os.path.join(save_dir, "train_log.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_rouge1", "val_rouge2", "val_rougeL", "lr", "time of epoch"])

    best_rougeL = 0.0
    train_losses, val_rougeLs = [], []

    # ===============================
    # 主训练循环
    # ===============================
    start_train_time = time.time()
    if "ablate" in args.save_dir:
        exp_type = "Ablation"
    elif "search_hyperparams" in args.save_dir:
        exp_type = "Hyperparam Search"
    else:
        exp_type = "Base"

    for epoch in range(1, args.max_epochs + 1):
        start_epoch_time = time.time()
        print(f"\n===== Epoch {epoch}/{args.max_epochs} ({exp_type} Experiment) =====")
        model.train()
        total_loss = 0.0

        # -------------------- 训练 --------------------
        for batch in data_loaders["train"]:
            src_ids = batch["input_ids"].to(device)
            src_mask = batch["attention_mask"].to(device)
            tgt_ids = batch["labels"].to(device)
            tgt_mask = batch["decoder_mask"].to(device)

            B = tgt_ids.size(0)
            bos_col = torch.full((B, 1), tokenizer.cls_id, device=device)
            dec_input = torch.cat([bos_col, tgt_ids[:, :-1]], dim=1)
            dec_mask = torch.cat([torch.ones((B, 1), device=device), tgt_mask[:, :-1]], dim=1)

            logits = model(src_ids, src_mask, dec_input, dec_mask)
            loss = loss_fn(logits.reshape(-1, vocab_size), tgt_ids.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(data_loaders["train"])
        train_losses.append(avg_train_loss)
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        print(f"[Train] 平均损失: {avg_train_loss:.4f}，用时: {epoch_time:.4f}")

        # -------------------- 验证 --------------------
        model.eval()
        r1_list, r2_list, rl_list = [], [], []
        with torch.no_grad():
            for batch in data_loaders["val"]:
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
                if len(rl_list) >= args.val_eval_samples:
                    break
        
        avg_r1, avg_r2, avg_rl = np.mean(r1_list), np.mean(r2_list), np.mean(rl_list)
        val_rougeLs.append(avg_rl)
        print(f"[Val] R1={avg_r1:.4f}, R2={avg_r2:.4f}, RL={avg_rl:.4f}")

        # 写入日志
        with open(csv_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_r1, avg_r2, avg_rl, scheduler.get_last_lr()[0], epoch_time])

        # 保存最佳模型
        if avg_rl > best_rougeL:
            best_rougeL = avg_rl
            torch.save({
                "model_state": model.state_dict(),
                "args": args,
                "tokenizer_info": {"pad_id": tokenizer.pad_id, "bos_id": tokenizer.cls_id, "eos_id": tokenizer.eos_id}
            }, os.path.join(save_dir, "best_summ.pt"))
            print(f"[Save] 最优模型更新（ROUGE-L: {best_rougeL:.4f}）")

        scheduler.step()
    
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    # -------------------- 绘制曲线 --------------------
    plot_and_save(train_losses, os.path.join(save_dir, "train_loss.png"), "Train Loss", "Loss")
    plot_and_save(val_rougeLs, os.path.join(save_dir, "val_rougeL.png"), "Val ROUGE-L", "ROUGE-L Score")

    # -------------------- 写入最终 summary --------------------
    summary_path = os.path.join(base_root, "summary.csv")
    header = ["lr", "batch_size", "dropout", "best_rougeL", "save_dir", "train_time"]
    row = [args.lr, args.batch_size, args.dropout, best_rougeL, save_dir, train_time]
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"\n✅ 训练完成！最优ROUGE-L: {best_rougeL:.4f}（保存路径: {save_dir}/best_summ.pt）")
    print(f"[Summary] 已记录到 {summary_path}, 总用时 {train_time:.4f}")


if __name__ == "__main__":
    main()
