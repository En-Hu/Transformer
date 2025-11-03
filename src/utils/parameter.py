import argparse
import yaml

def _auto_cast(args):
    """自动将字符串型数字转为数值型"""
    for key, val in vars(args).items():
        if isinstance(val, str):
            # 尝试自动识别科学计数法或普通数字
            try:
                if "." in val or "e" in val or "E" in val:
                    setattr(args, key, float(val))
                elif val.isdigit():
                    setattr(args, key, int(val))
            except Exception:
                pass
    return args

def parse_args():
    ap = argparse.ArgumentParser()
    # 1️⃣ 额外增加配置文件选项
    ap.add_argument("--config", type=str, default=None, help="YAML 配置文件路径（可选）")

    # 数据相关
    ap.add_argument("--data_dir", type=str, default="datasets/CNN_DailyMail")
    ap.add_argument("--save_dir", type=str, default="results")
    ap.add_argument("--sample_ratio", type=float, default=0.1, help="数据集抽样比例（加速调试）")

    # 模型相关
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    # 训练相关
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # 文本处理相关
    ap.add_argument("--max_input_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--val_eval_samples", type=int, default=128, help="验证时评估的样本数")

    # 生成相关
    ap.add_argument("--min_gen_len", type=int, default=20)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)

    # 消融实验
    ap.add_argument("--use_pos_encoding", type=bool, default=True)
    ap.add_argument("--use_layer_norm", type=bool, default=True)
    ap.add_argument("--use_feedforward", type=bool, default=True)

    args = ap.parse_args()

    # 2️⃣ 若指定 YAML 文件，则读取并覆盖 argparse 默认参数
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
        print(f"[Config] 从 {args.config} 加载参数：")
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                print(f"  - {k}: {getattr(args, k)} → {v}")
                setattr(args, k, v)
            else:
                print(f"[WARN] YAML 中包含未知参数: {k}")
    args = _auto_cast(args)
    
    return args
