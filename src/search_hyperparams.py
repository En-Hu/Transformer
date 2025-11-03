import os
import yaml
import csv
import itertools
import subprocess
import pandas as pd


def load_yaml(path):
    """安全加载 YAML 文件"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ✅ 统一确定项目根路径
    project_root = "/data/huen/Code/Transformer"
    config_dir = os.path.join(project_root, "configs")
    train_script = os.path.join(project_root, "src", "train.py")

    # ✅ 指定超参搜索配置文件路径
    search_cfg_path = os.path.join(config_dir, "search_hyperparams.yaml")
    grid_cfg = load_yaml(search_cfg_path)

    # ✅ 检查 base_config 是否存在
    if "base_config" not in grid_cfg:
        raise KeyError("[ERROR] search_hyperparams.yaml 中缺少 'base_config' 字段。")

    # ✅ 加载 base 配置文件
    base_cfg_path = os.path.join(project_root, grid_cfg["base_config"])
    base_cfg = load_yaml(base_cfg_path)

    # ✅ 结果目录
    root_dir = os.path.join(project_root, "results", "search_hyperparams")
    os.makedirs(root_dir, exist_ok=True)

    # ✅ 读取搜索空间
    search_space = grid_cfg["search_space"]

    # ✅ 生成所有超参组合
    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = list(itertools.product(*values))
    print(f"[Search] 共 {len(combos)} 组超参数组合待运行。")
    print(f"[INFO] 基础配置文件: {base_cfg_path}")

    summary_records = []

    # ============================================================
    # 🚀 三层配置合并逻辑:
    # base_cfg < grid_cfg(顶层字段) < overrides(每组搜索参数)
    # ============================================================
    for combo in combos:
        overrides = dict(zip(keys, combo))
        run_name = "_".join(f"{k}{v}" for k, v in overrides.items())
        save_dir = os.path.join(root_dir, f"run_{run_name}")
        os.makedirs(save_dir, exist_ok=True)

        # 1️⃣ 从 base.yaml 复制基础配置
        config = base_cfg.copy()

        # 2️⃣ 应用 search_hyperparams.yaml 顶层字段（跳过非超参字段）
        for k, v in grid_cfg.items():
            if k not in ["base_config", "search_space"]:
                config[k] = v

        # 3️⃣ 应用当前组合参数覆盖
        config.update(overrides)

        # 4️⃣ 保存路径
        config["save_dir"] = save_dir

        # ✅ 写入当前实验配置文件
        tmp_yaml = os.path.join(save_dir, "config.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        print(f"\n🚀 运行实验: {run_name}")
        print(f"[INFO] 使用配置: {tmp_yaml}")

        # ✅ 调用 train.py（固定 cwd）
        subprocess.run(
            ["python", train_script, "--config", tmp_yaml],
            cwd=project_root,
            check=True
        )

        # ✅ 收集实验结果
        log_csv = os.path.join(save_dir, "train_log.csv")
        if os.path.exists(log_csv):
            df = pd.read_csv(log_csv)
            best_row = df.iloc[df["val_rougeL"].idxmax()]
            summary_records.append({
                **overrides,
                "best_epoch": int(best_row["epoch"]),
                "best_rougeL": best_row["val_rougeL"],
                "save_dir": save_dir
            })
        else:
            print(f"[WARN] 未找到日志文件: {log_csv}")

    # ✅ 保存汇总结果
    summary_path = os.path.join(root_dir, "summary_search.csv")
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    print(f"\n✅ 所有搜索完成！汇总已保存到 {summary_path}")


if __name__ == "__main__":
    main()
