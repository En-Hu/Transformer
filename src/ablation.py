import os
import yaml
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
    # ✅ 项目根目录与路径设置
    project_root = "/data/huen/Code/Transformer"
    config_dir = os.path.join(project_root, "configs")
    train_script = os.path.join(project_root, "src", "train.py")

    # ✅ 加载消融主配置文件
    ablate_cfg_path = os.path.join(config_dir, "ablation.yaml")
    ablate_cfg = load_yaml(ablate_cfg_path)

    # ✅ 检查 base_config 是否存在
    if "base_config" not in ablate_cfg:
        raise KeyError("[ERROR] ablation.yaml 中缺少 'base_config' 字段。")

    base_cfg_path = os.path.join(project_root, ablate_cfg["base_config"])
    base_cfg = load_yaml(base_cfg_path)

    # ✅ 结果目录
    root_dir = os.path.join(project_root, "results", "ablate")
    os.makedirs(root_dir, exist_ok=True)

    experiments = ablate_cfg["experiments"]
    print(f"[Ablation] 共 {len(experiments)} 组消融实验待运行。")
    print(f"[INFO] 基础配置文件: {base_cfg_path}")

    summary_records = []

    # =============================================================
    # 🚀 三层配置合并逻辑:
    # base_cfg < ablate_cfg(顶层字段) < overrides(实验覆盖)
    # =============================================================
    for exp in experiments:
        name = exp["name"]
        overrides = exp.get("overrides", {})

        run_name = f"run_{name}"
        save_dir = os.path.join(root_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)

        # 1️⃣ 从 base.yaml 加载基础配置
        config = base_cfg.copy()

        # 2️⃣ 应用 ablation.yaml 顶层字段（跳过非超参字段）
        for k, v in ablate_cfg.items():
            if k not in ["experiments", "base_config"]:
                config[k] = v

        # 3️⃣ 应用当前实验覆盖
        config.update(overrides)

        # 4️⃣ 保存目录
        config["save_dir"] = save_dir

        # ✅ 写入配置文件
        tmp_yaml = os.path.join(save_dir, "config.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        print(f"\n🚀 运行消融实验: {name}")
        print(f"[INFO] 使用配置: {tmp_yaml}")

        # ✅ 执行训练脚本
        subprocess.run(
            ["python", train_script, "--config", tmp_yaml],
            cwd=project_root,
            check=True
        )

        # ✅ 收集训练结果
        log_csv = os.path.join(save_dir, "train_log.csv")
        if os.path.exists(log_csv):
            df = pd.read_csv(log_csv)
            best_row = df.iloc[df["val_rougeL"].idxmax()]
            summary_records.append({
                "experiment": name,
                **overrides,
                "best_epoch": int(best_row["epoch"]),
                "best_rougeL": best_row["val_rougeL"],
                "save_dir": save_dir
            })
        else:
            print(f"[WARN] 未找到日志文件: {log_csv}")

    # ✅ 保存汇总表
    summary_path = os.path.join(root_dir, "summary_ablation.csv")
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    print(f"\n✅ 所有消融实验完成！汇总已保存到 {summary_path}")


if __name__ == "__main__":
    main()
