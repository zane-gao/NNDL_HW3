#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_eval_results(eval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    per_class_rows: List[Dict] = []
    for path in sorted(eval_dir.glob("eval_*.json")):
        data = load_json(path)
        eval_data = data.get("evaluation", {})
        checkpoint_info = data.get("checkpoint_info", {})
        rows.append(
            {
                "file": path.name,
                "timestamp_raw": data.get("timestamp"),
                "model": data.get("model"),
                "split": data.get("split"),
                "dataset_size": data.get("dataset_size"),
                "loss": eval_data.get("loss"),
                "accuracy": eval_data.get("accuracy"),
                "eval_time_seconds": eval_data.get("eval_time_seconds"),
                "epoch": checkpoint_info.get("epoch"),
                "best_acc": checkpoint_info.get("best_acc"),
            }
        )
        per_class = data.get("per_class_accuracy") or {}
        for class_name, acc in per_class.items():
            per_class_rows.append(
                {
                    "file": path.name,
                    "model": data.get("model"),
                    "class": class_name,
                    "accuracy": acc,
                }
            )
    eval_df = pd.DataFrame(rows)
    if not eval_df.empty:
        eval_df["timestamp"] = pd.to_datetime(
            eval_df["timestamp_raw"], format="%Y%m%d_%H%M%S", errors="coerce"
        )
        eval_df.sort_values("timestamp", inplace=True)
    per_class_df = pd.DataFrame(per_class_rows)
    return eval_df, per_class_df


def plot_eval_overview(eval_df: pd.DataFrame, per_class_df: pd.DataFrame, out_dir: Path) -> None:
    if eval_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=eval_df, x="timestamp", y="accuracy", hue="model", marker="o", ax=ax)
    ax.set_title("Evaluation Accuracy Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_accuracy_over_time.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=eval_df, x="timestamp", y="loss", hue="model", marker="o", ax=ax)
    ax.set_title("Evaluation Loss Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_loss_over_time.png", dpi=300)
    plt.close(fig)

    grouped = eval_df.groupby("model", as_index=False)["accuracy"].max()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=grouped.sort_values("accuracy", ascending=False), x="model", y="accuracy", ax=ax)
    ax.set_title("Best Test Accuracy by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_best_accuracy_bar.png", dpi=300)
    plt.close(fig)

    if not per_class_df.empty:
        pivot = per_class_df.pivot_table(index="class", columns="model", values="accuracy")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
        ax.set_title("Per-Class Accuracy Heatmap")
        ax.set_xlabel("Model")
        ax.set_ylabel("Class")
        fig.tight_layout()
        fig.savefig(out_dir / "eval_per_class_heatmap.png", dpi=300)
        plt.close(fig)


def collect_train_histories(logs_dir: Path) -> List[Dict]:
    histories: List[Dict] = []
    for path in sorted(logs_dir.rglob("train_history.json")):
        model_name = path.parent.name
        df = pd.DataFrame(load_json(path))
        if df.empty:
            continue
        if "epoch" in df.columns:
            df = df.sort_values("epoch")
        df["model"] = model_name
        histories.append({"model": model_name, "path": path, "df": df})
    return histories


def plot_train_histories(histories: List[Dict], out_dir: Path) -> None:
    if not histories:
        return

    merged_frames = []
    for item in histories:
        df = item["df"]
        model = item["model"]
        merged_frames.append(df)

        if {"epoch", "train_acc", "val_acc"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x="epoch", y="train_acc", label="Train Accuracy", ax=ax)
            sns.lineplot(data=df, x="epoch", y="val_acc", label="Validation Accuracy", ax=ax)
            ax.set_title(f"{model} Accuracy Curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(out_dir / f"{model}_accuracy_curve.png", dpi=300)
            plt.close(fig)

        if {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss", ax=ax)
            sns.lineplot(data=df, x="epoch", y="val_loss", label="Validation Loss", ax=ax)
            ax.set_title(f"{model} Loss Curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            fig.savefig(out_dir / f"{model}_loss_curve.png", dpi=300)
            plt.close(fig)

    if not merged_frames:
        return

    combined = pd.concat(merged_frames, ignore_index=True)

    if {"epoch", "val_acc", "model"}.issubset(combined.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=combined, x="epoch", y="val_acc", hue="model", ax=ax)
        ax.set_title("Validation Accuracy Comparison")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(out_dir / "val_accuracy_comparison.png", dpi=300)
        plt.close(fig)

    if {"epoch", "train_loss", "model"}.issubset(combined.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=combined, x="epoch", y="train_loss", hue="model", ax=ax)
        ax.set_title("Train Loss Comparison")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(out_dir / "train_loss_comparison.png", dpi=300)
        plt.close(fig)


def collect_cv_results(logs_dir: Path) -> List[Dict]:
    cv_items: List[Dict] = []
    for path in sorted(logs_dir.rglob("cv_results_*.json")):
        data = load_json(path)
        records = data.get("records", [])
        if not records:
            continue
        df = pd.json_normalize(records)
        df = df.rename(columns={col: col.split("params.", 1)[1] for col in df.columns if col.startswith("params.")})
        if "fold_scores" in df.columns:
            df = df.drop(columns=["fold_scores"])
        df["source_file"] = path.name
        cv_items.append({"path": path, "summary": data, "df": df})
    return cv_items


def _combine_param_columns(df: pd.DataFrame, param_cols: List[str]) -> pd.Series:
    def _summarize(row) -> str:
        parts = []
        for col in param_cols:
            value = row.get(col)
            if isinstance(value, float):
                value = f"{value:.4g}"
            parts.append(f"{col}={value}")
        return ", ".join(parts)

    return df.apply(_summarize, axis=1)


def plot_cv_results(cv_items: List[Dict], out_dir: Path) -> None:
    for item in cv_items:
        df = item["df"]
        source = item["path"].stem
        metric_cols = {"mean_accuracy", "std_accuracy", "time_minutes", "source_file"}
        param_cols = [col for col in df.columns if col not in metric_cols]

        if "mean_accuracy" not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        top_df = df.nlargest(10, "mean_accuracy").copy()
        if param_cols:
            top_df["param_combo"] = _combine_param_columns(top_df, param_cols)
            y_col = "param_combo"
        else:
            y_col = "source_file"
        sns.barplot(data=top_df, x="mean_accuracy", y=y_col, ax=ax, color="#4c72b0")
        ax.set_title(f"{source} Top 10 Mean Accuracy")
        ax.set_xlabel("Mean Accuracy")
        ax.set_ylabel("Parameter Combo")
        fig.tight_layout()
        fig.savefig(out_dir / f"{source}_top10_bar.png", dpi=300)
        plt.close(fig)

        if len(param_cols) < 2:
            continue
        x_col, y_col = param_cols[:2]
        pivot = df.copy()
        pivot[x_col] = pivot[x_col].astype(str)
        pivot[y_col] = pivot[y_col].astype(str)
        pivot = pivot.pivot_table(index=y_col, columns=x_col, values="mean_accuracy", aggfunc="max")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="magma", ax=ax)
        ax.set_title(f"{source} Mean Accuracy Heatmap")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()
        fig.savefig(out_dir / f"{source}_heatmap.png", dpi=300)
        plt.close(fig)


def ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training and evaluation artifacts from eval_results and logs.")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_results"), help="Path to eval_results directory")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Path to logs directory")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"), help="Directory where figures are saved")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    eval_df, per_class_df = collect_eval_results(args.eval_dir)
    plot_eval_overview(eval_df, per_class_df, args.output_dir)

    histories = collect_train_histories(args.logs_dir)
    plot_train_histories(histories, args.output_dir)

    cv_items = collect_cv_results(args.logs_dir)
    plot_cv_results(cv_items, args.output_dir)

    print(f"Figures saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
