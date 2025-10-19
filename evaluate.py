
"""模型评估脚本：用于在 CIFAR-10 上评估已训练权重。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.augmentation import build_advanced_transforms
from data.cifar10 import load_cifar10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

try:
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models import Cifar10CNN, ModelConfig, ResNet18, WideResNet28_10


class NumpyCifarDataset(Dataset):
    """基于 NumPy 数据的 Dataset 封装。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: transforms.Compose | None = None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index]).float()
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels[index])
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估训练好的 CIFAR-10 模型")
    parser.add_argument("--weights", type=str, required=True, help="权重文件路径（.pth / checkpoint）")
    parser.add_argument("--model", type=str, default="resnet18", choices=["cnn", "resnet18", "wideresnet28_10"], help="模型类型")
    parser.add_argument("--batch-size", type=int, default=256, help="评估批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--device", type=str, default=None, help="评估设备，默认自动检测")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="评估数据划分")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="当需要验证集时的划分比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--config", type=str, default=None, help="可选的 YAML/JSON 配置文件")
    
    # 结果保存
    parser.add_argument("--output-dir", type=str, default="eval_results", help="评估结果保存目录")
    parser.add_argument("--save-predictions", action="store_true", help="是否保存详细的预测结果")
    parser.add_argument("--no-save", action="store_true", help="不保存评估结果（仅打印）")

    # CNN 专属参数
    parser.add_argument("--cnn-dropout", type=float, default=0.3, help="CNN 卷积块 Dropout")
    parser.add_argument("--cnn-linear-dropout", type=float, default=0.5, help="CNN 全连接 Dropout")
    parser.add_argument("--cnn-norm", type=str, default="batch", choices=["batch", "layer", "group", "none"], help="CNN 归一化类型")
    parser.add_argument("--cnn-group-norm-groups", type=int, default=8, help="CNN GroupNorm 分组数")

    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, object]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - 仅在缺依赖时触发
            raise ImportError("评估脚本需要 PyYAML 来解析 YAML 配置，请安装: pip install pyyaml") from exc
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"不支持的配置文件格式: {ext}")


def merge_config(args: argparse.Namespace, config: Dict[str, object]) -> None:
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args: argparse.Namespace) -> nn.Module:
    name = args.model.lower()
    if name == "cnn":
        config = ModelConfig(
            num_classes=10,
            dropout=args.cnn_dropout,
            linear_dropout=args.cnn_linear_dropout,
            normalization=args.cnn_norm,
            group_norm_groups=args.cnn_group_norm_groups,
        )
        return Cifar10CNN(config)
    if name == "resnet18":
        return ResNet18(num_classes=10)
    if name == "wideresnet28_10":
        return WideResNet28_10(num_classes=10)
    raise ValueError(f"未知模型类型: {args.model}")


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> Tuple[int, float]:
    """加载模型权重，支持多种保存格式。"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    start_epoch = 0
    best_acc = 0.0

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_acc = float(checkpoint.get("best_val_acc", 0.0))
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_acc = float(checkpoint.get("best_acc", 0.0))
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "config" in checkpoint and isinstance(model, Cifar10CNN):
                # 可选：如需保证配置一致，可在此校验 config
                pass
        else:
            # 可能直接是 state_dict 存储
            try:
                model.load_state_dict(checkpoint)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("无法从提供的权重文件中解析模型参数，请检查保存格式。") from exc
    else:
        raise ValueError("权重文件格式不正确，预期为字典或 state_dict。")

    return start_epoch, best_acc



@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    """执行评估并返回损失、准确率以及标签/预测列表。"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_preds


def compute_per_class_accuracy(y_true: List[int], y_pred: List[int], num_classes: int = 10) -> Dict[int, float]:
    """计算每个类别的准确率。"""
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    
    for true_label, pred_label in zip(y_true, y_pred):
        per_class_total[true_label] += 1
        if true_label == pred_label:
            per_class_correct[true_label] += 1
    
    per_class_acc = {}
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_acc[i] = per_class_correct[i] / per_class_total[i]
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc


def save_evaluation_results(
    output_dir: str,
    args: argparse.Namespace,
    loss: float,
    acc: float,
    y_true: List[int],
    y_pred: List[int],
    start_epoch: int,
    best_acc: float,
    eval_time: float,
) -> None:
    """保存评估结果到文件。"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每类准确率
    per_class_acc = compute_per_class_accuracy(y_true, y_pred)
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 准备结果字典
    results = {
        "timestamp": timestamp,
        "model": args.model,
        "weights_path": args.weights,
        "split": args.split,
        "dataset_size": len(y_true),
        "evaluation": {
            "loss": float(loss),
            "accuracy": float(acc),
            "eval_time_seconds": float(eval_time),
        },
        "checkpoint_info": {
            "epoch": int(start_epoch) if start_epoch else None,
            "best_acc": float(best_acc) if best_acc else None,
        },
        "per_class_accuracy": {
            CIFAR10_CLASSES[i]: float(per_class_acc[i])
            for i in range(len(CIFAR10_CLASSES))
        },
        "config": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(args.device) if args.device else "auto",
        },
    }
    
    # 如果使用 CNN，添加 CNN 特定配置
    if args.model == "cnn":
        results["config"]["cnn"] = {
            "dropout": args.cnn_dropout,
            "linear_dropout": args.cnn_linear_dropout,
            "normalization": args.cnn_norm,
            "group_norm_groups": args.cnn_group_norm_groups,
        }
    
    # 保存主结果文件
    model_name = args.model
    result_filename = f"eval_{model_name}_{args.split}_{timestamp}.json"
    result_path = os.path.join(output_dir, result_filename)
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {result_path}")
    
    # 如果需要保存详细预测
    if args.save_predictions:
        predictions = {
            "true_labels": y_true,
            "predicted_labels": y_pred,
            "class_names": CIFAR10_CLASSES,
        }
        pred_filename = f"predictions_{model_name}_{args.split}_{timestamp}.json"
        pred_path = os.path.join(output_dir, pred_filename)
        
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"详细预测结果已保存到: {pred_path}")
    
    # 保存可读的文本报告
    report_filename = f"report_{model_name}_{args.split}_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CIFAR-10 模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"权重: {args.weights}\n")
        f.write(f"数据集: {args.split} ({len(y_true)} 样本)\n")
        f.write(f"设备: {args.device if args.device else 'auto'}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("整体性能\n")
        f.write("=" * 60 + "\n")
        f.write(f"损失 (Loss):     {loss:.4f}\n")
        f.write(f"准确率 (Acc):     {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"评估耗时:        {eval_time:.2f} 秒\n\n")
        
        if start_epoch or best_acc:
            f.write("检查点信息:\n")
            if start_epoch:
                f.write(f"  训练轮数: {start_epoch}\n")
            if best_acc:
                f.write(f"  最佳准确率: {best_acc:.4f}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("每类准确率\n")
        f.write("=" * 60 + "\n")
        for i, class_name in enumerate(CIFAR10_CLASSES):
            f.write(f"{class_name:12s}: {per_class_acc[i]:.4f} ({per_class_acc[i]*100:.2f}%)\n")
        f.write("\n")
        
        # 如果有 sklearn，添加分类报告
        if SKLEARN_AVAILABLE:
            from sklearn.metrics import classification_report
            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(CIFAR10_CLASSES))),
                target_names=CIFAR10_CLASSES,
                digits=4,
                zero_division=0,
            )
            f.write("=" * 60 + "\n")
            f.write("详细分类报告\n")
            f.write("=" * 60 + "\n")
            f.write(report)
            f.write("\n")
    
    print(f"文本报告已保存到: {report_path}")


def main() -> None:
    args = parse_args()
    if args.config:
        config_dict = load_config_file(args.config)
        merge_config(args, config_dict)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    set_seed(args.seed)

    # 构建评估数据集（仅使用归一化，不使用数据增强）
    _, eval_transform = build_advanced_transforms(
        augment=False,
        normalize=True,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )

    data = load_cifar10(validation_ratio=args.val_ratio, flatten=False, dtype=np.float32)
    if args.split not in data:
        raise ValueError(f"数据划分 {args.split} 不存在，请检查 val_ratio 或 split 设置。")

    images, labels = data[args.split]
    dataset = NumpyCifarDataset(images, labels, transform=eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args).to(device)
    start_epoch, best_acc = load_weights(model, args.weights, device)
    print(f"已加载权重: {args.weights}")
    if start_epoch:
        print(f"checkpoint 中记录的 epoch={start_epoch}")
    if best_acc:
        print(f"checkpoint 中记录的最佳准确率={best_acc:.4f}")

    print(f"\n开始评估 {args.split} 数据集...")
    eval_start_time = time.time()
    loss, acc, y_true, y_pred = evaluate(model, dataloader, device)
    eval_time = time.time() - eval_start_time
    
    print("\n=== 评估结果 ===")
    print(f"数据集: {args.split}")
    print(f"样本数: {len(y_true)}")
    print(f"损失:   {loss:.4f}")
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"耗时:   {eval_time:.2f} 秒")
    
    # 显示每类准确率
    per_class_acc = compute_per_class_accuracy(y_true, y_pred)
    print("\n=== 每类准确率 ===")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        print(f"{class_name:12s}: {per_class_acc[i]:.4f} ({per_class_acc[i]*100:.2f}%)")

    if SKLEARN_AVAILABLE:
        from sklearn.metrics import classification_report
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CIFAR10_CLASSES))),
            target_names=CIFAR10_CLASSES,
            digits=4,
            zero_division=0,
        )
        print("\n=== 详细分类报告 ===")
        print(report)
    else:
        print("\n提示: 未安装 scikit-learn，无法生成详细分类报告。")
        print("安装方法: pip install scikit-learn")
    
    # 保存结果
    if not args.no_save:
        save_evaluation_results(
            args.output_dir,
            args,
            loss,
            acc,
            y_true,
            y_pred,
            start_epoch,
            best_acc,
            eval_time,
        )


if __name__ == "__main__":
    main()
