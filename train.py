
"""CIFAR-10 卷积神经网络训练与调参脚本。"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.cifar10 import load_cifar10
from models import Cifar10CNN, ModelConfig, ResNet18, WideResNet28_10

# 检查 PyTorch 版本以使用正确的 AMP API
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
USE_NEW_AMP_API = PYTORCH_VERSION >= (1, 13)  # PyTorch 1.13+ 使用新 API

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("警告: PyYAML 未安装，无法使用 YAML 配置文件。请运行: pip install pyyaml")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def setup_logger(log_dir: str | None = None) -> logging.Logger:
    """设置日志记录器。"""
    logger = logging.getLogger("CIFAR10_Training")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"日志保存到: {log_file}")
    
    return logger


def create_model(model_name: str, model_config: ModelConfig | None = None, num_classes: int = 10) -> nn.Module:
    """根据模型名称创建模型。"""
    if model_name == "cnn":
        if model_config is None:
            model_config = ModelConfig(num_classes=num_classes)
        return Cifar10CNN(model_config)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "wideresnet28_10":
        return WideResNet28_10(num_classes=num_classes, dropout=0.3)
    else:
        raise ValueError(f"未知的模型类型: {model_name}")


@dataclass
class TrainMetrics:
    """训练或验证过程中收集的指标。"""

    loss: float
    accuracy: float


class NumpyCifarDataset(Dataset):
    """将 NumPy 形式的图像封装为 PyTorch Dataset。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: transforms.Compose | None = None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index]).float()
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels[index])
        return image, label


def set_seed(seed: int | None) -> None:
    """设置随机种子确保复现。"""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """构建训练与验证阶段所需的变换。"""
    if augment:
        train_sequence = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    else:
        train_sequence = [transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]

    eval_sequence = [transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(train_sequence), transforms.Compose(eval_sequence)


def build_dataloaders(
    batch_size: int,
    num_workers: int,
    augment: bool,
    *,
    validation_ratio: float,
    seed: int | None,
) -> Dict[str, DataLoader]:
    """加载 CIFAR-10 数据并返回对应的 DataLoader。"""
    data_splits = load_cifar10(validation_ratio=validation_ratio, flatten=False, dtype=np.float32)
    train_transform, eval_transform = build_transforms(augment)

    train_images, train_labels = data_splits["train"]
    train_dataset = NumpyCifarDataset(train_images, train_labels, transform=train_transform)

    loaders: Dict[str, DataLoader] = {}
    pin_memory = torch.cuda.is_available()

    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if "val" in data_splits:
        val_images, val_labels = data_splits["val"]
        val_dataset = NumpyCifarDataset(val_images, val_labels, transform=eval_transform)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_images, test_labels = data_splits["test"]
    test_dataset = NumpyCifarDataset(test_images, test_labels, transform=eval_transform)
    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loaders


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
) -> TrainMetrics:
    """执行一个训练轮次。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # 使用新旧 API 兼容的写法
            if USE_NEW_AMP_API:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return TrainMetrics(loss=running_loss / total, accuracy=correct / total)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> TrainMetrics:
    """在验证或测试集上评估模型。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return TrainMetrics(loss=running_loss / total, accuracy=correct / total)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_val_acc: float,
    checkpoint_dir: str,
    model_name: str,
    *,
    is_best: bool = False,
    extra_filename: str | None = None,
) -> None:
    """保存训练检查点。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "model_name": model_name,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # 保存最新的检查点
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # 可选地额外保存一份带有自定义文件名的检查点
    if extra_filename:
        extra_path = os.path.join(checkpoint_dir, extra_filename)
        torch.save(checkpoint, extra_path)
    
    # 如果是最佳模型，也保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> Tuple[int, float]:
    """加载训练检查点。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    start_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint["best_val_acc"]
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return start_epoch, best_val_acc


def run_training(args: argparse.Namespace) -> None:
    """执行标准训练流程。"""
    # 设置日志
    logger = setup_logger(args.log_dir)
    
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"使用设备: {device}")
    logger.info(f"模型类型: {args.model}")

    loaders = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        validation_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 创建模型
    model_config = None  # 初始化为 None，避免作用域问题
    if args.model == "cnn":
        model_config = ModelConfig(
            num_classes=10,
            dropout=args.dropout,
            linear_dropout=args.linear_dropout,
            normalization=args.normalization,
            group_norm_groups=args.group_norm_groups,
        )
        model = create_model(args.model, model_config=model_config)
    else:
        model = create_model(args.model, num_classes=10)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # 创建混合精度训练的 GradScaler（兼容新旧 API）
    scaler = None
    if args.amp and device.type in ["cuda", "cpu"]:
        if USE_NEW_AMP_API:
            scaler = torch.amp.GradScaler(device.type)
        else:
            scaler = torch.cuda.amp.GradScaler()

    # 从检查点恢复
    start_epoch = 1
    best_val_acc = -math.inf
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"从检查点恢复: {args.resume}")
            start_epoch, best_val_acc = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch += 1  # 从下一个 epoch 开始
            logger.info(f"从 epoch {start_epoch} 继续训练，当前最佳验证准确率: {best_val_acc:.4f}")
        else:
            logger.warning(f"检查点文件不存在: {args.resume}，从头开始训练")

    best_state = None
    train_history = []
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler=scaler)
        val_metrics = evaluate(model, loaders["val"], criterion, device) if "val" in loaders else None
        if scheduler is not None:
            scheduler.step()

        lr_current = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch:02d}/{args.epochs} - lr={lr_current:.4g} - "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f}"
        )
        
        is_best = False
        if val_metrics is not None:
            message += f" - val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_state = model.state_dict()
                is_best = True
                message += " [新最佳]"
        
        logger.info(message)
        
        # 保存训练历史
        epoch_record = {
            "epoch": epoch,
            "lr": lr_current,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
        }
        if val_metrics is not None:
            epoch_record["val_loss"] = val_metrics.loss
            epoch_record["val_acc"] = val_metrics.accuracy
        train_history.append(epoch_record)
        
        # 保存检查点
        if args.checkpoint_dir:
            # 定期保存检查点
            if epoch % args.save_freq == 0:
                save_checkpoint(
                    epoch, model, optimizer, scheduler, best_val_acc,
                    args.checkpoint_dir, args.model, is_best=False,
                    extra_filename=f"checkpoint_epoch_{epoch:04d}.pth"
                )
                logger.info(f"检查点已保存到: {args.checkpoint_dir}")
            
            # 如果是最佳模型，立即保存
            if is_best:
                save_checkpoint(
                    epoch, model, optimizer, scheduler, best_val_acc,
                    args.checkpoint_dir, args.model, is_best=True
                )
                logger.info(f"最佳模型已保存 (epoch {epoch}, acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time
    logger.info(f"训练耗时 {elapsed / 60:.2f} 分钟")
    
    # 训练结束后保存最终检查点
    if args.checkpoint_dir:
        save_checkpoint(
            args.epochs, model, optimizer, scheduler, best_val_acc,
            args.checkpoint_dir, args.model, is_best=False
        )
        logger.info(f"最终检查点已保存到: {args.checkpoint_dir}")

    # 保存训练历史
    if args.log_dir:
        history_path = os.path.join(args.log_dir, "train_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, indent=2, ensure_ascii=False)
        logger.info(f"训练历史已保存到: {history_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"验证集最佳准确率: {best_val_acc:.4f}")

    test_metrics = evaluate(model, loaders["test"], criterion, device)
    logger.info(f"测试集: loss={test_metrics.loss:.4f} acc={test_metrics.accuracy:.4f}")

    if args.save_path:
        save_dict = {"model": model.state_dict(), "model_name": args.model}
        if args.model == "cnn" and model_config is not None:
            save_dict["config"] = model_config.__dict__
        torch.save(save_dict, args.save_path)
        logger.info(f"最终模型已保存到: {args.save_path}")


def _product_dict(grid: Dict[str, Sequence]) -> Iterable[Dict[str, float | str]]:
    """构建参数网格的笛卡尔积。"""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=True))


def build_cv_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> Tuple[DataLoader, DataLoader]:
    """基于给定索引构建交叉验证使用的 DataLoader。"""
    train_transform, eval_transform = build_transforms(augment)
    train_dataset = NumpyCifarDataset(images[train_indices], labels[train_indices], transform=train_transform)
    val_dataset = NumpyCifarDataset(images[val_indices], labels[val_indices], transform=eval_transform)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader




def run_cross_validation(args: argparse.Namespace) -> None:
    """执行交叉验证并记录结果。"""
    logger = setup_logger(args.log_dir)

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"交叉验证使用设备: {device}")
    logger.info(f"交叉验证折数: {args.kfolds}")
    logger.info(f"每折训练轮数: {args.cv_epochs}")

    data = load_cifar10(validation_ratio=0.0, flatten=False, dtype=np.float32)
    train_images, train_labels = data["train"]

    rng = np.random.default_rng(args.seed)
    num_samples = train_images.shape[0]
    if args.cv_sample_size is not None and 0 < args.cv_sample_size < num_samples:
        chosen = rng.choice(num_samples, size=args.cv_sample_size, replace=False)
        train_images = train_images[chosen]
        train_labels = train_labels[chosen]
        num_samples = train_images.shape[0]
        logger.info(f"交叉验证采样 {num_samples} 个样本用于网格搜索")
    else:
        logger.info(f"使用全部训练数据: {num_samples} 个样本")

    indices = np.arange(num_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, args.kfolds)

    grid = {
        "lr": args.lr_grid,
        "dropout": args.dropout_grid,
        "linear_dropout": args.linear_dropout_grid,
        "normalization": args.norm_grid,
        "weight_decay": args.weight_decay_grid,
    }

    # 计算总组合数
    total_combinations = 1
    for values in grid.values():
        total_combinations *= len(values)
    logger.info(f"=== 开始交叉验证网格搜索 ===")
    logger.info(f"搜索空间: {grid}")
    logger.info(f"总组合数: {total_combinations}")
    logger.info(f"预计训练次数: {total_combinations * args.kfolds}")
    logger.info("=" * 50)

    best_combo: Dict[str, float | str] | None = None
    best_score = -math.inf
    records: List[Dict[str, object]] = []
    
    combo_idx = 0
    start_time = time.time()
    
    for combo in _product_dict(grid):
        combo_idx += 1
        combo_start_time = time.time()
        logger.info(f"\n[{combo_idx}/{total_combinations}] 评估组合: {combo}")
        fold_scores: List[float] = []
        for fold_id in range(args.kfolds):
            val_indices = folds[fold_id]
            train_indices = np.concatenate([folds[i] for i in range(args.kfolds) if i != fold_id])

            train_loader, val_loader = build_cv_loaders(
                train_images,
                train_labels,
                train_indices,
                val_indices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                augment=not args.no_augment,
            )

            model_config = ModelConfig(
                num_classes=10,
                dropout=float(combo["dropout"]),
                linear_dropout=float(combo["linear_dropout"]),
                normalization=str(combo["normalization"]),
                group_norm_groups=args.group_norm_groups,
            )
            model = Cifar10CNN(model_config).to(device)
            optimizer = AdamW(
                model.parameters(),
                lr=float(combo["lr"]),
                weight_decay=float(combo["weight_decay"]),
            )
            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, args.cv_epochs + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, criterion, device)
            fold_scores.append(metrics.accuracy)
            logger.info(f"  折 {fold_id + 1}/{args.kfolds}: acc={metrics.accuracy:.4f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        combo_elapsed = time.time() - combo_start_time
        logger.info(f"组合结果: 平均准确率={avg_score:.4f} ± {std_score:.4f}, 耗时={combo_elapsed/60:.2f}分钟")
        
        combo_serializable = {
            key: float(value) if isinstance(value, (np.floating, float, int)) else str(value)
            for key, value in combo.items()
        }
        records.append(
            {
                "params": combo_serializable,
                "fold_scores": [float(score) for score in fold_scores],
                "mean_accuracy": avg_score,
                "std_accuracy": std_score,
                "time_minutes": combo_elapsed / 60,
            }
        )
        
        # 保存中间结果（防止程序崩溃丢失数据）
        if args.log_dir and combo_idx % 5 == 0:
            intermediate_path = Path(args.log_dir) / "cv_results_intermediate.json"
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            with open(intermediate_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "processed_combinations": combo_idx,
                    "total_combinations": total_combinations,
                    "records": records,
                }, fp, ensure_ascii=False, indent=2)
            logger.info(f"中间结果已保存 ({combo_idx}/{total_combinations})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_combo = combo_serializable
            logger.info(f"🎉 发现新的最佳组合！准确率: {best_score:.4f}")

    total_elapsed = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info("=== 交叉验证完成 ===")
    logger.info(f"总耗时: {total_elapsed/60:.2f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    logger.info(f"最佳组合: {best_combo}")
    logger.info(f"最佳准确率: {best_score:.4f}")
    
    # 显示 Top 5 结果
    sorted_records = sorted(records, key=lambda x: x["mean_accuracy"], reverse=True)
    logger.info("\n=== Top 5 最佳组合 ===")
    for i, record in enumerate(sorted_records[:5], 1):
        logger.info(f"{i}. 准确率={record['mean_accuracy']:.4f} ± {record['std_accuracy']:.4f}, 参数={record['params']}")

    if args.log_dir:
        timestamp = time.strftime("cv_results_%Y%m%d_%H%M%S.json")
        output_path = Path(args.log_dir) / timestamp
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_combo": best_combo,
            "best_score": float(best_score),
            "kfolds": args.kfolds,
            "cv_epochs": args.cv_epochs,
            "total_combinations": total_combinations,
            "total_time_minutes": total_elapsed / 60,
            "records": records,
            "top_5": sorted_records[:5],
        }
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        logger.info(f"\n最终结果已保存到: {output_path}")




def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CIFAR-10 卷积神经网络训练脚本")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None, 
                       help="配置文件路径 (.yaml 或 .json)，如果指定则从配置文件加载参数")
    
    # 模型选择
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet18", "wideresnet28_10"], 
                       help="选择模型类型: cnn (默认CNN), resnet18 (ResNet-18), wideresnet28_10 (WideResNet-28-10)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--cv-epochs", type=int, default=5, help="交叉验证每个折内的训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减系数")
    
    # CNN 模型特定参数
    parser.add_argument("--dropout", type=float, default=0.3, help="卷积块 Dropout 概率 (仅用于CNN模型)")
    parser.add_argument("--linear-dropout", type=float, default=0.5, help="全连接层 Dropout 概率 (仅用于CNN模型)")
    parser.add_argument("--normalization", type=str, default="batch", choices=["batch", "layer", "group", "none"], 
                       help="归一化类型 (仅用于CNN模型)")
    parser.add_argument("--group-norm-groups", type=int, default=8, help="GroupNorm 分组数 (仅用于CNN模型)")
    
    # 学习率调度
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step"], help="学习率调度策略")
    
    # 数据加载
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 工作线程数")
    parser.add_argument("--device", type=str, default=None, help="指定训练设备，如 cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="训练集划分验证集比例")
    parser.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    
    # 保存与恢复
    parser.add_argument("--save-path", type=str, default=None, help="训练结束后保存模型的路径")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练的路径")
    parser.add_argument("--save-freq", type=int, default=5, help="每隔多少个 epoch 保存一次检查点")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
    
    # 其他
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    
    # 交叉验证参数
    parser.add_argument("--cross-validate", action="store_true", help="启用交叉验证模式")
    parser.add_argument("--kfolds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 5e-4], help="交叉验证学习率候选")
    parser.add_argument("--dropout-grid", type=float, nargs="+", default=[0.2, 0.3], help="卷积 Dropout 候选")
    parser.add_argument("--linear-dropout-grid", type=float, nargs="+", default=[0.4, 0.5], help="全连接 Dropout 候选")
    parser.add_argument("--norm-grid", type=str, nargs="+", default=["batch", "group"], help="归一化类型候选")
    parser.add_argument("--weight-decay-grid", type=float, nargs="+", default=[1e-4, 5e-4], help="权重衰减候选")
    parser.add_argument("--cv-sample-size", type=int, default=None, help="交叉验证时可选的样本子集大小")
    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, any]:
    """从配置文件加载参数。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    ext = os.path.splitext(config_path)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError("使用 YAML 配置文件需要安装 PyYAML: pip install pyyaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif ext == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}，请使用 .yaml, .yml 或 .json")


def merge_config_and_args(config_dict: Dict[str, any], args: argparse.Namespace) -> argparse.Namespace:
    """合并配置文件和命令行参数（命令行参数优先）。"""
    # 获取命令行中明确设置的参数（非默认值）
    parser = argparse.ArgumentParser()
    defaults = {}
    
    # 首先用配置文件的值更新 args
    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args


def main() -> None:
    args = parse_args()
    
    # 如果指定了配置文件，先加载配置文件
    if args.config:
        print(f"从配置文件加载参数: {args.config}")
        config_dict = load_config_file(args.config)
        
        # 将配置文件的值合并到 args 中
        # 注意：命令行显式指定的参数应该覆盖配置文件
        config_model = config_dict.get('model', args.model)
        
        # 保存命令行中显式设置的值
        cli_config_path = args.config
        
        # 用配置文件更新 args
        for key, value in config_dict.items():
            if hasattr(args, key):
                # 特殊处理：如果命令行没有指定 config 以外的参数，使用配置文件的值
                setattr(args, key, value)
        
        # 恢复 config 路径
        args.config = cli_config_path
        print(f"使用模型: {args.model}")
        print(f"训练轮数: {args.epochs}")
        print(f"批大小: {args.batch_size}")
        print(f"学习率: {args.lr}")
    
    if args.cross_validate:
        run_cross_validation(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
