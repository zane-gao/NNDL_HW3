
"""基于高级数据增强的 CIFAR-10 微调训练脚本。"""

from __future__ import annotations

import argparse
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.augmentation import AugmentationConfig, build_advanced_transforms, print_augmentation_info
from data.cifar10 import load_cifar10
from models import Cifar10CNN, ModelConfig, ResNet18, WideResNet28_10

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("警告: PyYAML 未安装，无法使用 YAML 配置文件。请运行: pip install pyyaml")




def setup_logger(log_dir: str | None = None) -> logging.Logger:
    """初始化日志记录器，支持控制台与可选文件输出。"""
    logger = logging.getLogger("CIFAR10_Finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"finetune_{timestamp}.log"
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
class NumpyCifarDataset(Dataset):
    """使用 NumPy 数据构建的简单 Dataset 封装。"""

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


def set_seed(seed: int | None) -> None:
    """设置随机种子，保证实验结果可复现。"""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(args: argparse.Namespace, aug_config: AugmentationConfig, logger: logging.Logger) -> Dict[str, DataLoader]:
    """根据配置构建训练/验证/测试数据加载器。"""
    train_transform, eval_transform = build_advanced_transforms(aug_config)
    if args.print_aug:
        print_augmentation_info(aug_config, logger=logger)

    dataset = load_cifar10(validation_ratio=args.val_ratio, flatten=False, dtype=np.float32)
    pin_memory = torch.cuda.is_available()

    loaders: Dict[str, DataLoader] = {}
    train_images, train_labels = dataset["train"]
    train_dataset = NumpyCifarDataset(train_images, train_labels, transform=train_transform)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if "val" in dataset:
        val_images, val_labels = dataset["val"]
        val_dataset = NumpyCifarDataset(val_images, val_labels, transform=eval_transform)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    test_images, test_labels = dataset["test"]
    test_dataset = NumpyCifarDataset(test_images, test_labels, transform=eval_transform)
    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return loaders


def build_model(args: argparse.Namespace) -> nn.Module:
    """根据命令行参数创建模型实例。"""
    model_name = args.model.lower()
    if model_name == "cnn":
        config = ModelConfig(
            num_classes=10,
            dropout=args.cnn_dropout,
            linear_dropout=args.cnn_linear_dropout,
            normalization=args.cnn_norm,
            group_norm_groups=args.cnn_group_norm_groups,
        )
        model = Cifar10CNN(config)
    elif model_name == "resnet18":
        model = ResNet18(num_classes=10)
    elif model_name == "wideresnet28_10":
        model = WideResNet28_10(num_classes=10)
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")

    if args.freeze_features:
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """冻结特征提取部分，仅训练分类头，实现简单的微调。"""
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("当前模型缺少可识别的分类头属性，无法仅训练分类头")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """执行单个 epoch 的训练，返回平均损失和准确率。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """在验证或测试集上评估模型表现。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float, path: Path) -> None:
    """保存训练 checkpoint，便于后续恢复。"""
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CIFAR-10 高级数据增强与微调训练脚本")

    # 配置文件
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (.yaml 或 .json)，如果指定则从配置文件加载参数")

    # 基础训练参数
    parser.add_argument("--model", type=str, default="resnet18", choices=["cnn", "resnet18", "wideresnet28_10"], help="选择要训练的模型")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减系数")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="训练集划分验证集比例")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 使用的线程数")
    parser.add_argument("--device", type=str, default=None, help="指定训练设备，默认自动检测")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 模型微调相关
    parser.add_argument("--freeze-features", action="store_true", help="冻结特征提取层，仅训练分类头")

    # CNN 专属超参
    parser.add_argument("--cnn-dropout", type=float, default=0.3, help="CNN 卷积块 Dropout 概率")
    parser.add_argument("--cnn-linear-dropout", type=float, default=0.5, help="CNN 全连接层 Dropout 概率")
    parser.add_argument("--cnn-norm", type=str, default="batch", choices=["batch", "layer", "group", "none"], help="CNN 归一化方式")
    parser.add_argument("--cnn-group-norm-groups", type=int, default=8, help="CNN GroupNorm 分组数")

    # 数据增强相关
    parser.add_argument("--no-augment", action="store_true", help="关闭训练阶段的随机增强")
    parser.add_argument("--augment-policy", type=str, default="advanced", choices=["basic", "advanced", "autoaugment", "custom"], help="增强策略预设")
    parser.add_argument("--autoaugment", action="store_true", help="强制启用 AutoAugment")
    parser.add_argument("--color-jitter", action="store_true", help="启用颜色扰动")
    parser.add_argument("--random-rotation", action="store_true", help="启用随机旋转")
    parser.add_argument("--gaussian-blur", action="store_true", help="启用高斯模糊")
    parser.add_argument("--cutout", action="store_true", help="启用 Cutout 增强")
    parser.add_argument("--random-erasing", action="store_true", help="启用 RandomErasing 增强")
    parser.add_argument("--cutout-holes", type=int, default=1, help="Cutout 遮挡块数量")
    parser.add_argument("--cutout-length", type=int, default=16, help="Cutout 遮挡块边长")
    parser.add_argument("--rotation-degrees", type=int, default=15, help="随机旋转角度范围")
    parser.add_argument("--gaussian-blur-prob", type=float, default=0.5, help="高斯模糊应用概率")
    parser.add_argument("--random-erasing-prob", type=float, default=0.5, help="RandomErasing 应用概率")
    parser.add_argument("--print-aug", action="store_true", help="打印增强策略详情")

    # Checkpoint 设置
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="保存 checkpoint 的目录")
    parser.add_argument("--resume", type=str, default=None, help="可选的 checkpoint 路径，用于恢复训练")
    parser.add_argument("--save-freq", type=int, default=5, help="每隔多少个 epoch 额外保存一次检查点")
    parser.add_argument("--log-dir", type=str, default=None, help="训练日志保存目录")

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


def merge_config_with_args(config_dict: Dict[str, any], args: argparse.Namespace) -> argparse.Namespace:
    """合并配置文件和命令行参数。
    
    配置文件中的参数会覆盖默认值，但命令行显式指定的参数优先级最高。
    特殊处理：配置文件中的 dropout/linear_dropout 映射到 cnn-dropout/cnn-linear-dropout。
    """
    # 参数映射：配置文件中的键 -> argparse 中的属性名
    param_mapping = {
        'dropout': 'cnn_dropout',
        'linear_dropout': 'cnn_linear_dropout',
        'normalization': 'cnn_norm',
        'group_norm_groups': 'cnn_group_norm_groups',
        'batch_size': 'batch_size',
        'weight_decay': 'weight_decay',
        'num_workers': 'num_workers',
        'val_ratio': 'val_ratio',
        'no_augment': 'no_augment',
    }
    
    # 应用配置文件中的参数
    for key, value in config_dict.items():
        # 使用映射表转换键名
        attr_name = param_mapping.get(key, key.replace('-', '_'))
        
        if hasattr(args, attr_name):
            setattr(args, attr_name, value)
    
    return args


def build_aug_config(args: argparse.Namespace) -> AugmentationConfig:
    """依据命令行参数生成数据增强配置。"""
    return AugmentationConfig(
        augment=not args.no_augment,
        augment_policy=args.augment_policy,
        cutout=args.cutout,
        cutout_holes=args.cutout_holes,
        cutout_length=args.cutout_length,
        random_erasing=args.random_erasing,
        random_erasing_prob=args.random_erasing_prob,
        autoaugment=args.autoaugment,
        color_jitter=args.color_jitter,
        random_rotation=args.random_rotation,
        rotation_degrees=args.rotation_degrees,
        gaussian_blur=args.gaussian_blur,
        gaussian_blur_prob=args.gaussian_blur_prob,
    )


def resume_if_needed(model: nn.Module, optimizer: torch.optim.Optimizer, path: str | None, device: torch.device, logger: logging.Logger) -> Tuple[int, float]:
    """如果提供了 checkpoint，则加载状态并返回起始 epoch 与当前最佳准确率。
    
    兼容两种 checkpoint 格式：
    - train.py 保存的格式：model_state_dict, optimizer_state_dict, best_val_acc
    - train_finetune.py 保存的格式：model_state, optimizer_state, best_acc
    """
    start_epoch = 1
    best_acc = 0.0
    if path is None:
        return start_epoch, best_acc

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint 文件: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    
    # 兼容不同的键名
    model_state_key = "model_state" if "model_state" in payload else "model_state_dict"
    optimizer_state_key = "optimizer_state" if "optimizer_state" in payload else "optimizer_state_dict"
    best_acc_key = "best_acc" if "best_acc" in payload else "best_val_acc"
    
    model.load_state_dict(payload[model_state_key])
    
    # optimizer 可能不存在（例如只保存了模型）
    if optimizer_state_key in payload:
        optimizer.load_state_dict(payload[optimizer_state_key])
    
    start_epoch = int(payload.get("epoch", 0)) + 1
    best_acc = float(payload.get(best_acc_key, 0.0))
    logger.info(f"已从 {checkpoint_path} 恢复训练，起始 epoch={start_epoch}，最佳准确率={best_acc:.4f}")
    return start_epoch, best_acc



def main() -> None:
    args = parse_args()

    config_loaded = False
    config_path = args.config
    if config_path:
        config_dict = load_config_file(config_path)
        args = merge_config_with_args(config_dict, args)
        config_loaded = True

    set_seed(args.seed)

    logger = setup_logger(args.log_dir)
    if config_loaded and config_path:
        logger.info(f"从配置文件加载参数: {config_path}")

    logger.info(f"使用模型: {args.model}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    aug_config = build_aug_config(args)
    loaders = build_dataloaders(args, aug_config, logger)

    model = build_model(args).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_val_acc = resume_if_needed(model, optimizer, args.resume, device, logger)

    if start_epoch > args.epochs:
        logger.info("checkpoint 的 epoch 已超过目标训练轮数，无需继续训练。")
        return

    best_checkpoint_path = None
    latest_checkpoint_path = None
    ckpt_dir: Path | None = None
    if args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = ckpt_dir / "checkpoint_best.pth"
        latest_checkpoint_path = ckpt_dir / "checkpoint_latest.pth"
        logger.info(f"checkpoint 将保存至: {ckpt_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device) if "val" in loaders else (0.0, 0.0)

        message = (
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        logger.info(message)

        if latest_checkpoint_path is not None:
            save_checkpoint(model, optimizer, epoch, best_val_acc, latest_checkpoint_path)
            logger.info(f"已更新最新检查点: {latest_checkpoint_path}")
            if args.save_freq > 0 and ckpt_dir is not None and epoch % args.save_freq == 0:
                monitor_acc = val_acc if "val" in loaders else best_val_acc
                periodic_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth"
                save_checkpoint(model, optimizer, epoch, monitor_acc, periodic_path)
                logger.info(f"已按周期保存检查点: {periodic_path}")

        if "val" in loaders and val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_checkpoint_path is not None:
                save_checkpoint(model, optimizer, epoch, best_val_acc, best_checkpoint_path)
                logger.info(f"验证集最佳提升，已保存 checkpoint: {best_checkpoint_path} (acc={best_val_acc:.4f})")

    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    logger.info(f"测试集: loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
