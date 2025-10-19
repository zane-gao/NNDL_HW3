# CIFAR-10 图像分类训练项目

本项目实现了在 CIFAR-10 数据集上训练多种深度学习模型的完整流程，包括数据加载、模型训练、超参数调优、日志记录和断点重训等功能。

## 文档导航

- **[快速开始指南 (QUICKSTART.md)](QUICKSTART.md)** - 5 分钟快速上手
- **[配置文件说明 (configs/README.md)](configs/README.md)** - 详细的配置文件使用指南
- **[交叉验证指南 (CROSS_VALIDATION_GUIDE.md)](CROSS_VALIDATION_GUIDE.md)** - 超参数搜索完整教程
- **本文档** - 完整的项目文档

## 主要功能

- **多模型支持**: CNN、ResNet-18、WideResNet-28-10
- **高级数据增强**: AutoAugment、Cutout、RandomErasing、ColorJitter 等
- **配置文件系统**: YAML/JSON 配置，方便实验管理
- **训练日志**: 自动保存训练日志和历史
- **断点重训**: 完整的 checkpoint 系统
- **交叉验证**: 自动超参数搜索和网格搜索
- **模型评估**: 详细的评估报告和每类准确率
- **混合精度训练**: 支持 AMP 加速训练
- **多种学习率调度**: Cosine Annealing、StepLR

## 项目结构

```
HW3/
├── train.py                 # 主训练脚本
├── train_finetune.py        # 高级数据增强训练脚本
├── evaluate.py              # 模型评估脚本
├── configs/                 # 配置文件目录
│   ├── __init__.py         # 配置加载模块
│   ├── README.md           # 配置文件使用说明
│   ├── cnn_default.yaml    # CNN 默认配置
│   ├── cnn_cv_quick.yaml   # CNN 快速交叉验证配置
│   ├── cnn_cross_validation.yaml  # CNN 完整交叉验证配置
│   ├── cnn_advanced_augment.yaml  # CNN 高级数据增强配置
│   ├── resnet18_default.yaml      # ResNet-18 默认配置
│   ├── resnet18_quick_test.yaml   # ResNet-18 快速测试配置
│   ├── resnet18_advanced_augment.yaml  # ResNet-18 高级数据增强配置
│   ├── wideresnet28_10_default.yaml    # WideResNet-28-10 默认配置
│   └── wideresnet_advanced_augment.yaml # WideResNet 高级数据增强配置
├── data/                    # 数据处理模块
│   ├── __init__.py
│   ├── cifar10.py          # CIFAR-10 数据集加载
│   ├── config.py           # 数据配置
│   ├── splits.py           # 数据集划分
│   ├── utils.py            # 工具函数
│   └── augmentation.py     # 高级数据增强策略
├── models/                  # 模型定义模块
│   ├── __init__.py
│   ├── cnn.py              # 基础 CNN 模型
│   ├── resnet.py           # ResNet-18 模型
│   └── wideresnet.py       # WideResNet-28-10 模型
├── checkpoints/            # 训练检查点保存目录（自动创建）
├── logs/                   # 训练日志保存目录（自动创建）
├── eval_results/           # 评估结果保存目录（自动创建）
├── README.md               # 项目说明文档
├── QUICKSTART.md           # 快速开始指南
├── CROSS_VALIDATION_GUIDE.md  # 交叉验证使用指南
└── run_training.ps1        # Windows 训练脚本
```

## 环境要求

### Python 版本
- Python 3.8 或更高版本

### 依赖库
```bash
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.20.0
pyyaml>=5.4.0  # 用于读取 YAML 配置文件
```

### 安装依赖

```bash
pip install torch torchvision numpy pyyaml
```

或者如果你有 CUDA 支持的 GPU：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 支持的模型

项目支持以下三种模型架构：

1. **CNN** (`cnn`): 自定义的卷积神经网络
   - 支持多种归一化方式（Batch Norm, Layer Norm, Group Norm）
   - 可配置的 Dropout 率
   - 适合快速实验和调参

2. **ResNet-18** (`resnet18`): 经典的残差网络
   - 针对 CIFAR-10 (32×32) 优化
   - 使用较小的初始卷积核
   - 平衡性能与计算效率

3. **WideResNet-28-10** (`wideresnet28_10`): 更宽的残差网络
   - 深度 28 层，宽度因子 10
   - 在 CIFAR-10 上表现优异
   - 适合追求更高准确率

## 使用方法

### 三种训练方式

#### 方式一：标准训练（train.py + 配置文件）

使用预配置的 YAML 文件进行标准训练：

**CNN 模型：**
```bash
python train.py --config configs/cnn_default.yaml
```

**ResNet-18：**
```bash
python train.py --config configs/resnet18_default.yaml
```

**WideResNet-28-10：**
```bash
python train.py --config configs/wideresnet28_10_default.yaml
```

#### 方式二：高级数据增强训练（train_finetune.py）

使用增强的数据增强策略（AutoAugment、Cutout、RandomErasing 等）：

**CNN + 高级增强：**
```bash
python train_finetune.py --config configs/cnn_advanced_augment.yaml
```

**ResNet-18 + AutoAugment：**
```bash
python train_finetune.py --config configs/resnet18_advanced_augment.yaml
```

**WideResNet + AutoAugment + Cutout：**
```bash
python train_finetune.py --config configs/wideresnet_advanced_augment.yaml
```

支持的数据增强策略：
- **AutoAugment**: 自动数据增强策略（推荐用于 ResNet/WideResNet）
- **Cutout**: 随机遮挡图像区域
- **RandomErasing**: 随机擦除
- **ColorJitter**: 颜色扰动
- **RandomRotation**: 随机旋转
- **GaussianBlur**: 高斯模糊

#### 方式三：交叉验证与超参数搜索

自动搜索最佳超参数组合：

**快速测试（16 组合，30-60 分钟）：**
```bash
python train.py --config configs/cnn_cv_quick.yaml
```

**完整搜索（48 组合，2-4 小时）：**
```bash
python train.py --config configs/cnn_default.yaml
```

详细说明请参考 [交叉验证指南](CROSS_VALIDATION_GUIDE.md)。

### 方式一：使用配置文件（推荐）

项目提供了预配置的 YAML 文件，包含每个模型的推荐超参数：

**使用 CNN 默认配置：**
```bash
python train.py --config configs/cnn_default.yaml
```

**使用 ResNet-18 默认配置：**
```bash
python train.py --config configs/resnet18_default.yaml
```

**使用 WideResNet-28-10 默认配置：**
```bash
python train.py --config configs/wideresnet28_10_default.yaml
```

**快速测试（10 epochs）：**
```bash
python train.py --config configs/resnet18_quick_test.yaml
```

**超参数搜索：**
```bash
python train.py --config configs/cnn_cross_validation.yaml
```

配置文件优势：
- 可复现：配置文件记录了所有超参数
- 易管理：可以为不同实验创建不同配置文件
- 易分享：配置文件可以直接分享给他人
- 支持覆盖：命令行参数可以覆盖配置文件中的值

### 方式二：命令行参数

#### 基础训练

使用默认 CNN 模型训练 30 个 epoch：

```bash
python train.py
```

### 选择不同的模型

训练 ResNet-18 模型：

```bash
python train.py --model resnet18 --epochs 100 --lr 0.1 --scheduler cosine
```

训练 WideResNet-28-10 模型：

```bash
python train.py --model wideresnet28_10 --epochs 200 --lr 0.1 --scheduler cosine --batch-size 128
```

### 常用训练参数

```bash
python train.py \
    --model resnet18 \                    # 模型类型: cnn, resnet18, wideresnet28_10
    --epochs 100 \                        # 训练轮数
    --batch-size 128 \                    # 批大小
    --lr 0.1 \                           # 学习率
    --weight-decay 5e-4 \                # 权重衰减
    --scheduler cosine \                  # 学习率调度: none, cosine, step
    --val-ratio 0.1 \                    # 验证集比例
    --seed 42 \                          # 随机种子
    --device cuda                         # 设备: cuda 或 cpu
```

### CNN 模型特定参数

```bash
python train.py \
    --model cnn \
    --dropout 0.3 \                      # 卷积层 Dropout
    --linear-dropout 0.5 \               # 全连接层 Dropout
    --normalization batch \              # 归一化类型: batch, layer, group, none
    --group-norm-groups 8                # Group Norm 分组数
```

### 数据增强

默认启用数据增强（随机水平翻转、随机裁剪），可以使用 `--no-augment` 关闭：

```bash
python train.py --no-augment
```

### 混合精度训练

启用混合精度训练以加速训练并减少显存占用（需要 GPU 支持）：

```bash
python train.py --amp --model resnet18
```

## 模型评估

使用 `evaluate.py` 脚本评估训练好的模型：

### 基础评估

```bash
# 评估最佳检查点
python evaluate.py --weights checkpoints/cnn/checkpoint_best.pth --model cnn
python evaluate.py --weights checkpoints/cnn_advanced/checkpoint_epoch_0200.pth --model cnn

# 评估 ResNet-18
python evaluate.py --weights checkpoints/resnet18/checkpoint_best.pth --model resnet18
python evaluate.py --weights checkpoints/resnet18_advanced/checkpoint_epoch_0350.pth --model resnet18

# 评估 WideResNet
python evaluate.py --weights checkpoints/wideresnet28_10/checkpoint_best.pth --model wideresnet28_10
python evaluate.py --weights checkpoints/wideresnet28_10_advanced/checkpoint_epoch_0300.pth --model wideresnet28_10



```

### 详细评估（保存预测结果）

```bash
python evaluate.py \
    --weights checkpoints/resnet18/checkpoint_best.pth \
    --model resnet18 \
    --split test \
    --save-predictions \
    --output-dir eval_results
```

### 评估输出

评估脚本会自动保存三个文件到 `eval_results/` 目录：

1. **JSON 结果文件** (`eval_resnet18_test_20251019_143000.json`)
   ```json
   {
     "timestamp": "20251019_143000",
     "model": "resnet18",
     "evaluation": {
       "loss": 0.2845,
       "accuracy": 0.9123,
       "eval_time_seconds": 12.34
     },
     "per_class_accuracy": {
       "airplane": 0.925,
       "automobile": 0.938,
       ...
     }
   }
   ```

2. **文本报告** (`report_resnet18_test_20251019_143000.txt`)
   - 整体性能指标
   - 每类准确率
   - 详细分类报告（如果安装了 scikit-learn）

3. **预测结果** (`predictions_resnet18_test_20251019_143000.json`)（使用 `--save-predictions`）
   - 所有样本的真实标签和预测标签

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weights` | 权重文件路径 | 必需 |
| `--model` | 模型类型 | resnet18 |
| `--split` | 评估数据集 | test |
| `--output-dir` | 结果保存目录 | eval_results |
| `--save-predictions` | 保存详细预测 | False |
| `--no-save` | 不保存结果（仅打印） | False |

## 日志与检查点

### 日志记录

训练过程会自动记录到日志文件中，默认保存在 `logs/` 目录：

- 控制台输出：实时显示训练进度
- 日志文件：`logs/train_YYYYMMDD_HHMMSS.log`
- 训练历史：`logs/train_history.json`（包含每个 epoch 的详细指标）

指定日志目录：

```bash
python train.py --log-dir ./my_logs
```

### 检查点保存

训练过程会定期保存检查点，默认每 5 个 epoch 保存一次：

```bash
python train.py --checkpoint-dir checkpoints --save-freq 5
```

保存的检查点包括：
- `checkpoint_latest.pth`: 最新的检查点
- `checkpoint_best.pth`: 验证集上表现最好的检查点

检查点内容：
- 模型权重 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 学习率调度器状态 (`scheduler_state_dict`)
- 当前 epoch (`epoch`)
- 最佳验证准确率 (`best_val_acc`)
- 模型名称 (`model_name`)

### 断点重训（Resume Training）

从检查点恢复训练：

```bash
python train.py --resume checkpoints/checkpoint_latest.pth
```

从最佳检查点恢复：

```bash
python train.py --resume checkpoints/checkpoint_best.pth
```

完整的恢复训练示例：

```bash
python train.py \
    --model resnet18 \
    --resume checkpoints/checkpoint_latest.pth \
    --epochs 200 \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

**注意**：恢复训练时会自动从检查点加载：
- 模型权重
- 优化器状态（包括动量等）
- 学习率调度器状态
- 当前训练进度

训练将从保存的 epoch 的下一个 epoch 继续。

### 保存最终模型

训练结束后保存模型到指定路径：

```bash
python train.py --save-path final_model.pth
```

## 交叉验证与超参数搜索

使用交叉验证进行超参数搜索：

```bash
python train.py \
    --cross-validate \
    --kfolds 3 \
    --cv-epochs 10 \
    --lr-grid 1e-3 5e-4 1e-4 \
    --dropout-grid 0.2 0.3 0.4 \
    --linear-dropout-grid 0.4 0.5 \
    --norm-grid batch group \
    --weight-decay-grid 1e-4 5e-4
```

**注意**：交叉验证模式目前仅支持 CNN 模型。

## 配置文件详解

### 可用的配置文件

项目 `configs/` 目录下提供了以下配置文件：

| 配置文件 | 说明 | 推荐场景 |
|---------|------|---------|
| `cnn_default.yaml` | CNN 模型默认配置 | 快速实验、超参数调优 |
| `cnn_cv_quick.yaml` | CNN 快速交叉验证 | 快速测试搜索流程 |
| `cnn_cross_validation.yaml` | CNN 完整交叉验证 | 深度超参数搜索 |
| `cnn_advanced_augment.yaml` | CNN 高级数据增强 | 提升模型性能 |
| `resnet18_default.yaml` | ResNet-18 标准训练配置 | 追求性能与速度平衡 |
| `resnet18_quick_test.yaml` | ResNet-18 快速测试 | 代码验证、快速迭代 |
| `resnet18_advanced_augment.yaml` | ResNet-18 + AutoAugment | 高性能训练 |
| `wideresnet28_10_default.yaml` | WideResNet-28-10 完整训练 | 追求最高准确率 |
| `wideresnet_advanced_augment.yaml` | WideResNet + 最强增强 | 极致性能优化 |

### 配置文件格式

配置文件使用 YAML 格式（也支持 JSON），包含所有训练参数：

```yaml
# 模型配置
model: resnet18

# 训练参数
epochs: 100
batch_size: 128
lr: 0.1
weight_decay: 0.0005

# 学习率调度
scheduler: cosine

# 保存与恢复
checkpoint_dir: checkpoints/resnet18
log_dir: logs/resnet18
save_freq: 10

# ... 更多参数
```

### 自定义配置文件

你可以复制现有配置文件并修改：

```bash
# 复制默认配置
cp configs/resnet18_default.yaml configs/my_resnet18.yaml

# 编辑配置文件
# 修改你需要的参数，如 epochs, lr 等

# 使用自定义配置训练
python train.py --config configs/my_resnet18.yaml
```

### 配置文件 + 命令行参数

命令行参数可以覆盖配置文件中的值：

```bash
# 使用配置文件，但覆盖 epochs 和 batch_size
python train.py --config configs/resnet18_default.yaml --epochs 50 --batch-size 256

# 使用配置文件，但从检查点恢复
python train.py --config configs/resnet18_default.yaml --resume checkpoints/checkpoint_latest.pth
```

## 训练示例

### 示例 1：使用配置文件快速开始（推荐）

```bash
# CNN 标准训练
python train.py --config configs/cnn_default.yaml

# CNN 高级数据增强训练
python train_finetune.py --config configs/cnn_advanced_augment.yaml

# ResNet-18 完整训练
python train.py --config configs/resnet18_default.yaml

# ResNet-18 + AutoAugment
python train_finetune.py --config configs/resnet18_advanced_augment.yaml

# WideResNet 高性能训练
python train.py --config configs/wideresnet28_10_default.yaml

# WideResNet + 最强数据增强
python train_finetune.py --config configs/wideresnet_advanced_augment.yaml

# 快速交叉验证测试
python train.py --config configs/cnn_cv_quick.yaml
```

### 示例 2：快速测试（10 epochs）

```bash
python train.py --config configs/cnn_default.yaml
```

或使用命令行：

```bash
python train.py --epochs 10 --batch-size 256
```

### 示例 3：ResNet-18 标准训练

**使用配置文件（推荐）：**
```bash
python train.py --config configs/resnet18_default.yaml
```

**使用命令行参数：**
```bash
python train.py \
    --model resnet18 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --scheduler cosine \
    --checkpoint-dir checkpoints/resnet18 \
    --log-dir logs/resnet18 \
    --save-path models/resnet18_final.pth
```

### 示例 4：WideResNet-28-10 高性能训练

**使用配置文件（推荐）：**
```bash
python train.py --config configs/wideresnet28_10_default.yaml
```

**使用命令行参数：**
```bash
python train.py \
    --model wideresnet28_10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --scheduler cosine \
    --amp \
    --checkpoint-dir checkpoints/wideresnet \
    --log-dir logs/wideresnet \
    --save-path models/wideresnet_final.pth
```

### 示例 5：从断点恢复训练

假设训练在第 50 个 epoch 意外中断：

**使用配置文件（推荐）：**
```bash
python train.py \
    --config configs/resnet18_default.yaml \
    --resume checkpoints/resnet18/checkpoint_latest.pth
```

**使用命令行参数：**
```bash
python train.py \
    --model resnet18 \
    --resume checkpoints/checkpoint_latest.pth \
    --epochs 100 \
    --lr 0.1 \
    --scheduler cosine \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

训练将从第 51 个 epoch 继续，保留之前的优化器状态和学习率。

### 示例 6：超参数搜索（交叉验证）

**快速测试（30-60 分钟）：**
```bash
python train.py --config configs/cnn_cv_quick.yaml
```

**完整搜索（2-4 小时）：**
```bash
# 使用预定义的交叉验证配置
python train.py --config configs/cnn_default.yaml

# 或自定义搜索空间
python train.py \
    --cross-validate \
    --kfolds 5 \
    --cv-epochs 10 \
    --lr-grid 0.001 0.0005 0.0001 \
    --dropout-grid 0.2 0.3 0.4
```

详细说明：[交叉验证指南](CROSS_VALIDATION_GUIDE.md)

### 示例 7：模型评估

```bash
# 评估训练好的模型
python evaluate.py --weights checkpoints/resnet18/checkpoint_best.pth --model resnet18

# 保存详细评估结果
python evaluate.py \
    --weights checkpoints/cnn/checkpoint_best.pth \
    --model cnn \
    --split test \
    --save-predictions \
    --output-dir eval_results
```

## 常见问题

### Q1: 如何查看训练历史？

训练历史保存在 `logs/train_history.json` 文件中，可以使用 Python 读取和可视化：

```python
import json
import matplotlib.pyplot as plt

with open('logs/train_history.json', 'r') as f:
    history = json.load(f)

epochs = [h['epoch'] for h in history]
train_acc = [h['train_acc'] for h in history]
val_acc = [h['val_acc'] for h in history]

plt.plot(epochs, train_acc, label='Train')
plt.plot(epochs, val_acc, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Q2: 如何选择合适的模型？

- **CNN**: 快速实验、资源有限、需要调参灵活性
- **ResNet-18**: 平衡性能与速度，适合大多数场景
- **WideResNet-28-10**: 追求最高准确率，但需要更多计算资源

### Q3: 训练时显存不足怎么办？

尝试以下方法：
1. 减小 batch size: `--batch-size 64` 或 `--batch-size 32`
2. 启用混合精度训练: `--amp`
3. 减少数据加载线程: `--num-workers 2`
4. 选择更小的模型: `--model resnet18`

### Q4: 如何获得更好的性能？

1. 增加训练轮数（ResNet-18: 100-200, WideResNet: 200-300）
2. 使用余弦退火学习率调度: `--scheduler cosine`
3. 适当增大初始学习率: `--lr 0.1`
4. 调整权重衰减: `--weight-decay 5e-4`
5. 确保使用数据增强（默认开启）

### Q5: 如何在 CPU 上训练？

```bash
python train.py --device cpu --batch-size 32 --num-workers 0
```

## 性能参考

在 CIFAR-10 数据集上的典型性能（测试集准确率）：

### 标准训练

| 模型 | Epochs | 数据增强 | 准确率 | 训练时间（GPU）|
|------|--------|---------|--------|---------------|
| CNN  | 100    | 基础 | ~75-80% | ~15 分钟     |
| ResNet-18 | 100 | 基础 | ~93-95% | ~30 分钟    |
| WideResNet-28-10 | 200 | 基础 | ~95-96% | ~2 小时 |

### 高级数据增强训练（train_finetune.py）

| 模型 | Epochs | 数据增强策略 | 准确率 | 训练时间（GPU）|
|------|--------|------------|--------|---------------|
| CNN  | 200    | Advanced | ~82-85% | ~30 分钟     |
| ResNet-18 | 150 | AutoAugment + Cutout | ~94-96% | ~45 分钟    |
| WideResNet-28-10 | 350 | AutoAugment + Cutout | ~96-97% | ~4 小时 |

*测试环境：NVIDIA RTX 3090, Batch Size 128*

### 数据增强策略对比

| 策略 | ResNet-18 准确率 | 提升 |
|------|-----------------|------|
| 无增强 | ~88% | - |
| 基础（Flip + Crop） | ~93% | +5% |
| + Cutout | ~94% | +1% |
| + AutoAugment | ~95% | +1% |
| + AutoAugment + Cutout | ~96% | +1% |

## 脚本说明

### train.py - 标准训练脚本

适用场景：
- 标准模型训练
- 基础数据增强（RandomFlip + RandomCrop）
- 交叉验证与超参数搜索
- 快速实验

### train_finetune.py - 高级训练脚本

适用场景：
- 高级数据增强（AutoAugment、Cutout 等）
- 性能优化和微调
- 追求更高准确率
- 研究数据增强影响

支持的增强策略：
- AutoAugment（CIFAR10 策略）
- Cutout（随机遮挡）
- RandomErasing（随机擦除）
- ColorJitter（颜色扰动）
- RandomRotation（随机旋转）
- GaussianBlur（高斯模糊）

### evaluate.py - 模型评估脚本

功能：
- 测试集评估
- 每类准确率统计
- 详细分类报告（需要 scikit-learn）
- 自动保存评估结果（JSON + TXT）
- 支持保存预测结果

输出文件：
- `eval_results/eval_{model}_{split}_{timestamp}.json` - JSON 格式结果
- `eval_results/report_{model}_{split}_{timestamp}.txt` - 可读文本报告
- `eval_results/predictions_{model}_{split}_{timestamp}.json` - 预测结果（可选）

## 高级功能

### 1. 混合精度训练（AMP）

加速训练并减少显存占用：

```bash
python train.py --config configs/resnet18_default.yaml --amp
```

支持 PyTorch 1.x 和 2.x，自动检测版本并使用正确的 API。

### 2. 学习率调度

支持三种调度策略：

```yaml
scheduler: cosine  # 余弦退火（推荐）
scheduler: step    # 阶梯衰减
scheduler: none    # 固定学习率
```

### 3. 数据增强策略选择

**基础增强**（train.py）：
- RandomHorizontalFlip
- RandomCrop(32, padding=4)

**高级增强**（train_finetune.py）：
```yaml
augment_policy: basic      # 基础增强
augment_policy: advanced   # 高级增强（ColorJitter + Rotation + Blur）
augment_policy: autoaugment  # AutoAugment（最强）
```

### 4. 断点重训

完整保存训练状态：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 训练进度

```bash
python train.py --config configs/resnet18_default.yaml \
    --resume checkpoints/resnet18/checkpoint_latest.pth
```

### 5. 交叉验证自动化

特性：
- 自动网格搜索
- 进度显示 `[5/48]`
- 中间结果自动保存（每 5 组合）
- Top 5 最佳配置
- 详细统计（均值±标准差）

python evaluate.py --weights checkpoints/cnn_advanced/checkpoint_best.pth --model cnn --split test
