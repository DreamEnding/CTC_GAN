# CTC_GAN

**Language / 语言**: [English](README.md) | [中文](#)

基于 CycleGAN 的 CTC（循环肿瘤细胞）合成图像生成与评估工具集。包含训练、生成、FID 评估脚本与简易日志/检查点管理，适合快速实验与模型对比。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#许可证)

- 开源许可证：MIT
- 贡献指南：详见 CONTRIBUTING.md

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细安装指南](#详细安装指南)
- [数据准备](#数据准备)
- [配置说明](#配置说明)
- [使用教程](#使用教程)
- [项目架构](#项目架构)
- [常见问题](#常见问题)
- [故障排查](#故障排查)
- [作者信息](#作者信息)
- [许可证](#许可证)

## 项目简介

CTC_GAN 是一个基于 CycleGAN 的图像生成项目，专门用于生成合成的循环肿瘤细胞（CTC）图像。该项目实现了完整的训练、生成和评估流程，使用 FID（Fréchet Inception Distance）指标来评估生成图像的质量。

**适用场景**：
- 医学图像数据增强
- CTC 图像合成研究
- 深度学习模型训练数据扩充
- 图像生成质量评估

### 作者信息

- 项目负责人：[Chream]
- Email: yaom7917@gmail.com


## 功能特性

- ✨ **模型训练**：使用 `train.py` 在真实 CTC 风格数据上训练 CycleGAN 模型
- 🎨 **图像生成**：使用 `generate.py` 利用已训练的生成器生成合成图像
- 📊 **质量评估**：使用 FID 指标评估模型质量与排序（`evaluate.py`）
- 💾 **检查点管理**：自动保存和加载训练检查点，支持断点续训
- 📝 **日志记录**：完整的训练日志和生成日志记录
- ⚙️ **灵活配置**：通过 YAML 文件轻松调整模型参数和训练设置
- 🚀 **GPU 加速**：支持 CUDA 加速训练和生成过程

## 环境要求

### 硬件要求
- **CPU**：多核处理器（推荐 4 核以上）
- **内存**：至少 8GB RAM（推荐 16GB 以上）
- **GPU**（可选但强烈推荐）：
  - NVIDIA GPU，显存至少 6GB（推荐 8GB 以上）
  - 支持 CUDA 11.0 或更高版本
- **存储**：至少 10GB 可用空间（用于数据、模型和生成的图像）

### 软件要求
- **操作系统**：Windows 10/11、Linux（Ubuntu 18.04+）、macOS 10.15+
- **Python**：3.10 或更高版本（推荐 3.10 或 3.11）
- **依赖库**：
  - PyTorch 2.0+（支持 CUDA 的版本）
  - torchvision
  - Pillow（图像处理）
  - NumPy
  - SciPy（FID 计算）
  - PyYAML（配置文件解析）

## 快速开始

以下是快速上手的步骤（详细步骤请参考[详细安装指南](#详细安装指南)）：

### Windows 用户

```powershell
# 1. 克隆项目
git clone https://github.com/DreamEnding/CTC_GAN.git
cd CTC_GAN

# 2. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 3. 安装依赖
pip install -U pip
pip install torch torchvision pillow numpy scipy pyyaml

# 4. 准备数据（将图像放入 data/ctc_real 目录）
mkdir data\ctc_real
# 将您的 CTC 图像复制到 data\ctc_real 目录

# 5. 训练模型
python train.py

# 6. 生成图像
python generate.py

# 7. 评估模型
python evaluate.py --checkpoint_dir checkpoints\ --top_k 3
```

### Linux/macOS 用户

```bash
# 1. 克隆项目
git clone https://github.com/DreamEnding/CTC_GAN.git
cd CTC_GAN

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -U pip
pip install torch torchvision pillow numpy scipy pyyaml

# 4. 准备数据（将图像放入 data/ctc_real 目录）
mkdir -p data/ctc_real
# 将您的 CTC 图像复制到 data/ctc_real 目录

# 5. 训练模型
python train.py

# 6. 生成图像
python generate.py

# 7. 评估模型
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

## 详细安装指南

### 步骤 1：安装 Python

#### Windows
1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 Python 3.10 或 3.11 安装程序
3. 运行安装程序，**务必勾选 "Add Python to PATH"**
4. 验证安装：
   ```powershell
   python --version
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3 --version
```

#### macOS
```bash
# 使用 Homebrew 安装
brew install python@3.10
python3 --version
```

### 步骤 2：克隆项目

```bash
# 使用 HTTPS
git clone https://github.com/DreamEnding/CTC_GAN.git

# 或使用 SSH
git clone git@github.com:DreamEnding/CTC_GAN.git

cd CTC_GAN
```

### 步骤 3：创建并激活虚拟环境

虚拟环境可以隔离项目依赖，避免与系统 Python 包冲突。

#### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

激活成功后，命令提示符前会显示 `(.venv)`。

#### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活成功后，命令提示符前会显示 `(.venv)`。

### 步骤 4：安装 PyTorch

根据您的系统和是否有 GPU，选择合适的 PyTorch 版本。

#### 有 NVIDIA GPU（推荐）

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取最新的安装命令。

**Windows/Linux with CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Windows/Linux with CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 仅使用 CPU（不推荐，训练会很慢）

```bash
pip install torch torchvision torchaudio
```

#### 验证 PyTorch 安装

```python
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

预期输出示例：
```
PyTorch 版本: 2.1.0+cu118
CUDA 可用: True
```

### 步骤 5：安装其他依赖

**方法一：使用 requirements.txt（推荐）**

项目提供了 `requirements.txt` 文件，可以一键安装所有依赖：

```bash
pip install -r requirements.txt
```

**方法二：手动安装**

```bash
pip install -U pip
pip install pillow numpy scipy pyyaml
```

**注意**：如果使用方法一，PyTorch 会自动安装 CPU 版本。如果需要 GPU 支持，请先按步骤 4 安装 PyTorch，然后再安装其他依赖：

```bash
# 1. 先安装 PyTorch（GPU 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 再安装其他依赖（跳过 torch 和 torchvision）
pip install pillow numpy scipy pyyaml
```

### 步骤 6：验证安装

运行以下命令验证所有依赖是否正确安装：

```python
python -c "import torch, torchvision, PIL, numpy, scipy, yaml; print('所有依赖安装成功！')"
```

## 数据准备

### 数据集要求

- **格式**：PNG、JPG 或 JPEG 格式
- **内容**：真实的 CTC（循环肿瘤细胞）图像
- **数量**：建议至少 100 张图像（更多更好）
- **大小**：图像会自动调整为配置文件中指定的大小（默认 256x256）

### 数据集目录结构

```
CTC_GAN/
├── data/
│   └── ctc_real/          # 放置真实 CTC 图像的目录
│       ├── image_001.png
│       ├── image_002.jpg
│       ├── image_003.png
│       └── ...
├── output/
│   └── synthetic/         # 生成的合成图像输出目录（自动创建）
├── checkpoints/           # 模型检查点目录（自动创建）
└── logs/                  # 日志目录（自动创建）
```

### 创建数据目录

#### Windows
```powershell
mkdir data\ctc_real
```

#### Linux/macOS
```bash
mkdir -p data/ctc_real
```

### 准备数据

1. 将您的 CTC 图像文件复制到 `data/ctc_real` 目录
2. 确保图像文件格式正确（PNG、JPG 或 JPEG）
3. 验证数据：
   ```python
   python -c "import os; imgs = [f for f in os.listdir('data/ctc_real') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]; print(f'找到 {len(imgs)} 张图像')"
   ```

**注意**：如果您没有真实的 CTC 图像数据，您可以使用任何其他类型的图像数据进行测试，但生成的结果可能不符合 CTC 的特征。

## 配置说明

项目的主要配置文件位于 `configs/train_config.yaml`，您可以根据需要修改各项参数。

### 配置文件结构

```yaml
model:
  image_size: 256              # 图像大小（会自动调整）
  in_channels: 3               # 输入通道数（RGB=3）
  out_channels: 3              # 输出通道数（RGB=3）
  gen_features: 64             # 生成器特征数
  disc_features: 64            # 判别器特征数
  residual_blocks: 9           # 残差块数量
  use_attention: True          # 是否使用注意力机制

training:
  batch_size: 4                # 批次大小（根据显存调整）
  learning_rate: 0.0002        # 学习率
  beta1: 0.5                   # Adam 优化器参数
  beta2: 0.999                 # Adam 优化器参数
  lambda_cycle: 10.0           # 循环一致性损失权重
  lambda_identity: 0.0         # 身份损失权重
  lambda_gp: 10.0              # 梯度惩罚权重
  num_epochs: 100              # 训练轮数
  save_every: 10               # 每隔多少轮保存一次模型
  validate_every: 5            # 每隔多少轮验证一次

paths:
  data_dir: "./data/ctc_real"           # 真实数据目录
  output_dir: "./output/synthetic"      # 合成图像输出目录
  checkpoint_dir: "./checkpoints"       # 检查点保存目录
  log_dir: "./logs"                     # 日志目录

device: "cuda"                 # 使用的设备（"cuda" 或 "cpu"）
```

### 常用配置调整

#### 1. 根据显存调整批次大小

| GPU 显存 | 推荐 batch_size |
|---------|----------------|
| 4GB     | 1-2            |
| 6GB     | 2-4            |
| 8GB     | 4-8            |
| 12GB+   | 8-16           |

如果遇到显存不足错误（OOM），请减小 `batch_size`。

#### 2. 切换到 CPU 模式

如果没有 GPU，将 `device` 改为 `"cpu"`：
```yaml
device: "cpu"
```

**警告**：CPU 训练会非常慢，可能需要数天甚至数周。

#### 3. 调整训练轮数

根据数据集大小和质量需求调整 `num_epochs`：
- 小数据集（<100 张）：50-100 轮
- 中等数据集（100-500 张）：100-200 轮
- 大数据集（>500 张）：200-300 轮

#### 4. 修改保存频率

如果想更频繁地保存检查点（以便评估中间结果）：
```yaml
save_every: 5  # 每 5 轮保存一次
```

## 使用教程

### 1. 训练模型

训练是整个流程的第一步，用于从真实 CTC 图像中学习生成模型。

```bash
python train.py
```

**训练过程说明**：
- 程序会自动创建 `checkpoints/` 和 `logs/` 目录
- 每隔指定轮数（默认 10 轮）保存一次模型检查点
- 检查点文件命名格式：`model_epoch_010.pth`、`model_epoch_020.pth` 等
- 训练日志会实时显示损失值和训练进度

**训练输出示例**：
```
🚀 Starting Training...
Epoch [1/100] Batch 0: D=0.5234, G=1.2345, GP=0.0123
Epoch [1/100] Batch 10: D=0.4567, G=1.1234, GP=0.0089
...
✅ Epoch 1 | Avg D Loss: 0.4800 | Avg G Loss: 1.1800
💾 Model saved at checkpoints/model_epoch_010.pth
```

**断点续训**：
如果训练中断，再次运行 `python train.py` 会自动从最新的检查点继续训练。

**监控训练**：
- 观察损失值是否下降
- 判别器损失（D）和生成器损失（G）应该保持相对平衡
- 如果 D 或 G 损失持续为 0 或非常大，可能需要调整学习率

### 2. 生成合成图像

训练完成后，使用生成器创建合成的 CTC 图像。

#### 方法一：使用命令行（推荐）

编辑 `generate.py` 文件，修改以下参数：

```python
if __name__ == "__main__":
    generate_synthetic_images(
        config_path="configs/train_config.yaml",
        checkpoint_path="checkpoints/model_epoch_050.pth",  # 修改为您的检查点路径
        num_images=100  # 修改为想要生成的图像数量
    )
```

然后运行：
```bash
python generate.py
```

#### 方法二：作为 Python 模块调用

```python
from generate import generate_synthetic_images

generate_synthetic_images(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/model_epoch_050.pth",
    num_images=50
)
```

**生成过程说明**：
- 生成的图像保存在 `output/synthetic/` 目录
- 文件命名格式：`synthetic_ctc_001.png`、`synthetic_ctc_002.png` 等
- 程序会自动处理磁盘空间和内存管理
- 支持 GPU 加速（如果可用）

**生成输出示例**：
```
✅ Saved output/synthetic/synthetic_ctc_001.png
✅ Saved output/synthetic/synthetic_ctc_002.png
...
进度 50/100，剩余磁盘≈5000.0MB
```

### 3. 评估模型质量

使用 FID（Fréchet Inception Distance）指标评估生成器的质量。FID 分数越低，表示生成的图像质量越好。

#### 评估所有检查点并排名

```bash
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

**参数说明**：
- `--checkpoint_dir`：检查点目录路径
- `--top_k`：显示前 K 个最佳模型（默认 3）

**评估输出示例**：
```
🔍 Found 10 checkpoint files
📊 Evaluating all models...

[1/10] Evaluating model_epoch_010.pth...
✅ Epoch 10: FID = 45.2341

[2/10] Evaluating model_epoch_020.pth...
✅ Epoch 20: FID = 38.5672

...

🏆 Model Ranking (Best to Worst):
==================================================
1. Epoch  50 | FID:  28.3456 | model_epoch_050.pth
2. Epoch  60 | FID:  29.1234 | model_epoch_060.pth
3. Epoch  40 | FID:  31.5678 | model_epoch_040.pth

🥇 Best Model: model_epoch_050.pth
📊 Best FID Score: 28.3456
📁 Path: checkpoints/model_epoch_050.pth
```

#### 评估单个检查点

```bash
python evaluate.py --single checkpoints/model_epoch_050.pth
```

**输出示例**：
```
🔍 Evaluating single model: checkpoints/model_epoch_050.pth
📊 FID Score: 28.3456
```

### 4. 完整工作流程示例

以下是一个完整的从零开始的工作流程：

```bash
# 1. 准备环境
cd CTC_GAN
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 2. 检查数据
python -c "import os; print(f'数据集图像数量: {len([f for f in os.listdir(\"data/ctc_real\") if f.endswith((\".png\", \".jpg\"))])}')"

# 3. 训练模型（假设训练 100 轮）
python train.py

# 4. 评估所有模型，找出最佳检查点
python evaluate.py --checkpoint_dir checkpoints/ --top_k 5

# 5. 使用最佳模型生成图像
# 编辑 generate.py，设置 checkpoint_path 为最佳模型路径
python generate.py

# 6. 查看生成的图像
ls output/synthetic/  # Windows: dir output\synthetic\
```

## 项目架构

### 目录结构说明

```
CTC_GAN/
├── configs/                    # 配置文件目录
│   └── train_config.yaml      # 主配置文件
├── data/                      # 数据目录
│   └── ctc_real/              # 真实 CTC 图像
├── datasets/                  # 数据集模块
│   ├── __init__.py
│   └── ctc_dataset.py        # 数据集加载器
├── models/                    # 模型定义
│   ├── cyclegan_system.py    # CycleGAN 系统（训练逻辑）
│   ├── discriminator.py      # 判别器网络
│   └── generator.py          # 生成器网络
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── checkpoint.py         # 检查点管理
│   ├── logger.py             # 日志记录
│   └── metrics.py            # 评估指标（FID）
├── output/                    # 输出目录
│   └── synthetic/            # 生成的合成图像
├── checkpoints/              # 模型检查点
├── logs/                     # 训练日志
├── train.py                  # 训练脚本
├── generate.py               # 图像生成脚本
├── evaluate.py               # 模型评估脚本
├── README.md                 # 英文说明文档
├── README_CN.md              # 中文说明文档（本文件）
└── CONTRIBUTING.md           # 贡献指南
```

### 核心模块说明

#### 1. Generator（生成器）
- 文件：`models/generator.py`
- 功能：将真实图像转换为合成图像
- 架构：基于 ResNet 的生成器，包含注意力机制

#### 2. Discriminator（判别器）
- 文件：`models/discriminator.py`
- 功能：区分真实图像和生成图像
- 架构：PatchGAN 判别器

#### 3. CycleGAN System
- 文件：`models/cyclegan_system.py`
- 功能：整合生成器和判别器，实现训练逻辑
- 包含：循环一致性损失、对抗损失、梯度惩罚等

#### 4. Dataset Loader
- 文件：`datasets/ctc_dataset.py`
- 功能：加载和预处理图像数据
- 支持：数据增强（翻转、颜色抖动等）

#### 5. Metrics
- 文件：`utils/metrics.py`
- 功能：计算 FID 分数
- 原理：比较真实图像和生成图像的特征分布

## 常见问题

### Q1: 训练需要多长时间？

**答**：取决于多个因素：
- **数据集大小**：100 张图像约需 2-4 小时（GPU）
- **训练轮数**：100 轮
- **硬件配置**：
  - GPU（RTX 3080）：约 2-3 小时
  - GPU（GTX 1060）：约 6-8 小时
  - CPU：数天（不推荐）

### Q2: 显存不足怎么办？

**答**：尝试以下方法：
1. 减小 `batch_size`（改为 1 或 2）
2. 减小 `image_size`（改为 128）
3. 减少 `residual_blocks`（改为 6）
4. 关闭 `use_attention`（改为 False）

### Q3: 生成的图像质量不好怎么办？

**答**：可能的原因和解决方案：
1. **训练不足**：增加训练轮数
2. **数据质量差**：使用更高质量的训练数据
3. **数据量太少**：收集更多训练图像（至少 100 张）
4. **模型未收敛**：检查损失曲线，确保模型收敛
5. **检查点选择不当**：评估多个检查点，选择 FID 最低的

### Q4: 如何选择最佳的检查点？

**答**：
1. 使用 `evaluate.py` 评估所有检查点
2. 选择 FID 分数最低的模型
3. 也可以手动查看生成的图像质量
4. 通常在训练中期（50-70%）的模型效果较好

### Q5: 可以使用其他类型的图像数据吗？

**答**：可以！虽然项目是为 CTC 图像设计的，但可以用于任何图像生成任务：
1. 将您的图像放入 `data/ctc_real/`
2. 调整配置文件中的参数
3. 按正常流程训练和生成

### Q6: 如何导出模型供其他项目使用？

**答**：检查点文件（`.pth`）包含完整的模型权重，可以这样加载：

```python
import torch
from models.generator import Generator

# 加载配置和模型
checkpoint = torch.load('checkpoints/model_epoch_050.pth')
gen = Generator(img_channels=3, num_features=64, num_residuals=9, use_attention=True)
gen.load_state_dict(checkpoint['gen_state_dict'])
gen.eval()
```

### Q7: 训练中断了，可以继续吗？

**答**：可以！项目支持断点续训：
- 再次运行 `python train.py`
- 程序会自动检测最新的检查点并继续训练

### Q8: 如何调整生成图像的数量？

**答**：编辑 `generate.py` 文件：
```python
generate_synthetic_images(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints/model_epoch_050.pth",
    num_images=200  # 修改这里
)
```

## 故障排查

### 问题 1：`RuntimeError: CUDA out of memory`

**原因**：GPU 显存不足

**解决方案**：
```yaml
# 在 configs/train_config.yaml 中修改
training:
  batch_size: 1  # 减小批次大小
```

### 问题 2：`ValueError: No images found in data/ctc_real`

**原因**：数据目录为空或图像格式不支持

**解决方案**：
1. 检查数据目录：
   ```bash
   ls data/ctc_real/  # Linux/macOS
   dir data\ctc_real\  # Windows
   ```
2. 确保图像格式为 PNG、JPG 或 JPEG
3. 确保至少有一张图像

### 问题 3：`ModuleNotFoundError: No module named 'torch'`

**原因**：PyTorch 未安装或虚拟环境未激活

**解决方案**：
```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows

# 安装 PyTorch
pip install torch torchvision
```

### 问题 4：训练损失不下降或变成 NaN

**原因**：学习率过大或梯度爆炸

**解决方案**：
```yaml
# 在 configs/train_config.yaml 中修改
training:
  learning_rate: 0.0001  # 减小学习率
  lambda_gp: 10.0        # 确保梯度惩罚启用
```

### 问题 5：生成图像全黑或全白

**原因**：模型未收敛或训练不足

**解决方案**：
1. 增加训练轮数
2. 检查数据质量
3. 尝试不同的检查点
4. 验证数据预处理是否正确

### 问题 6：`PermissionError: 输出目录不可写`

**原因**：没有写入权限

**解决方案**：
```bash
# Linux/macOS
chmod -R 755 output/

# Windows：右键目录 → 属性 → 安全 → 编辑权限
```

### 问题 7：FID 评估时间过长

**原因**：样本数量过多

**解决方案**：
- FID 计算默认使用 1000 个样本
- 可以在 `evaluate.py` 中修改 `num_samples` 参数
- 减少样本数会加快评估但可能降低准确性

### 问题 8：如何查看详细日志？

**解决方案**：
```bash
# 查看训练日志
cat logs/config.json  # Linux/macOS
type logs\config.json  # Windows

# 实时监控训练输出
python train.py 2>&1 | tee training.log
```

## 性能优化建议

### 1. 使用 GPU 加速
- 确保安装支持 CUDA 的 PyTorch 版本
- 检查 GPU 是否被正确识别：
  ```python
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### 2. 数据加载优化
- 增加 `num_workers` 以加快数据加载：
  ```python
  # 在 datasets/ctc_dataset.py 中
  num_workers=4  # 根据 CPU 核心数调整
  ```

### 3. 混合精度训练（高级）
- 使用 PyTorch 的 AMP（Automatic Mixed Precision）
- 可以减少显存使用并加快训练

### 4. 定期清理缓存
- 在 `generate.py` 中已实现自动缓存清理
- 训练时可以定期调用 `torch.cuda.empty_cache()`

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

详细信息请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目采用 MIT 许可证。您可以自由地使用、复制、修改与分发本项目的代码，但需保留原始许可声明。

---

## 获取帮助

如果您遇到问题或有疑问：

1. 📖 首先查阅本文档的[常见问题](#常见问题)和[故障排查](#故障排查)部分
2. 🔍 搜索 [Issues](https://github.com/DreamEnding/CTC_GAN/issues) 查看是否有类似问题
3. 💬 创建新的 [Issue](https://github.com/DreamEnding/CTC_GAN/issues/new) 描述您的问题
4. 📧 联系作者：yaom7917@gmail.com

---

**祝您使用愉快！Happy Coding! 🎉**