# CTC_GAN

**Language / 语言**: [English](README.md) | [中文](#)

基于 WGAN-GP（Wasserstein GAN with Gradient Penalty）的 CTC 合成图像生成工具。模型从随机噪声生成图像，学习训练数据的分布特征，可生成无限数量的独特高质量图像。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#许可证)

- 开源许可证：MIT
- 贡献指南：详见 CONTRIBUTING.md

### 作者

- 项目负责人：[Chream]
- Email:yaom7917@gmail.com


### 快速开始

```bash
cd CTC_GAN

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -U pip
pip install torch torchvision pillow numpy scipy pyyaml

# 训练模型
python train.py

# 从随机噪声生成图像
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 100

# 评估模型质量
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

---

### 项目功能概述
- **训练**：使用 `train.py` 在真实 CTC 数据上训练 WGAN-GP 模型，学习数据分布
- **生成**：使用 `generate.py` 从随机噪声生成合成图像（无需真实图像输入）
- **评估**：使用 FID 指标评估模型质量与排序（`evaluate.py`、`utils/metrics.py`）
- **管理**：提供日志记录与检查点管理（`utils/logger.py`、`utils/checkpoint.py`）

### 核心特性
✅ **无条件生成**：从随机噪声生成图像（无需输入图像）  
✅ **无限变化**：通过不同随机种子生成无限数量的独特图像  
✅ **高质量输出**：使用 ResNet 模块和自注意力机制  
✅ **WGAN-GP 训练**：Wasserstein 损失与梯度惩罚保证稳定训练  
✅ **可复现性**：设置随机种子可重现特定图像

### 安装说明
- 环境要求：建议 Python 3.10+，Windows 下使用虚拟环境。
- 创建并激活虚拟环境：

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\activate
```

- 安装依赖：

```powershell
pip install -U pip
```

```powershell
pip install torch torchvision pillow numpy scipy pyyaml
```

### 配置指南
- 主配置文件：`configs\train_config.yaml`
- 关键字段说明：
  - `model.latent_dim`：随机噪声向量维度（默认：512）
  - `model.image_size`：输出图像尺寸（默认：64x64）
  - `model.gen_features`、`model.disc_features`：基础特征维度
  - `model.residual_blocks`：ResNet 模块数量（默认：9）
  - `model.use_attention`：是否启用自注意力层（默认：True）
  - `training.batch_size`、`training.learning_rate`、`training.num_epochs`
  - `paths.data_dir`（训练数据目录）、`paths.output_dir`（生成图像输出目录）
  - `device`：`"cuda"` 或 `"cpu"`

示例配置：
```yaml
model:
  image_size: 64
  in_channels: 3
  latent_dim: 512
  gen_features: 64
  disc_features: 64
  residual_blocks: 9
  use_attention: True

training:
  batch_size: 4
  learning_rate: 0.0002
  num_epochs: 100
  save_every: 10
  lambda_gp: 10.0

paths:
  data_dir: "./data/ctc_real"
  output_dir: "./output/synthetic"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"

device: "cuda"
```

### 使用示例

**训练模型：**
```bash
python train.py
```
模型从 `data/ctc_real/` 中的真实图像学习，但生成时使用随机噪声。

**生成图像：**
```bash
# 生成 100 张图像
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 100

# 使用特定种子以便复现
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 50 --seed 42

# 使用更大批次加速生成
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 1000 --batch_size 16
```

**评估模型：**
```bash
# 评估所有检查点并排序前 3 名
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3

# 评估单个检查点
python evaluate.py --single checkpoints/model_epoch_050.pth
```

### 模型架构

**生成器（Generator）：**
```
输入：随机噪声 z ~ N(0, 1), 形状 [batch_size, 512]
  ↓
全连接层 + Reshape 到 [batch_size, 256, 4, 4]
  ↓
上采样到 8x8 (256 通道)
  ↓
上采样到 16x16 (256 通道)
  ↓
9 个残差模块 (256 通道)
  ↓
自注意力层（可选）
  ↓
上采样到 32x32 (128 通道)
  ↓
上采样到 64x64 (64 通道)
  ↓
输出：3x64x64 RGB 图像，范围 [-1, 1]
```

**判别器（Discriminator）：**
- WGAN-GP 批评器，带梯度惩罚
- 多尺度卷积架构
- 区分真实图像与生成图像

### 开发依赖
- `torch`、`torchvision`（模型与数据操作）
- `numpy`、`scipy`（FID 计算）
- `Pillow`（图像 IO）
- `pyyaml`（配置文件）
- 可选：CUDA 支持（加速训练与评估）

### 模型特性
- **无条件 GAN**：从随机噪声生成图像，不依赖输入图像
- **无重建损失**：纯对抗训练（无 L1/L2 损失）
- **WGAN-GP**：Wasserstein 损失与梯度惩罚保证稳定训练
- **可扩展**：可生成无限数量的独特图像

### 测试方法
- 确认 `data/ctc_real` 目录包含训练图像
- 运行 `python train.py` 训练几个 epoch
- 使用 `python generate.py --checkpoint <路径> --num_images 10` 生成图像
- 确认生成的图像出现在 `output/synthetic/`
- 确保日志写入 `logs/`，检查点保存到 `checkpoints/`

### 注意事项
- 模型学习训练数据的**分布**，而非记忆特定图像
- 不同随机种子产生不同输出
- 训练需要真实图像，但生成不需要
- MIT 许可证

### 贡献指南
- 欢迎提交 Issue 与 Pull Request
- 详细流程与规范请参考 `CONTRIBUTING.md`

### 许可证
- 本项目采用 MIT 许可证
- 你可以自由使用、复制、修改与分发本项目的代码，但需保留原始许可声明