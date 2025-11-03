# CTC_GAN

**Language / 语言**: [English](README.md) | [中文](#)

基于 WGAN-GP（Wasserstein GAN with Gradient Penalty）+ **AdaIN 噪声注入**的 CTC 合成图像生成工具。采用混合架构：真实图像提供内容/结构，随机噪声增加多样性，可从一张图像生成无限不重复的高质量变体。

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

# 通过真实图像 + 噪声生成多样化变体
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 10

# 评估模型质量
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

---

### 项目功能概述
- **训练**：使用 `train.py` 训练基于 AdaIN 噪声注入的 WGAN-GP 模型
- **生成**：使用 `generate.py` 通过真实图像+不同噪声生成无限多样化变体
- **评估**：使用 FID 指标评估模型质量与排序（`evaluate.py`、`utils/metrics.py`）
- **管理**：提供日志记录与检查点管理（`utils/logger.py`、`utils/checkpoint.py`）

### 核心特性
✅ **混合架构**：真实图像提供内容，随机噪声增加多样性  
✅ **无限变体**：一张图像 + 不同噪声 = 无限不重复变体  
✅ **AdaIN 风格调制**：在保持结构的同时添加细节变化  
✅ **感知损失**：保持语义相似性而非像素级重建  
✅ **WGAN-GP 训练**：Wasserstein 损失与梯度惩罚保证稳定训练  
✅ **可复现性**：设置随机种子可重现特定变体

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
  - `model.use_noise_injection`：启用 AdaIN 噪声注入（默认：True）
  - `model.gen_features`、`model.disc_features`：基础特征维度
  - `model.residual_blocks`：ResNet 模块数量（默认：9）
  - `model.use_attention`：是否启用自注意力层（默认：True）
  - `training.lambda_perceptual`：感知损失权重（默认：1.0）
  - `training.lambda_cycle`：循环一致性损失权重（设为 0.0 以获得纯多样性）
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
  use_noise_injection: True

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
模型从 `data/ctc_real/` 中的真实图像学习，通过注入噪声生成多样化变体。

**生成图像：**
```bash
# 从 10 张源图像生成 100 个变体（每张 10 个）
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 10

# 从 10 张源图像生成 1000 个变体（每张 100 个）
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 100

# 使用特定种子以便复现
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 5 --num_variants_per_image 20 --seed 42
```

**评估模型：**
```bash
# 评估所有检查点并排序前 3 名
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3

# 评估单个检查点
python evaluate.py --single checkpoints/model_epoch_050.pth
```

### 模型架构

**混合生成器（带 AdaIN）：**
```
输入 1：真实图像 x, 形状 [batch_size, 3, 64, 64]
输入 2：随机噪声 z ~ N(0, 1), 形状 [batch_size, 512]
  ↓
[图像编码器]
  真实图像 → 下采样到 16x16 (256 通道) → 内容特征
  ↓
[噪声编码器]
  噪声向量 → MLP → 样式参数（均值、标准差）
  ↓
[AdaIN 调制]
  9 个残差模块配合 AdaIN（融合内容+样式）
  ↓
[自注意力]
  捕获全局依赖关系（可选）
  ↓
[解码器]
  上采样到 32x32 (128 通道)
  ↓
  上采样到 64x64 (64 通道)
  ↓
输出：3x64x64 RGB 变体图像，范围 [-1, 1]
```

**核心创新 - AdaIN（自适应实例归一化）：**
- 归一化来自图像的内容特征
- 使用从噪声派生的样式参数进行调制
- 公式：`AdaIN(content, style) = σ_style * normalize(content) + μ_style`
- 结果：保持内容结构，细节由噪声变化

**判别器（Discriminator）：**
- WGAN-GP 批评器，带梯度惩罚
- 多尺度卷积架构
- 区分真实图像与生成的变体

### 开发依赖
- `torch`、`torchvision`（模型与数据操作，VGG19 用于感知损失）
- `numpy`、`scipy`（FID 计算）
- `Pillow`（图像 IO）
- `pyyaml`（配置文件）
- 可选：CUDA 支持（加速训练与评估）

### 模型特性
- **混合架构**：真实图像提供内容，噪声提供多样性
- **基于 AdaIN 的样式注入**：在保持结构的同时调制特征
- **感知损失**：保持语义相似性（VGG 特征）而非像素级重建
- **WGAN-GP**：Wasserstein 损失与梯度惩罚保证稳定训练
- **无限可扩展**：一张图像 + 不同噪声 = 无限独特变体

### 训练策略
1. **判别器**：区分真实图像与生成的变体
2. **生成器**：
   - 接收真实图像 + 随机噪声作为输入
   - 对抗损失：欺骗判别器
   - 感知损失：保持与源图像的语义相似性
   - 无像素级重建损失（允许创造性变化）

### 测试方法
- 确认 `data/ctc_real` 目录包含训练图像
- 运行 `python train.py` 训练几个 epoch
- 使用 `python generate.py --checkpoint <路径> --num_images 5 --num_variants_per_image 10` 生成变体
- 确认生成的图像出现在 `output/synthetic/`
- 检查同一源的变体在细节上有所不同但保持结构
- 确保日志写入 `logs/`，检查点保存到 `checkpoints/`

### 预期效果
- ✅ 生成的变体保留源图像的主要特征（颜色、结构、风格）
- ✅ 每个变体具有独特的细节（纹理、精细结构、局部变化）
- ✅ 高质量输出，无噪声伪影
- ✅ 训练稳定且可靠收敛
- ✅ 可为每个源图像生成无限变体

### 注意事项
- 模型学习**生成多样化变体**同时保持内容
- 不同噪声向量从同一源图像产生不同输出
- 训练和生成都需要真实图像（与纯噪声 GAN 不同）
- 使用 `--seed` 参数实现可复现的生成
- MIT 许可证

### 贡献指南
- 欢迎提交 Issue 与 Pull Request
- 详细流程与规范请参考 `CONTRIBUTING.md`

### 许可证
- 本项目采用 MIT 许可证
- 你可以自由使用、复制、修改与分发本项目的代码，但需保留原始许可声明