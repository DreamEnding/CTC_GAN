# CTC_GAN

**Language / 语言**: [English](README.md) | [中文](#)

基于 CycleGAN 的 CTC 合成图像生成与评估工具集。包含训练、生成、FID 评估脚本与简易日志/检查点管理，适合快速实验与模型对比。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#许可证)

- 开源许可证：MIT
- 贡献指南：详见 CONTRIBUTING.md

### 作者

- 项目负责人：[Chream]
- Email:yaom7917@gmail.com


### 基本使用说明（代码示例）

```python
# Python 代码示例：直接调用生成函数
from generate import generate_synthetic_images

generate_synthetic_images(
    config_path="configs/train_config.yaml",
    checkpoint_path="checkpoints\\model_epoch_050.pth",
    num_images=50
)
```

```powershell
# 训练模型
python train.py
```

```powershell
# 生成合成图像（使用 generate.py 默认参数）
python generate.py
```

```powershell
# 评估所有检查点（Top-K）
python evaluate.py --checkpoint_dir checkpoints\ --top_k 3
```

```powershell
# 评估单个检查点
python evaluate.py --single checkpoints\model_epoch_050.pth
```

---

### 项目功能概述
- 训练：使用 `train.py` 在真实 CTC 风格数据上训练 CycleGAN。
- 生成：使用 `generate.py` 利用已训练的 `Generator` 生成合成图像。
- 评估：使用 FID 指标评估模型与排序（`evaluate.py`、`utils/metrics.py`）。
- 管理：提供日志记录与检查点管理（`utils/logger.py`、`utils/checkpoint.py`）。

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
  - `model.image_size`、`model.in_channels`、`model.out_channels`、`model.gen_features`、`model.disc_features`、`model.residual_blocks`、`model.use_attention`
  - `training.batch_size`、`training.learning_rate`、`training.num_epochs`、`training.save_every`、`training.validate_every`、`training.lambda_*`
  - `paths.data_dir`（真实数据目录）、`paths.output_dir`（合成图像输出目录）、`paths.checkpoint_dir`、`paths.log_dir`
  - `device`：`"cuda"` 或 `"cpu"`

示例（节选）：
```yaml
model:
  image_size: 256
  in_channels: 3
  out_channels: 3
  gen_features: 64
  disc_features: 64
  residual_blocks: 9
  use_attention: True

training:
  batch_size: 4
  learning_rate: 0.0002
  num_epochs: 100
  save_every: 10

paths:
  data_dir: "./data/ctc_real"
  output_dir: "./output/synthetic"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"

device: "cuda"
```

### 使用示例
- 训练并自动保存检查点到 `checkpoints\`：

```powershell
python train.py
```

- 生成合成图像（默认输出到 `output\synthetic\`，数量可在 `generate.py` 中修改）：

```powershell
python generate.py
```

- 评估所有检查点的 FID 并输出前 3 名：

```powershell
python evaluate.py --checkpoint_dir checkpoints\ --top_k 3
```

- 评估单个检查点：

```powershell
python evaluate.py --single checkpoints\model_epoch_050.pth
```

### 开发依赖
- 核心库：`torch`、`torchvision`
- 评估：`numpy`、`scipy`
- 图像：`Pillow`
- 配置：`pyyaml`
- 运行辅助：`itertools`、`gc`、`shutil`
- 可选：CUDA（提升训练与评估速度）

### 测试方法
- 环境自检：
  - 确认 `data\ctc_real` 目录存在且包含图像。
  - 运行 `python train.py`，观察日志和检查点写入是否正常。
- 功能检查：
  - 运行 `python generate.py`，确认生成图像出现在 `output\synthetic\`。
  - 运行 `python evaluate.py`，确认 FID 评估与排序输出。
- 资源监测：
  - 若使用 CUDA，监测显存；生成阶段默认每 50 步清理一次缓存。
  - 生成阶段会检查输出目录写权限与磁盘空间并记录日志。

### 贡献指南
- 欢迎提交 Issue 与 Pull Request。
- 详细流程与规范请参考 `CONTRIBUTING.md`（若不存在请先创建或在 PR 中补充）。

### 许可证
- 本项目采用 MIT 许可证。你可以自由地使用、复制、修改与分发本项目的代码，但需保留原始许可声明。
- 建议在仓库根目录添加 `LICENSE` 文件以正式声明。