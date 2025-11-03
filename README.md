# CTC_GAN

**Language / 语言**: [English](#) | [中文](README_CN.md)

WGAN-GP (Wasserstein GAN with Gradient Penalty) for generating synthetic CTC images from random noise. The model learns the distribution of training data and can generate unlimited unique, high-quality images.

- License: MIT
- Contribution Guide: see CONTRIBUTING.md

### Quick Start

```bash
cd CTC_GAN

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -U pip
pip install torch torchvision pillow numpy scipy pyyaml

# Train the model
python train.py

# Generate synthetic images from random noise
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 100

# Evaluate model quality
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

### Overview
- **Train** a WGAN-GP model (`train.py`) to learn the distribution of CTC-style images
- **Generate** unlimited synthetic images from random noise (`generate.py`)
- **Evaluate** model quality using FID metric (`evaluate.py`, `utils/metrics.py`)

### Key Features
✅ **Unconditional Generation**: Generate images from random noise (no input image required)  
✅ **Infinite Variety**: Create unlimited unique images by sampling different random seeds  
✅ **High Quality**: Uses ResNet blocks and self-attention for detailed outputs  
✅ **WGAN-GP Training**: Stable training with Wasserstein loss and gradient penalty  
✅ **Reproducible**: Set random seeds to regenerate specific images

### Installation
- Python 3.7+ recommended
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  pip install torch torchvision pillow numpy scipy pyyaml
  ```

### Configuration
- Main configuration file: `configs/train_config.yaml`
- Key fields:
  - `model.latent_dim`: Dimension of random noise vector (default: 512)
  - `model.image_size`: Output image size (default: 64x64)
  - `model.gen_features`, `model.disc_features`: Base feature dimensions
  - `model.residual_blocks`: Number of ResNet blocks (default: 9)
  - `model.use_attention`: Enable self-attention layer (default: True)
  - `training.batch_size`, `training.learning_rate`, `training.num_epochs`
  - `paths.data_dir`, `paths.output_dir`, `paths.checkpoint_dir`, `paths.log_dir`
  - `device`: `"cuda"` or `"cpu"`

### Usage Examples

**Training:**
```bash
python train.py
```
The model learns from real images in `data/ctc_real/` but generates from random noise.

**Generating Images:**
```bash
# Generate 100 images
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 100

# Generate with specific seed for reproducibility
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 50 --seed 42

# Generate with larger batch size (faster)
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 1000 --batch_size 16
```

**Evaluation:**
```bash
# Evaluate all checkpoints and rank top 3
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3

# Evaluate a single checkpoint
python evaluate.py --single checkpoints/model_epoch_050.pth
```

### Architecture

**Generator:**
```
Input: Random noise z ~ N(0, 1), shape [batch_size, 512]
  ↓
Dense + Reshape to [batch_size, 256, 4, 4]
  ↓
Upsample to 8x8 (256 channels)
  ↓
Upsample to 16x16 (256 channels)
  ↓
9 Residual Blocks (256 channels)
  ↓
Self-Attention (optional)
  ↓
Upsample to 32x32 (128 channels)
  ↓
Upsample to 64x64 (64 channels)
  ↓
Output: 3x64x64 RGB image, range [-1, 1]
```

**Discriminator:**
- WGAN-GP critic with gradient penalty
- Multi-scale convolutional architecture
- Distinguishes real images from generated ones

### Development Dependencies
- `torch`, `torchvision` (model and data ops)
- `numpy`, `scipy` (FID computation)
- `Pillow` (image IO)
- `pyyaml` (config)
- Optional CUDA support for faster training/evaluation

### Model Characteristics
- **Unconditional GAN**: Generates images from random noise, not from input images
- **No reconstruction loss**: Pure adversarial training (no L1/L2 loss)
- **WGAN-GP**: Wasserstein loss with gradient penalty for stable training
- **Scalable**: Can generate unlimited unique images

### Testing
- Verify data folder `data/ctc_real` contains training images
- Run `python train.py` for a few epochs
- Generate images with `python generate.py --checkpoint <path> --num_images 10`
- Confirm generated images appear in `output/synthetic/`
- Ensure logs are written to `logs/` and checkpoints to `checkpoints/`

### Notes
- The model learns the **distribution** of training data, not specific images
- Different random seeds produce different outputs
- Training requires real images, but generation does not
- MIT license