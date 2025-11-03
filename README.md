# CTC_GAN

**Language / 语言**: [English](#) | [中文](README_CN.md)

WGAN-GP (Wasserstein GAN with Gradient Penalty) with **AdaIN-based noise injection** for generating diverse synthetic CTC images. The model uses a hybrid approach: real images provide content/structure while random noise adds diversity, enabling infinite unique high-quality variants.

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

# Generate diverse variants from real images + noise
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 10

# Evaluate model quality
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3
```

### Overview
- **Train** a WGAN-GP model (`train.py`) with AdaIN-based noise injection
- **Generate** unlimited diverse variants by combining real images with different noise vectors (`generate.py`)
- **Evaluate** model quality using FID metric (`evaluate.py`, `utils/metrics.py`)

### Key Features
✅ **Hybrid Architecture**: Combines real image content with random noise for diversity  
✅ **Infinite Variety**: One image + different noise = unlimited unique variants  
✅ **High Quality**: AdaIN style modulation preserves semantic content while adding variation  
✅ **WGAN-GP Training**: Stable training with Wasserstein loss and gradient penalty  
✅ **Perceptual Loss**: Maintains semantic similarity without forcing pixel-perfect reconstruction  
✅ **Reproducible**: Set random seeds to regenerate specific variants

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
  - `model.use_noise_injection`: Enable AdaIN-based noise injection (default: True)
  - `model.gen_features`, `model.disc_features`: Base feature dimensions
  - `model.residual_blocks`: Number of ResNet blocks (default: 9)
  - `model.use_attention`: Enable self-attention layer (default: True)
  - `training.lambda_perceptual`: Weight for perceptual loss (default: 1.0)
  - `training.lambda_cycle`: Weight for cycle consistency (set to 0.0 for pure diversity)
  - `training.batch_size`, `training.learning_rate`, `training.num_epochs`
  - `paths.data_dir`, `paths.output_dir`, `paths.checkpoint_dir`, `paths.log_dir`
  - `device`: `"cuda"` or `"cpu"`

### Usage Examples

**Training:**
```bash
python train.py
```
The model learns to generate diverse variants from real images in `data/ctc_real/` by injecting noise.

**Generating Images:**
```bash
# Generate 100 variants from 10 source images (10 variants each)
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 10

# Generate 1000 variants from 10 source images (100 variants each)
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 10 --num_variants_per_image 100

# Generate with specific seed for reproducibility
python generate.py --checkpoint checkpoints/model_epoch_100.pth --num_images 5 --num_variants_per_image 20 --seed 42
```

**Evaluation:**
```bash
# Evaluate all checkpoints and rank top 3
python evaluate.py --checkpoint_dir checkpoints/ --top_k 3

# Evaluate a single checkpoint
python evaluate.py --single checkpoints/model_epoch_050.pth
```

### Architecture

**Hybrid Generator with AdaIN:**
```
Input 1: Real image x, shape [batch_size, 3, 64, 64]
Input 2: Random noise z ~ N(0, 1), shape [batch_size, 512]
  ↓
[Image Encoder]
  Real image → Downsample to 16x16 (256 channels) → Content features
  ↓
[Noise Encoder]  
  Noise vector → MLP → Style parameters (mean, std)
  ↓
[AdaIN Modulation]
  9 Residual Blocks with AdaIN (fuse content + style)
  ↓
[Self-Attention]
  Capture global dependencies (optional)
  ↓
[Decoder]
  Upsample to 32x32 (128 channels)
  ↓
  Upsample to 64x64 (64 channels)
  ↓
Output: 3x64x64 RGB variant image, range [-1, 1]
```

**Key Innovation - AdaIN (Adaptive Instance Normalization):**
- Normalizes content features from the image
- Modulates them using style parameters derived from noise
- Formula: `AdaIN(content, style) = σ_style * normalize(content) + μ_style`
- Result: Content structure preserved, details varied by noise

**Discriminator:**
- WGAN-GP critic with gradient penalty
- Multi-scale convolutional architecture
- Distinguishes real images from generated variants

### Development Dependencies
- `torch`, `torchvision` (model and data ops, VGG19 for perceptual loss)
- `numpy`, `scipy` (FID computation)
- `Pillow` (image IO)
- `pyyaml` (config)
- Optional CUDA support for faster training/evaluation

### Model Characteristics
- **Hybrid Architecture**: Real image provides content, noise provides diversity
- **AdaIN-based Style Injection**: Modulates features while preserving structure
- **Perceptual Loss**: Maintains semantic similarity (VGG features) instead of pixel-level reconstruction
- **WGAN-GP**: Wasserstein loss with gradient penalty for stable training
- **Infinite Scalability**: One image + different noise = unlimited unique variants

### Training Strategy
1. **Discriminator**: Distinguishes between real images and generated variants
2. **Generator**:
   - Takes real image + random noise as input
   - Adversarial loss: Fools the discriminator
   - Perceptual loss: Maintains semantic similarity to source image
   - No pixel-wise reconstruction loss (allows creative variation)

### Testing
- Verify data folder `data/ctc_real` contains training images
- Run `python train.py` for a few epochs
- Generate variants with `python generate.py --checkpoint <path> --num_images 5 --num_variants_per_image 10`
- Confirm generated images appear in `output/synthetic/`
- Check that variants of the same source differ in details but preserve structure
- Ensure logs are written to `logs/` and checkpoints to `checkpoints/`

### Expected Results
- ✅ Generated variants preserve main features of source images (color, structure, style)
- ✅ Each variant has unique details (textures, fine structures, local variations)
- ✅ High quality output without noise artifacts
- ✅ Training is stable and converges reliably
- ✅ Can generate unlimited variants per source image

### Notes
- The model learns to **generate diverse variants** while preserving content
- Different noise vectors produce different outputs from the same source
- Training and generation both require real images (unlike pure noise GANs)
- Use `--seed` parameter for reproducible generation
- MIT license