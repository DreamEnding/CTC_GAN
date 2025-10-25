# CTC_GAN

**Language / 语言**: [English](#) | [中文](README_CN.md)

CycleGAN-based pipeline for generating synthetic CTC images and evaluating model quality via FID. Provides training, generation, and evaluation scripts, with simple logging and checkpoint management.

- License: MIT
- Contribution Guide: see CONTRIBUTING.md

### Quick Start

`cd d:\CTC_GAN`

`python -m venv .venv`

`.venv\Scripts\activate`

`pip install -U pip`

`pip install torch torchvision pillow numpy scipy pyyaml`

`python train.py`

`python generate.py`

`python evaluate.py --checkpoint_dir checkpoints\ --top_k 3`

### Overview
- Train a CycleGAN (`train.py`) to learn mappings on CTC-style images.
- Generate synthetic images using a trained `Generator` (`generate.py`).
- Evaluate checkpoints with FID (`evaluate.py`, `utils/metrics.py`).

### Installation
- Python 3.10+ recommended.
- Create a virtual environment and install dependencies:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
  - `pip install torch torchvision pillow numpy scipy pyyaml`

### Configuration
- Main configuration file: `configs\train_config.yaml`
- Key fields:
  - `model.image_size`, `model.in_channels`, `model.out_channels`
  - `training.batch_size`, `training.learning_rate`, `training.num_epochs`
  - `paths.data_dir`, `paths.output_dir`, `paths.checkpoint_dir`, `paths.log_dir`
  - `device`: `"cuda"` or `"cpu"`

### Usage Examples
- Train:
  - `python train.py`
- Generate synthetic images (edit defaults inside `generate.py` or pass args if extended):
  - `python generate.py`
- Evaluate checkpoints (FID, top-k ranking):
  - `python evaluate.py --checkpoint_dir checkpoints\ --top_k 3`
- Evaluate a single checkpoint:
  - `python evaluate.py --single checkpoints\model_epoch_050.pth`

### Development Dependencies
- `torch`, `torchvision` (model and data ops)
- `numpy`, `scipy` (FID computation)
- `Pillow` (image IO)
- `pyyaml` (config)
- `itertools`, `gc`, `shutil` (runtime utilities)
- Optional CUDA support for faster training/evaluation

### Testing
- Sanity test run:
  - Verify data folder `data\ctc_real` contains images.
  - Run `python train.py` for a few epochs.
  - Run `python evaluate.py` to compute FID for saved checkpoints.
- Functional checks:
  - Confirm images appear in `output\synthetic`.
  - Ensure logs are written to `logs\` and checkpoints to `checkpoints\`.

### Notes
- Strings in logs/prints are not translated as they are runtime outputs, not comments.
- MIT license notice included here; add a LICENSE file to formalize.