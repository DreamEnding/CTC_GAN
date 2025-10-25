# generate.py
import yaml
import torch
from models.generator import Generator
from PIL import Image
from torchvision import transforms
import os
from utils.logger import Logger
import itertools, gc, shutil

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_synthetic_images(config_path, checkpoint_path, num_images=20):
    config = load_config(config_path)
    device = torch.device(config['device'])

    # Load generator
    gen = Generator(
        img_channels=config['model']['in_channels'],
        num_features=config['model']['gen_features'],
        num_residuals=config['model']['residual_blocks'],
        use_attention=config['model']['use_attention']
    ).to(device)

    gen.load_state_dict(torch.load(checkpoint_path, map_location=device)['gen_state_dict'])
    gen.eval()

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load real images for generation
    from datasets.ctc_dataset import CTCImageDataset
    dataset = CTCImageDataset(config['paths']['data_dir'], transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Create output directory
    os.makedirs(config['paths']['output_dir'], exist_ok=True)

    # Initialize logger
    logger = Logger(config['paths']['log_dir'])
    logger.log_info(f"开始生成合成图像: 目标数量={num_images}")
    logger.log_config(config)

    # Device availability check and fallback
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        logger.log_warning("CUDA 不可用，回退到 CPU")
        device = torch.device('cpu')

    # Output directory and write permission check
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        logger.log_error(f"输出目录不可写: {output_dir}")
        raise PermissionError(f"输出目录不可写: {output_dir}")

    # Pre-check disk space (rough estimate)
    total, used, free = shutil.disk_usage(output_dir)
    est_per_img_kb = 300  # Empirical estimate, average PNG size ≈ 300KB
    required = est_per_img_kb * 1024 * num_images
    if free < required:
        logger.log_warning(f"磁盘空间可能不足: 需要≈{required/1024/1024:.1f}MB，可用≈{free/1024/1024:.1f}MB")

    # Iterate over DataLoader endlessly to exceed dataset size [译注: uses itertools.cycle to loop infinitely]
    data_iter = itertools.cycle(dataloader)
    saved = 0
    step = 0

    with torch.no_grad():
        while saved < num_images:
            try:
                real = next(data_iter).to(device)
                fake = gen(real)
                fake = (fake * 0.5 + 0.5).clamp(0, 1)  # De-normalize to [0, 1]

                img = transforms.ToPILImage()(fake.squeeze(0).cpu())

                # Ensure unique filenames to avoid overwriting existing files
                save_path = os.path.join(output_dir, f"synthetic_ctc_{saved+1:03d}.png")
                while os.path.exists(save_path):
                    save_path = os.path.join(output_dir, f"synthetic_ctc_{saved+1:03d}_{step}.png")

                img.save(save_path)
                print(f"✅ Saved {save_path}")
                logger.log_info(f"已保存 {save_path} ({saved+1}/{num_images})")
                saved += 1

                step += 1
                # Periodic resource cleanup and status logging
                if device.type == 'cuda' and step % 50 == 0:
                    torch.cuda.empty_cache()
                if step % 50 == 0:
                    gc.collect()
                    free_now = shutil.disk_usage(output_dir).free
                    logger.log_info(f"进度 {saved}/{num_images}，剩余磁盘≈{free_now/1024/1024:.1f}MB")
            except Exception as e:
                # Do not abort or increment count; continue until target is reached
                logger.log_error(f"第 {step} 次生成失败: {e}")
            finally:
                # Assist garbage collection
                try:
                    del real, fake, img
                except Exception:
                    pass

    logger.log_info(f"生成完成，总计 {saved}/{num_images} 张。")

if __name__ == "__main__":
    generate_synthetic_images(
        config_path="configs/train_config.yaml",
        checkpoint_path="",
        num_images=100
    )