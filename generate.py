# generate.py
import yaml
import torch
from models.generator import Generator
from PIL import Image
from torchvision import transforms
import os
from utils.logger import Logger
import gc, shutil
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_synthetic_images(config_path, checkpoint_path, num_images=20, seed=None, batch_size=1):
    config = load_config(config_path)
    device = torch.device(config['device'])

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"🌱 Random seed set to {seed}")

    # Load generator
    gen = Generator(
        latent_dim=config['model']['latent_dim'],
        img_channels=config['model']['in_channels'],
        num_features=config['model']['gen_features'],
        num_residuals=config['model']['residual_blocks'],
        use_attention=config['model']['use_attention']
    ).to(device)

    gen.load_state_dict(torch.load(checkpoint_path, map_location=device)['gen_state_dict'])
    gen.eval()

    # Create output directory
    os.makedirs(config['paths']['output_dir'], exist_ok=True)

    # Initialize logger
    logger = Logger(config['paths']['log_dir'])
    logger.log_info(f"开始生成合成图像: 目标数量={num_images}, 随机种子={seed}")
    logger.log_config(config)

    # Device availability check and fallback
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        logger.log_warning("CUDA 不可用，回退到 CPU")
        device = torch.device('cpu')
        gen = gen.to(device)

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

    saved = 0
    latent_dim = config['model']['latent_dim']

    with torch.no_grad():
        while saved < num_images:
            try:
                # Sample random noise from normal distribution
                current_batch_size = min(batch_size, num_images - saved)
                noise = torch.randn(current_batch_size, latent_dim).to(device)
                
                # Generate fake images from noise
                fake = gen(noise)
                fake = (fake * 0.5 + 0.5).clamp(0, 1)  # De-normalize to [0, 1]

                # Save each image in the batch
                for i in range(current_batch_size):
                    img = transforms.ToPILImage()(fake[i].cpu())

                    # Ensure unique filenames to avoid overwriting existing files
                    save_path = os.path.join(output_dir, f"synthetic_ctc_{saved+1:05d}.png")
                    
                    img.save(save_path)
                    print(f"✅ Saved {save_path}")
                    logger.log_info(f"已保存 {save_path} ({saved+1}/{num_images})")
                    saved += 1

                # Periodic resource cleanup and status logging
                if device.type == 'cuda' and saved % 50 == 0:
                    torch.cuda.empty_cache()
                if saved % 50 == 0:
                    gc.collect()
                    free_now = shutil.disk_usage(output_dir).free
                    logger.log_info(f"进度 {saved}/{num_images}，剩余磁盘≈{free_now/1024/1024:.1f}MB")
                    
            except Exception as e:
                # Log error but continue
                logger.log_error(f"生成失败: {e}")
                print(f"❌ Error generating image: {e}")
                break
            finally:
                # Assist garbage collection
                try:
                    del noise, fake, img
                except Exception:
                    pass

    logger.log_info(f"生成完成，总计 {saved}/{num_images} 张。")
    print(f"🎉 Generation complete! Saved {saved} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic images from trained WGAN-GP model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    
    args = parser.parse_args()
    
    generate_synthetic_images(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        seed=args.seed,
        batch_size=args.batch_size
    )