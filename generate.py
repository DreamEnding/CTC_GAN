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

def generate_synthetic_images(config_path, checkpoint_path, num_images=20, seed=None, 
                              batch_size=1, num_variants_per_image=5):
    """
    Generate synthetic images using real images + noise injection
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        num_images: Number of source real images to use
        seed: Random seed for reproducibility
        batch_size: Batch size for generation (not used in variant mode)
        num_variants_per_image: Number of variants to generate per source image
    """
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
        use_attention=config['model']['use_attention'],
        use_noise_injection=config['model'].get('use_noise_injection', True)
    ).to(device)

    gen.load_state_dict(torch.load(checkpoint_path, map_location=device)['gen_state_dict'])
    gen.eval()

    # Initialize logger
    logger = Logger(config['paths']['log_dir'])
    logger.log_info(f"开始生成合成图像: 源图像数={num_images}, 每张变体数={num_variants_per_image}, 随机种子={seed}")
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
    total_images = num_images * num_variants_per_image
    total, used, free = shutil.disk_usage(output_dir)
    est_per_img_kb = 300  # Empirical estimate, average PNG size ≈ 300KB
    required = est_per_img_kb * 1024 * total_images
    if free < required:
        logger.log_warning(f"磁盘空间可能不足: 需要≈{required/1024/1024:.1f}MB，可用≈{free/1024/1024:.1f}MB")

    # Load real images dataset
    from datasets.ctc_dataset import CTCImageDataset, get_transforms
    transform = get_transforms(config['model']['image_size'])
    dataset = CTCImageDataset(config['paths']['data_dir'], transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    saved = 0
    latent_dim = config['model']['latent_dim']

    with torch.no_grad():
        for img_idx, real in enumerate(dataloader):
            if img_idx >= num_images:
                break
                
            real = real.to(device)
            
            # Generate multiple variants for each real image
            for variant_idx in range(num_variants_per_image):
                try:
                    # Sample different noise for each variant
                    noise = torch.randn(1, latent_dim, device=device)
                    
                    # Generate variant image
                    fake = gen(real, noise)
                    fake = (fake * 0.5 + 0.5).clamp(0, 1)  # De-normalize to [0, 1]

                    img = transforms.ToPILImage()(fake.squeeze(0).cpu())

                    # Save with informative filename
                    save_path = os.path.join(output_dir, f"generated_{saved:05d}_src{img_idx:03d}_var{variant_idx:03d}.png")
                    
                    img.save(save_path)
                    print(f"✅ Saved {save_path}")
                    logger.log_info(f"已保存 {save_path} ({saved+1}/{total_images})")
                    saved += 1

                    # Periodic resource cleanup
                    if device.type == 'cuda' and saved % 50 == 0:
                        torch.cuda.empty_cache()
                    if saved % 50 == 0:
                        gc.collect()
                        free_now = shutil.disk_usage(output_dir).free
                        logger.log_info(f"进度 {saved}/{total_images}，剩余磁盘≈{free_now/1024/1024:.1f}MB")
                        
                except Exception as e:
                    logger.log_error(f"生成失败: {e}")
                    print(f"❌ Error generating image: {e}")
                    continue
                finally:
                    try:
                        del noise, fake, img
                    except Exception:
                        pass

    logger.log_info(f"生成完成，总计 {saved}/{total_images} 张图像，来自 {img_idx+1} 张源图像。")
    print(f"🎉 Generation complete! Generated {saved} images from {img_idx+1} source images")
    print(f"   Average {saved/(img_idx+1):.1f} variants per source image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic images using real images + noise injection")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_images", type=int, default=10, help="Number of source real images to use")
    parser.add_argument("--num_variants_per_image", type=int, default=10, help="Number of variants to generate per source image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (deprecated, kept for compatibility)")
    
    args = parser.parse_args()
    
    generate_synthetic_images(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        seed=args.seed,
        batch_size=args.batch_size,
        num_variants_per_image=args.num_variants_per_image
    )