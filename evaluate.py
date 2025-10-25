# evaluate.py
from utils.metrics import calculate_fid
from datasets.ctc_dataset import get_dataloader
from models.generator import Generator
import torch
import yaml
import argparse
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_fid(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['device'])

    gen = Generator(
        img_channels=config['model']['in_channels'],
        num_features=config['model']['gen_features'],
        num_residuals=config['model']['residual_blocks'],
        use_attention=config['model']['use_attention']
    ).to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device)['gen_state_dict'])
    gen.eval()

    real_dataloader = get_dataloader(config)
    fid_score = calculate_fid(real_dataloader, gen, device, num_samples=1000)
    print(f"📊 FID Score: {fid_score:.4f}")

def evaluate_all_models(config_path, checkpoint_dir, top_k=3):
    """Evaluate all models and return the best one"""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in: {checkpoint_dir}")
        return
    
    results = []
    print(f"🔍 Found {len(checkpoint_files)} checkpoint files")
    print("📊 Evaluating all models...\n")
    
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        epoch_num = int(checkpoint_file.split('_epoch_')[1].split('.')[0])
        
        print(f"[{i}/{len(checkpoint_files)}] Evaluating {checkpoint_file}...")
        try:
            fid_score = calculate_fid_for_checkpoint(config_path, checkpoint_path)
            results.append({
                'checkpoint': checkpoint_file,
                'epoch': epoch_num,
                'fid_score': fid_score,
                'path': checkpoint_path
            })
            print(f"✅ Epoch {epoch_num}: FID = {fid_score:.4f}\n")
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_file}: {e}\n")
    
    if not results:
        print("❌ No models were successfully evaluated")
        return
    
    # Sort by FID score (lower is better)
    results.sort(key=lambda x: x['fid_score'])
    
    print("🏆 Model Ranking (Best to Worst):")
    print("=" * 50)
    for i, result in enumerate(results[:top_k], 1):
        print(f"{i}. Epoch {result['epoch']:3d} | FID: {result['fid_score']:8.4f} | {result['checkpoint']}")
    
    best_model = results[0]
    print(f"\n🥇 Best Model: {best_model['checkpoint']}")
    print(f"📊 Best FID Score: {best_model['fid_score']:.4f}")
    print(f"📁 Path: {best_model['path']}")
    
    return best_model

def calculate_fid_for_checkpoint(config_path, checkpoint_path):
    """Compute FID score for a single checkpoint"""
    config = load_config(config_path)
    device = torch.device(config['device'])

    gen = Generator(
        img_channels=config['model']['in_channels'],
        num_features=config['model']['gen_features'],
        num_residuals=config['model']['residual_blocks'],
        use_attention=config['model']['use_attention']
    ).to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device)['gen_state_dict'])
    gen.eval()

    real_dataloader = get_dataloader(config)
    fid_score = calculate_fid(real_dataloader, gen, device, num_samples=1000)
    return fid_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN model using FID score')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory containing model checkpoints')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top K best models')
    parser.add_argument('--single', type=str, default=None,
                        help='Evaluate single checkpoint file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    if args.single:
        # Evaluate single model
        if not os.path.exists(args.single):
            print(f"❌ Checkpoint file not found: {args.single}")
            return
        print(f"🔍 Evaluating single model: {args.single}")
        evaluate_fid(args.config, args.single)
    else:
        # Evaluate all models
        if not os.path.exists(args.checkpoint_dir):
            print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
            return
        evaluate_all_models(args.config, args.checkpoint_dir, args.top_k)

if __name__ == '__main__':
    main()