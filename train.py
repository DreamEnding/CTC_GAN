# train.py
import yaml
import torch
from models.cyclegan_system import CycleGANSystem
from datasets.ctc_dataset import get_dataloader
from utils.logger import Logger
from utils.checkpoint import CheckpointManager
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('configs/train_config.yaml')
    device = torch.device(config['device'])

    # Initialize system
    gan_system = CycleGANSystem(config, device)
    dataloader = get_dataloader(config)
    logger = Logger(config['paths']['log_dir'])
    ckpt_manager = CheckpointManager(config['paths']['checkpoint_dir'])

    start_epoch = 1
    if ckpt_manager.has_checkpoint():
        start_epoch = gan_system.load_checkpoint(ckpt_manager.latest_checkpoint()) + 1
        print(f".Resume training from epoch {start_epoch}")

    print("🚀 Starting Training...")
    gan_system.step_count = 0

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        num_batches = 0

        for batch_idx, real in enumerate(dataloader):
            gan_system.step_count += 1
            losses = gan_system.train_step(real)

            epoch_disc_loss += losses['loss_disc']
            epoch_gen_loss += losses['loss_gen']
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{config['training']['num_epochs']}] "
                      f"Batch {batch_idx}: D={losses['loss_disc']:.4f}, G={losses['loss_gen']:.4f}, GP={losses['gp']:.4f}")

        avg_disc_loss = epoch_disc_loss / num_batches
        avg_gen_loss = epoch_gen_loss / num_batches

        logger.log_scalar('Loss/Discriminator', avg_disc_loss, epoch)
        logger.log_scalar('Loss/Generator', avg_gen_loss, epoch)

        print(f"✅ Epoch {epoch} | Avg D Loss: {avg_disc_loss:.4f} | Avg G Loss: {avg_gen_loss:.4f}")

        # Save model
        if epoch % config['training']['save_every'] == 0:
            ckpt_path = os.path.join(config['paths']['checkpoint_dir'], f"model_epoch_{epoch:03d}.pth")
            gan_system.save_checkpoint(epoch, ckpt_path)
            print(f"💾 Model saved at {ckpt_path}")

        gan_system.scheduler_gen.step()
        gan_system.scheduler_disc.step()

    print("🎉 Training completed!")

if __name__ == "__main__":
    main()