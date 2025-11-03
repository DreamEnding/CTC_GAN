# models/cyclegan_system.py
import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator, gradient_penalty

class CycleGANSystem:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Initialize generator and discriminator
        self.gen = Generator(
            latent_dim=config['model']['latent_dim'],
            img_channels=config['model']['in_channels'],
            num_features=config['model']['gen_features'],
            num_residuals=config['model']['residual_blocks'],
            use_attention=config['model']['use_attention']
        ).to(device)

        self.disc = Discriminator(
            in_channels=config['model']['in_channels'],
            features=[config['model']['disc_features'], config['model']['disc_features']*2, 
                     config['model']['disc_features']*4, config['model']['disc_features']*8]
        ).to(device)

        # Optimizers
        self.opt_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        self.opt_disc = torch.optim.Adam(
            self.disc.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )

        # Learning rate schedulers
        self.scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_gen, T_max=config['training']['num_epochs']
        )
        self.scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc, T_max=config['training']['num_epochs']
        )
        
        # Training step counter
        self.step_count = 0

    def train_step(self, real):
        real = real.to(self.device)
        batch_size = real.shape[0]
        
        # Increment step counter
        self.step_count += 1

        # Sample random noise
        noise = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)

        # --- Train Discriminator ---
        fake = self.gen(noise)
        disc_real = self.disc(real)
        disc_fake = self.disc(fake.detach())

        gp = gradient_penalty(self.disc, real, fake, self.device)
        loss_disc = (
            -(torch.mean(disc_real) - torch.mean(disc_fake)) +
            self.config['training']['lambda_gp'] * gp
        )

        self.opt_disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()

        # --- Train Generator ---
        if self._should_train_gen():
            # Resample noise for generator training
            noise = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
            fake = self.gen(noise)
            disc_fake = self.disc(fake)
            loss_gen = -torch.mean(disc_fake)

            self.opt_gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()
        else:
            loss_gen = torch.tensor(0.0).to(self.device)

        return {
            'loss_disc': loss_disc.item(),
            'loss_gen': loss_gen.item(),
            'gp': gp.item()
        }

    def _should_train_gen(self):
        # Train generator every 5 steps to stabilize training
        return self.step_count % 5 == 0

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'gen_state_dict': self.gen.state_dict(),
            'disc_state_dict': self.disc.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.gen.load_state_dict(ckpt['gen_state_dict'])
        self.disc.load_state_dict(ckpt['disc_state_dict'])
        self.opt_gen.load_state_dict(ckpt['opt_gen_state_dict'])
        self.opt_disc.load_state_dict(ckpt['opt_disc_state_dict'])
        return ckpt['epoch']