# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for comparing feature similarity instead of pixel values"""
    def __init__(self):
        super().__init__()
        # Use VGG19 features up to relu3_4 (16th layer)
        vgg = vgg19(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, generated, target):
        """
        Calculate perceptual loss between generated and target images
        
        Args:
            generated: Generated images [B, 3, H, W]
            target: Target images [B, 3, H, W]
            
        Returns:
            Perceptual loss (MSE between VGG features)
        """
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        return F.mse_loss(gen_features, target_features)
