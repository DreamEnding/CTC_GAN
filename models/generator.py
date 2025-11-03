# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Self attention Layer """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma*out + x
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=512, img_channels=3, num_features=64, num_residuals=9, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.latent_dim = latent_dim

        # Project latent vector to initial feature map (256 channels, 4x4 spatial)
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # Upsample from 4x4 to 8x8 (256 channels)
        self.upsample1 = ConvBlock(256, 256, down=False, kernel_size=4, stride=2, padding=1)
        
        # Upsample from 8x8 to 16x16 (256 channels)
        self.upsample2 = ConvBlock(256, 256, down=False, kernel_size=4, stride=2, padding=1)

        # Residual blocks at 16x16 resolution
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )

        # Self-attention at 16x16 resolution
        if use_attention:
            self.attention = SelfAttention(256)

        # Upsample from 16x16 to 32x32 (reduce channels to 128)
        self.upsample3 = ConvBlock(256, 128, down=False, kernel_size=4, stride=2, padding=1)
        
        # Upsample from 32x32 to 64x64 (reduce channels to 64)
        self.upsample4 = ConvBlock(128, 64, down=False, kernel_size=4, stride=2, padding=1)

        # Final convolution to RGB image
        self.last = nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z: [batch_size, latent_dim]
        x = self.initial(z)  # [batch_size, 256*4*4]
        x = x.view(-1, 256, 4, 4)  # [batch_size, 256, 4, 4]
        
        x = self.upsample1(x)  # [batch_size, 256, 8, 8]
        x = self.upsample2(x)  # [batch_size, 256, 16, 16]
        
        x = self.residual_blocks(x)  # [batch_size, 256, 16, 16]
        
        if self.use_attention:
            x = self.attention(x)  # [batch_size, 256, 16, 16]
        
        x = self.upsample3(x)  # [batch_size, 128, 32, 32]
        x = self.upsample4(x)  # [batch_size, 64, 64, 64]
        
        return torch.tanh(self.last(x))  # [batch_size, 3, 64, 64]