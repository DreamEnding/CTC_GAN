# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseEncoder(nn.Module):
    """Encodes noise vector into style parameters for AdaIN"""
    def __init__(self, latent_dim=512, style_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, style_dim * 2)  # Output: [mean, std] for AdaIN
        )
        
    def forward(self, noise):
        """
        Args:
            noise: [B, latent_dim]
        Returns:
            style_params: [B, style_dim * 2] (concatenated mean and std)
        """
        return self.mlp(noise)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer"""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, content_features, style_params):
        """
        Apply AdaIN: modulate content features using style parameters
        
        Args:
            content_features: [B, C, H, W]
            style_params: [B, C*2] (mean and std concatenated)
        Returns:
            modulated_features: [B, C, H, W]
        """
        # Validate input dimensions
        batch_size, num_channels = content_features.size(0), content_features.size(1)
        expected_style_dim = num_channels * 2
        if style_params.size(1) != expected_style_dim:
            raise ValueError(f"Expected style_params dimension {expected_style_dim}, got {style_params.size(1)}")
        
        # Split style params into mean and std
        style_mean, style_std = style_params.chunk(2, dim=1)
        style_mean = style_mean.unsqueeze(-1).unsqueeze(-1)
        style_std = style_std.unsqueeze(-1).unsqueeze(-1)
        
        # Normalize content features
        size = content_features.size()
        content_mean = content_features.mean(dim=[2, 3], keepdim=True)
        content_std = content_features.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized_features = (content_features - content_mean) / content_std
        
        # Apply style modulation
        return normalized_features * style_std + style_mean

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
    def __init__(self, channels, use_adain=False):
        super().__init__()
        self.use_adain = use_adain
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")
        
        if use_adain:
            self.adain1 = AdaIN(channels)
            self.adain2 = AdaIN(channels)
        else:
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style_params=None):
        residual = x
        
        out = self.conv1(x)
        if self.use_adain and style_params is not None:
            out = self.adain1(out, style_params)
        else:
            out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if self.use_adain and style_params is not None:
            out = self.adain2(out, style_params)
        else:
            out = self.norm2(out)
        
        return residual + out

class Generator(nn.Module):
    def __init__(self, latent_dim=512, img_channels=3, num_features=64, num_residuals=9, use_attention=True, use_noise_injection=True):
        super().__init__()
        self.use_attention = use_attention
        self.latent_dim = latent_dim
        self.use_noise_injection = use_noise_injection

        # Image encoder: Encode input image to feature space
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            ConvBlock(img_channels, num_features, down=True, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 16x16
            ConvBlock(num_features, num_features * 2, down=True, kernel_size=4, stride=2, padding=1),
            # 16x16 -> 16x16 (keep resolution, increase channels)
            ConvBlock(num_features * 2, 256, down=True, kernel_size=3, stride=1, padding=1)
        )

        # Noise encoder: Encode noise vector to style parameters
        if use_noise_injection:
            self.noise_encoder = NoiseEncoder(latent_dim=latent_dim, style_dim=256)

        # Residual blocks with optional AdaIN
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256, use_adain=use_noise_injection) for _ in range(num_residuals)
        ])

        # Self-attention at 16x16 resolution
        if use_attention:
            self.attention = SelfAttention(256)

        # Decoder: Upsample back to original resolution
        # 16x16 -> 32x32
        self.upsample1 = ConvBlock(256, 128, down=False, kernel_size=4, stride=2, padding=1)
        # 32x32 -> 64x64
        self.upsample2 = ConvBlock(128, 64, down=False, kernel_size=4, stride=2, padding=1)

        # Final convolution to RGB image
        self.last = nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, noise=None):
        """
        Forward pass with hybrid image+noise input
        
        Args:
            x: Input image [B, 3, H, W]
            noise: Optional noise vector [B, latent_dim] for style injection
                   If None and use_noise_injection=True, random noise is sampled
                   
        Returns:
            Generated image [B, 3, H, W]
        """
        # Encode input image to content features
        content_features = self.encoder(x)  # [B, 256, 16, 16]
        
        # Encode noise to style parameters if using noise injection
        style_params = None
        if self.use_noise_injection:
            if noise is None:
                # Sample random noise if not provided
                noise = torch.randn(x.size(0), self.latent_dim, device=x.device)
            style_params = self.noise_encoder(noise)  # [B, 512]
        
        # Apply residual blocks with optional style modulation
        features = content_features
        for res_block in self.residual_blocks:
            features = res_block(features, style_params)
        
        # Apply self-attention
        if self.use_attention:
            features = self.attention(features)
        
        # Decode to output image
        features = self.upsample1(features)  # [B, 128, 32, 32]
        features = self.upsample2(features)  # [B, 64, 64, 64]
        
        return torch.tanh(self.last(features))  # [B, 3, 64, 64]