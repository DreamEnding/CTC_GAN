import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 model for feature extraction"""
    
    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        
        self.blocks = nn.ModuleList()
        
        inception = models.inception_v3(pretrained=True)
        
        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        
        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        
        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, inp):
        """Get Inception features
        
        Args:
            inp: Input tensor of shape (N, 3, H, W)
            
        Returns:
            list: Features from requested output blocks
        """
        outp = []
        x = inp
        
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = 2 * x - 1  # Scale [0, 1] to [-1, 1]
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            
            if idx == self.last_needed_block:
                break
        
        return outp

def get_activations(dataloader, model, device, num_samples=None):
    """Get Inception activations of real data
    
    Args:
        dataloader: DataLoader
        model: Inception model
        device: Device
        num_samples: Number of samples; None means use all data
        
    Returns:
        numpy.ndarray: Activation array
    """
    model.eval()
    activations = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # If it's a tuple, take the first element
            
            batch = batch.to(device)
            
            # Ensure inputs are in [0, 1] range
            if batch.min() < 0:
                batch = (batch + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            pred = model(batch)[0]
            
            # If output is 4D, flatten it
            if len(pred.shape) == 4:
                pred = pred.squeeze(3).squeeze(2)
            
            activations.append(pred.cpu().numpy())
            
            if num_samples is not None and (i + 1) * batch.size(0) >= num_samples:
                break
    
    activations = np.concatenate(activations, axis=0)
    
    if num_samples is not None:
        activations = activations[:num_samples]
    
    return activations

def get_generated_activations(generator, dataloader, model, device, num_samples=None):
    """Get Inception activations of generated data
    
    Args:
        generator: Generator model
        dataloader: Real data loader
        model: Inception model
        device: Device
        num_samples: Number of samples
        
    Returns:
        numpy.ndarray: Activation array
    """
    generator.eval()
    model.eval()
    activations = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(device)
            
            # Generate fake data
            fake = generator(batch)
            
            # Ensure outputs are in [0, 1] range
            fake = (fake + 1) / 2  # Convert from [-1, 1] to [0, 1]
            fake = torch.clamp(fake, 0, 1)
            
            pred = model(fake)[0]
            
            if len(pred.shape) == 4:
                pred = pred.squeeze(3).squeeze(2)
            
            activations.append(pred.cpu().numpy())
            
            if num_samples is not None and (i + 1) * batch.size(0) >= num_samples:
                break
    
    activations = np.concatenate(activations, axis=0)
    
    if num_samples is not None:
        activations = activations[:num_samples]
    
    return activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Frechet distance between two multivariate Gaussian distributions
    
    Args:
        mu1: Mean of the first distribution
        sigma1: Covariance of the first distribution
        mu2: Mean of the second distribution
        sigma2: Covariance of the second distribution
        eps: Small value for numerical stability
        
    Returns:
        float: Frechet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    
    diff = mu1 - mu2
    
    # Compute the square root of the covariance product
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical errors may introduce tiny imaginary parts
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(activations):
    """Compute statistics of activations
    
    Args:
        activations: Activation array
        
    Returns:
        tuple: (mean, covariance matrix)
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid(real_dataloader, generator, device, num_samples=1000):
    """Compute FID score
    
    Args:
        real_dataloader: Real data loader
        generator: Generator model
        device: Compute device
        num_samples: Number of samples used for FID
        
    Returns:
        float: FID score
    """
    print(f"🔄 Calculating FID with {num_samples} samples...")
    
    # Initialize Inception model
    inception_model = InceptionV3().to(device)
    inception_model.eval()
    
    # Extract activations from real images
    real_activations = get_activations(real_dataloader, inception_model, device, num_samples)
    
    # Extract activations from generated images
    fake_activations = get_generated_activations(generator, real_dataloader, inception_model, device, num_samples)
    
    # Compute statistics
    mu_real, sigma_real = calculate_activation_statistics(real_activations)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_activations)
    
    # Compute FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score