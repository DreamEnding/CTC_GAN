# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        for i, feature in enumerate(features[1:]):
            stride = 1 if i == len(features) - 2 else 2
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.InstanceNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)

def gradient_penalty(disc, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    disc_scores = disc(interpolated_images)
    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=disc_scores,
        grad_outputs=torch.ones_like(disc_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(BATCH_SIZE, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp