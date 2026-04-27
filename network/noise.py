import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class Identity(nn.Module):
    def forward(self, encoded, cover):
        return encoded


class MedianBlur(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2
    def forward(self, encoded, cover):
        return F.avg_pool2d(encoded, self.k, stride=1, padding=self.pad).clamp(-1, 1)


class Resize(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale
    def forward(self, encoded, cover):
        small = F.interpolate(encoded, scale_factor=self.scale, mode='nearest')
        return F.interpolate(small, size=encoded.shape[2:], mode='nearest')


class GaussianBlur(nn.Module):
    """Pure PyTorch Gaussian blur — no kornia needed."""
    def __init__(self, kernel_size=3, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # Build fixed Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)   # (1,1,k,k)
        self.register_buffer('kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, encoded, cover):
        return F.conv2d(encoded, self.kernel, padding=self.padding, groups=3).clamp(-1, 1)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
    def forward(self, encoded, cover):
        return (encoded + torch.randn_like(encoded) * self.std).clamp(-1, 1)


class Brightness(nn.Module):
    """Random brightness shift — no kornia needed."""
    def forward(self, encoded, cover):
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.8   # [0.6, 1.4]
        x = (encoded + 1) / 2
        return (x * factor).clamp(0, 1) * 2 - 1


class Contrast(nn.Module):
    """Random contrast adjustment — no kornia needed."""
    def forward(self, encoded, cover):
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.8   # [0.6, 1.4]
        x = (encoded + 1) / 2
        mean = x.mean(dim=[2, 3], keepdim=True)
        return ((mean + factor * (x - mean)).clamp(0, 1)) * 2 - 1


class SaltPepper(nn.Module):
    def __init__(self, prob=0.05):
        super().__init__()
        self.prob = prob
    def forward(self, encoded, cover):
        out  = encoded.clone()
        mask = torch.rand_like(encoded)
        out[mask < self.prob / 2]   =  1.0
        out[mask > 1 - self.prob/2] = -1.0
        return out


class Dropout(nn.Module):
    def __init__(self, prob=0.15):
        super().__init__()
        self.prob = prob
    def forward(self, encoded, cover):
        mask = (torch.rand(encoded.shape[0], 1, *encoded.shape[2:],
                           device=encoded.device) > self.prob).float()
        return encoded * mask + cover * (1 - mask)


# ── robustness pool ───────────────────────────────────────────────────────────

POOL_R = [
    Identity(),
    MedianBlur(kernel_size=3),
    Resize(scale=0.5),
    GaussianBlur(),
    GaussianNoise(std=0.05),
    Brightness(),
    Contrast(),
    SaltPepper(prob=0.05),
    Dropout(prob=0.15),
]


def apply_random_R(encoded, cover):
    layer = random.choice(POOL_R)
    return layer(encoded, cover)


def apply_explicit_F(encoded, images):
    B, C, H, W = encoded.shape
    forged = encoded.clone()
    mask   = torch.zeros(B, 1, H, W, device=encoded.device)

    idx   = torch.randperm(B, device=encoded.device)
    other = images[idx]

    for i in range(B):
        h1 = random.randint(0, H // 3)
        h2 = random.randint(2 * H // 3, H)
        w1 = random.randint(0, W // 3)
        w2 = random.randint(2 * W // 3, W)
        forged[i, :, h1:h2, w1:w2] = other[i, :, h1:h2, w1:w2]
        mask[i, 0, h1:h2, w1:w2]   = 1.0

    return forged, mask