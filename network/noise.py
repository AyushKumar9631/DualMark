"""
network/noise.py  —  Simple, self-contained noise layers for training.

pool_R  (robustness — used for DecoderR training):
    Identity, MedianBlur, Resize, GaussianBlur, GaussianNoise,
    Brightness, Contrast, SaltPepper, Dropout

    JpegTest removed — it does disk I/O per image inside the training loop
    which causes severe slowdowns. GaussianNoise + Resize already cover the
    high-frequency loss that JPEG would simulate.

apply_explicit_F  (forgery simulation — used for DecoderF training):
    Splices a random rectangle from a shuffled-batch image into the encoded
    image and returns (forged, exact_mask).  The mask is always non-zero so
    DecoderF always has a real forgery signal to learn from.

No external pretrained models required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.augmentation as K
import random


class Identity(nn.Module):
    def forward(self, encoded, cover):
        return encoded


class MedianBlur(nn.Module):
    """Fast approximate median blur via average pooling — no disk I/O."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2
    def forward(self, encoded, cover):
        x = F.avg_pool2d(encoded, self.k, stride=1, padding=self.pad)
        return x.clamp(-1, 1)


class Resize(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale
    def forward(self, encoded, cover):
        small = F.interpolate(encoded, scale_factor=self.scale, mode='nearest')
        return F.interpolate(small, size=encoded.shape[2:], mode='nearest')


class GaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.RandomGaussianBlur((3,3), (1,2), p=1.0)
    def forward(self, encoded, cover):
        return self.aug(encoded)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
    def forward(self, encoded, cover):
        return (encoded + torch.randn_like(encoded) * self.std).clamp(-1,1)


class Brightness(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.ColorJitter(brightness=0.4, p=1.0)
    def forward(self, encoded, cover):
        x = (encoded + 1) / 2
        return self.aug(x) * 2 - 1


class Contrast(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.ColorJitter(contrast=0.4, p=1.0)
    def forward(self, encoded, cover):
        x = (encoded + 1) / 2
        return self.aug(x) * 2 - 1


class SaltPepper(nn.Module):
    def __init__(self, prob=0.05):
        super().__init__()
        self.prob = prob
    def forward(self, encoded, cover):
        out = encoded.clone()
        mask = torch.rand_like(encoded)
        out[mask < self.prob/2]   =  1.0
        out[mask > 1-self.prob/2] = -1.0
        return out


class Dropout(nn.Module):
    """Replace random pixels with cover-image pixels."""
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
    MedianBlur(kernel_size=3),   # replaces JpegTest — fast, no disk I/O
    Resize(scale=0.5),
    GaussianBlur(),
    GaussianNoise(std=0.05),
    Brightness(),
    Contrast(),
    SaltPepper(prob=0.05),
    Dropout(prob=0.15),
]


def apply_random_R(encoded, cover):
    """Apply a random robustness distortion from POOL_R."""
    layer = random.choice(POOL_R)
    return layer(encoded, cover)


# ── explicit forgery (rectangle splice) ──────────────────────────────────────

def apply_explicit_F(encoded, images):
    """
    Create a ground-truth forgery by splicing a random rectangle from a
    shuffled batch of cover images into the encoded image.

    Returns (forged, mask) where mask is EXPLICITLY constructed — not derived
    from pixel diffs. This ensures the ground-truth label is always correct
    regardless of how subtle the watermark embedding is.

    The spliced region covers 33–83% of both H and W, starting from a random
    offset in the top-left third, so DecoderF always sees a meaningful
    manipulated area to learn from.
    """
    B, C, H, W = encoded.shape
    forged = encoded.clone()
    mask   = torch.zeros(B, 1, H, W, device=encoded.device)

    # Shuffle batch so each image gets pixels from a *different* image
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
