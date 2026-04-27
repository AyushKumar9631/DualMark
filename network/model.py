"""
network/model.py  —  All three lightweight networks.

  Encoder   : tiny 3-level U-Net, embeds 128-bit watermark as a bounded
              residual so the image looks unchanged to humans.
  DecoderR  : small CNN, recovers 128-bit message from distorted image.
  DecoderF  : tiny 3-level U-Net, outputs per-pixel manipulation map.

Total parameters: ~3-4 M  (original was ~50 M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared blocks ─────────────────────────────────────────────────────────────

def cbr(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(cbr(in_ch, out_ch, stride=2), cbr(out_ch, out_ch))
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + cat(skip) + conv.  in_ch = upsample_ch + skip_ch."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(cbr(in_ch, out_ch), cbr(out_ch, out_ch))
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        return self.block(torch.cat([x, skip], dim=1))


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Embeds a 128-bit message into an image as a small residual.
    Message is expanded to a spatial feature map and concatenated
    at each decoder level (HiDDeN-style injection).

    Output = clamp(image + strength * tanh(delta), -1, 1)
    strength=0.3 means the maximum per-channel delta is 0.3 in [-1,1] space,
    which is visually imperceptible on most images.
    """
    def __init__(self, msg_len=128, ch=32, strength=0.3):
        super().__init__()
        self.strength = strength
        self.msg_len  = msg_len
        C = ch

        # encoder (down path)
        self.e0 = cbr(3,     C)       # H   → (B, C,   H,   W)
        self.e1 = DownBlock(C,   C*2)  # H/2 → (B, 2C,  H/2, W/2)
        self.e2 = DownBlock(C*2, C*4)  # H/4 → (B, 4C,  H/4, W/4)
        self.e3 = DownBlock(C*4, C*8)  # H/8 → (B, 8C,  H/8, W/8)  bottleneck

        # message → spatial feature (broadcast to each spatial level)
        self.msg_fc = nn.Linear(msg_len, C*8)

        # decoder (up path): input = upsampled + skip + msg_broadcast
        # at each level, channels = prev_up_ch + skip_ch + C*8
        self.d2 = cbr(C*8 + C*4 + C*8, C*4)   # H/4
        self.d1 = cbr(C*4 + C*2 + C*8, C*2)   # H/2
        self.d0 = cbr(C*2 + C   + C*8, C)     # H

        self.out = nn.Conv2d(C, 3, 1)

    def _broadcast(self, msg_feat, h, w):
        return msg_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    def forward(self, image, message):
        # message embedding
        mf = self.msg_fc(message)                     # (B, C*8)

        # encoder
        s0 = self.e0(image)                           # (B, C,   H,   W)
        s1 = self.e1(s0)                              # (B, 2C,  H/2, W/2)
        s2 = self.e2(s1)                              # (B, 4C,  H/4, W/4)
        x  = self.e3(s2)                              # (B, 8C,  H/8, W/8)

        # decoder level 2  →  H/4
        x  = F.interpolate(x, size=s2.shape[2:], mode='nearest')
        m  = self._broadcast(mf, s2.shape[2], s2.shape[3])
        x  = self.d2(torch.cat([x, s2, m], dim=1))   # (B, 4C, H/4, W/4)

        # decoder level 1  →  H/2
        x  = F.interpolate(x, size=s1.shape[2:], mode='nearest')
        m  = self._broadcast(mf, s1.shape[2], s1.shape[3])
        x  = self.d1(torch.cat([x, s1, m], dim=1))   # (B, 2C, H/2, W/2)

        # decoder level 0  →  H
        x  = F.interpolate(x, size=s0.shape[2:], mode='nearest')
        m  = self._broadcast(mf, s0.shape[2], s0.shape[3])
        x  = self.d0(torch.cat([x, s0, m], dim=1))   # (B,  C, H,   W)

        delta = torch.tanh(self.out(x))               # (B, 3, H, W)  ∈ [-1,1]
        return torch.clamp(image + self.strength * delta, -1, 1)


# ── DecoderR ──────────────────────────────────────────────────────────────────

class DecoderR(nn.Module):
    """
    Recovers the 128-bit watermark from a (possibly distorted) image.
    Simple CNN encoder → global-average-pool → linear projection.
    """
    def __init__(self, msg_len=128, ch=32):
        super().__init__()
        C = ch
        self.net = nn.Sequential(
            cbr(3,   C),
            DownBlock(C,   C*2),
            DownBlock(C*2, C*4),
            DownBlock(C*4, C*8),
            DownBlock(C*8, C*8),
            nn.AdaptiveAvgPool2d(1),   # (B, C*8, 1, 1)
        )
        self.fc = nn.Linear(C*8, msg_len)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))   # (B, msg_len)


# ── DecoderF ──────────────────────────────────────────────────────────────────

class DecoderF(nn.Module):
    """
    Per-pixel manipulation detector.
    Output: (B, 1, H, W) sigmoid map.  1 = manipulated, 0 = authentic.
    """
    def __init__(self, ch=32):
        super().__init__()
        C = ch
        self.e0 = cbr(3,   C)
        self.e1 = DownBlock(C,   C*2)
        self.e2 = DownBlock(C*2, C*4)
        self.e3 = DownBlock(C*4, C*8)   # bottleneck

        self.d2 = UpBlock(C*8 + C*4,  C*4)
        self.d1 = UpBlock(C*4 + C*2,  C*2)
        self.d0 = UpBlock(C*2 + C,    C)

        self.out = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        b  = self.e3(s2)

        u = self.d2(b,  s2)
        u = self.d1(u,  s1)
        u = self.d0(u,  s0)
        return self.out(u)   # (B, 1, H, W) — raw logits for BCEWithLogitsLoss


# ── convenience ───────────────────────────────────────────────────────────────

def build_models(msg_len=128, ch=32, strength=0.3, device='cpu'):
    device = torch.device(device) if isinstance(device, str) else device
    enc  = Encoder(msg_len, ch, strength).to(device)
    decR = DecoderR(msg_len, ch).to(device)
    decF = DecoderF(ch).to(device)
    total = sum(p.numel() for m in [enc, decR, decF] for p in m.parameters())
    print(f"Encoder  : {sum(p.numel() for p in enc.parameters()):,} params")
    print(f"DecoderR : {sum(p.numel() for p in decR.parameters()):,} params")
    print(f"DecoderF : {sum(p.numel() for p in decF.parameters()):,} params")
    print(f"TOTAL    : {total:,} params")
    return enc, decR, decF
