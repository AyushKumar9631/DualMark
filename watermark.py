"""
watermark.py  —  Hashing utilities.

  caption  →  96-bit hash  (first 96 bits of SHA-256)
  user_id  →  32-bit hash  (first 32 bits of SHA-256)
  combined →  128-bit watermark  [caption_bits | userid_bits]

The 128 bits are stored as a float tensor of ±1 values
(+1 = bit 1,  -1 = bit 0) which is the standard convention
for watermark training with MSE loss.
"""

import hashlib
import torch
import numpy as np


def _str_to_bits(text: str, n_bits: int) -> list[int]:
    """SHA-256 of text → first n_bits bits as list of 0/1."""
    digest = hashlib.sha256(text.encode('utf-8')).digest()   # 32 bytes = 256 bits
    bits = []
    for byte in digest:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
            if len(bits) == n_bits:
                return bits
    return bits[:n_bits]


def make_watermark(caption: str, user_id: str,
                   caption_bits: int = 96,
                   userid_bits:  int = 32) -> torch.Tensor:
    """
    Returns a (128,) float32 tensor of ±1 values.
    Layout: [caption_bits (96) | userid_bits (32)]
    """
    c_bits = _str_to_bits(caption, caption_bits)
    u_bits = _str_to_bits(user_id, userid_bits)
    bits   = c_bits + u_bits                          # 128 elements, each 0 or 1
    tensor = torch.tensor([2*b - 1 for b in bits], dtype=torch.float32)  # 0→-1, 1→+1
    return tensor                                     # shape (128,)


def decode_watermark(message_tensor: torch.Tensor,
                     caption_bits: int = 96,
                     userid_bits:  int = 32) -> dict:
    """
    message_tensor : (128,) float tensor (raw decoder output or ±1 values)
    Returns dict with:
      'caption_bits' : list of 96 ints (0/1)
      'userid_bits'  : list of 32 ints (0/1)
      'caption_hex'  : hex string of the 96-bit hash
      'userid_hex'   : hex string of the 32-bit hash
    """
    bits = (message_tensor > 0).int().tolist()
    c_bits = bits[:caption_bits]
    u_bits = bits[caption_bits: caption_bits + userid_bits]

    def bits_to_hex(b):
        # pad to multiple of 8
        padded = b + [0] * (-len(b) % 8)
        hexval = ''
        for i in range(0, len(padded), 8):
            byte = 0
            for j, bit in enumerate(padded[i:i+8]):
                byte |= (bit << (7 - j))
            hexval += f'{byte:02x}'
        return hexval

    return {
        'caption_bits': c_bits,
        'userid_bits':  u_bits,
        'caption_hex':  bits_to_hex(c_bits),
        'userid_hex':   bits_to_hex(u_bits),
    }


def bit_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of bits correctly decoded.  pred and target are ±1 tensors."""
    return ((pred > 0) == (target > 0)).float().mean().item()
