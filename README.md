---
title: DualMark Invisible Watermarking
emoji: 🔏
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# DualMark — Invisible Image Watermarking

A dual-decoder invisible watermarking system for face images.

## Features
- **Embed** — invisibly encode a 128-bit watermark (caption + user ID hash) into any image
- **Verify** — confirm ownership by checking bit accuracy against original caption/user ID
- **Detect Forgery** — per-pixel manipulation heatmap to spot spliced or edited regions

## Model
Place your trained checkpoint at `model/checkpoint.pth` before deploying.
