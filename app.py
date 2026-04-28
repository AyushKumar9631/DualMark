"""
DualMark — Invisible Image Watermarking Demo
Hugging Face Spaces · Gradio UI
"""

import os, sys, hashlib
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from network.model import build_models
from watermark import make_watermark, decode_watermark, bit_accuracy

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH       = "model/checkpoint.pth"   # <-- place your .pth file here
ENCODE_STRENGTH = 0.1                       # must match training
MSG_LEN         = 128
CHANNELS        = 32
IMG_SIZE        = 128
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM   = transforms.Normalize([0.5]*3, [0.5]*3)
DENORM = lambda t: ((t.clamp(-1, 1) + 1) / 2)

# ── Load models once at startup ───────────────────────────────────────────────
print(f"Loading models from {CKPT_PATH} on {DEVICE} ...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
enc, decR, decF = build_models(MSG_LEN, CHANNELS, ENCODE_STRENGTH, DEVICE)
enc.load_state_dict(ckpt["encoder"])
decR.load_state_dict(ckpt["decoderR"])
decF.load_state_dict(ckpt["decoderF"])
enc.eval(); decR.eval(); decF.eval()
print("Models ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_tensor(pil_img):
    """PIL RGB → (1,3,H,W) float tensor in [-1,1]."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t   = transforms.ToTensor()(img)
    return NORM(t).unsqueeze(0).to(DEVICE)

def tensor_to_pil(t):
    """(1,3,H,W) tensor in [-1,1] → PIL RGB."""
    arr = (DENORM(t).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def psnr(orig, wm):
    mse = ((orig - wm) ** 2).mean().item()
    return 10 * np.log10(4.0 / (mse + 1e-10))

def expected_hashes(caption, user_id):
    """Return the expected caption_hex and userid_hex for given inputs."""
    wm = make_watermark(caption, user_id)
    result = decode_watermark(wm)
    return result["caption_hex"], result["userid_hex"]

# ── Tab 1: Embed ──────────────────────────────────────────────────────────────
def embed_watermark(image, caption, user_id):
    if image is None:
        return None, "Please upload an image."
    if not caption.strip() or not user_id.strip():
        return None, "Caption and User ID are required."

    img_t = pil_to_tensor(image)
    wm    = make_watermark(caption.strip(), user_id.strip()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        wm_img = enc(img_t, wm)

    p = psnr(img_t, wm_img)
    out_pil = tensor_to_pil(wm_img)

    # Resize output back to original size for display
    out_pil = out_pil.resize(image.size, Image.BILINEAR)

    cap_hex, uid_hex = expected_hashes(caption.strip(), user_id.strip())

    info = (
        f"**PSNR:** {p:.2f} dB  ({'Good — invisible' if p >= 38 else 'Borderline — may be visible'})\n\n"
        f"**Caption hash (96-bit):** `{cap_hex}`\n\n"
        f"**UserID hash (32-bit):** `{uid_hex}`\n\n"
        f"**Watermark embedded successfully.** Keep your Caption and User ID to verify later."
    )
    return out_pil, info

# ── Tab 2: Verify ─────────────────────────────────────────────────────────────
def verify_watermark(image, caption, user_id):
    if image is None:
        return "Please upload a watermarked image."
    if not caption.strip() or not user_id.strip():
        return "Caption and User ID are required to verify."

    img_t  = pil_to_tensor(image)
    wm_gt  = make_watermark(caption.strip(), user_id.strip())

    with torch.no_grad():
        pred = decR(img_t)

    ba = bit_accuracy(pred.squeeze(0).cpu(), wm_gt)
    result = decode_watermark(pred.squeeze(0).cpu())
    cap_hex, uid_hex = expected_hashes(caption.strip(), user_id.strip())

    if ba >= 0.85:
        verdict = "MATCH — Watermark verified"
        icon    = "**Watermark VERIFIED**"
    elif ba >= 0.70:
        verdict = "PARTIAL — Weak match (image may be compressed or resized)"
        icon    = "**Partial Match**"
    else:
        verdict = "NO MATCH — Watermark not found or wrong Caption/UserID"
        icon    = "**Watermark NOT found**"

    info = (
        f"## {icon}\n\n"
        f"**Bit Accuracy:** {ba*100:.1f}%  (random = 50%, match threshold = 85%)\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"---\n"
        f"**Expected caption hash:** `{cap_hex}`\n\n"
        f"**Decoded caption hash:** `{result['caption_hex']}`\n\n"
        f"**Expected UserID hash:** `{uid_hex}`\n\n"
        f"**Decoded UserID hash:** `{result['userid_hex']}`"
    )
    return info

# ── Tab 3: Forgery Detection ──────────────────────────────────────────────────
def detect_forgery(image):
    if image is None:
        return None, "Please upload an image."

    img_t = pil_to_tensor(image)

    with torch.no_grad():
        prob_map = torch.sigmoid(decF(img_t))   # (1,1,H,W) in [0,1]

    manip = prob_map.mean().item()
    p     = prob_map.squeeze().cpu().numpy()

    # Heatmap: red = manipulated, blue = authentic
    heatmap = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
    heatmap[:, :, 0] = (p * 255).clip(0, 255).astype(np.uint8)
    heatmap[:, :, 2] = ((1 - p) * 255).clip(0, 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    if manip > 0.35:
        verdict = "HIGH RISK — Image appears significantly manipulated"
        icon    = "**Likely FORGED**"
    elif manip > 0.15:
        verdict = "MODERATE — Possible manipulation detected"
        icon    = "**Suspicious**"
    else:
        verdict = "LOW RISK — Image appears authentic"
        icon    = "**Likely AUTHENTIC**"

    info = (
        f"## {icon}\n\n"
        f"**Manipulation score:** {manip*100:.1f}%\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"*Heatmap: red = manipulated region, blue = authentic region*"
    )
    return heatmap_pil, info

# ── Build UI ──────────────────────────────────────────────────────────────────
css = """
.gr-button-primary { background: #1a1a2e !important; }
h1 { text-align: center; }
.subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
"""

with gr.Blocks(title="DualMark — Invisible Watermarking", css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("# DualMark — Invisible Image Watermarking")
    gr.Markdown(
        "<div class='subtitle'>Embed invisible watermarks into images · Verify ownership · Detect forgery</div>"
    )

    with gr.Tabs():

        # ── Tab 1: Embed ──────────────────────────────────────────────────────
        with gr.TabItem("Embed Watermark"):
            gr.Markdown("Upload an image, enter a caption and user ID — the model will invisibly embed a 128-bit watermark.")
            with gr.Row():
                with gr.Column():
                    embed_img_in  = gr.Image(type="pil", label="Original Image")
                    embed_caption = gr.Textbox(label="Caption", placeholder="e.g. Photo taken by Alice on 2024-01-01")
                    embed_userid  = gr.Textbox(label="User ID", placeholder="e.g. alice_42")
                    embed_btn     = gr.Button("Embed Watermark", variant="primary")
                with gr.Column():
                    embed_img_out = gr.Image(type="pil", label="Watermarked Image")
                    embed_info    = gr.Markdown()
            embed_btn.click(embed_watermark,
                            inputs=[embed_img_in, embed_caption, embed_userid],
                            outputs=[embed_img_out, embed_info])

        # ── Tab 2: Verify ─────────────────────────────────────────────────────
        with gr.TabItem("Verify Watermark"):
            gr.Markdown("Upload a watermarked image and enter the original Caption and User ID to verify ownership.")
            with gr.Row():
                with gr.Column():
                    verify_img     = gr.Image(type="pil", label="Watermarked Image")
                    verify_caption = gr.Textbox(label="Caption", placeholder="Same caption used during embedding")
                    verify_userid  = gr.Textbox(label="User ID", placeholder="Same user ID used during embedding")
                    verify_btn     = gr.Button("Verify", variant="primary")
                with gr.Column():
                    verify_result = gr.Markdown()
            verify_btn.click(verify_watermark,
                             inputs=[verify_img, verify_caption, verify_userid],
                             outputs=[verify_result])

        # ── Tab 3: Forgery Detection ──────────────────────────────────────────
        with gr.TabItem("Detect Forgery"):
            gr.Markdown("Upload a watermarked image to check if it has been tampered with. The heatmap shows which regions appear manipulated.")
            with gr.Row():
                with gr.Column():
                    detect_img = gr.Image(type="pil", label="Image to Inspect")
                    detect_btn = gr.Button("Detect Forgery", variant="primary")
                with gr.Column():
                    detect_heatmap = gr.Image(type="pil", label="Forgery Heatmap  (red = manipulated)")
                    detect_result  = gr.Markdown()
            detect_btn.click(detect_forgery,
                             inputs=[detect_img],
                             outputs=[detect_heatmap, detect_result])

    gr.Markdown(
        "<br><center><sub>DualMark — dual-decoder invisible watermarking. "
        "Encoder embeds a 128-bit hash. DecoderR recovers it. DecoderF detects per-pixel manipulation.</sub></center>"
    )

if __name__ == "__main__":
    demo.launch()
