"""
DualMark — Invisible Image Watermarking
Hugging Face Spaces · Gradio UI
"""

import os, sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from network.model import build_models
from watermark import make_watermark, decode_watermark, bit_accuracy

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH       = "model/checkpoint.pth"
ENCODE_STRENGTH = 0.1
MSG_LEN         = 128
CHANNELS        = 32
IMG_SIZE        = 128
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM   = transforms.Normalize([0.5]*3, [0.5]*3)
DENORM = lambda t: ((t.clamp(-1, 1) + 1) / 2)

# ── Load models ───────────────────────────────────────────────────────────────
print(f"Loading models on {DEVICE} ...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
enc, decR, decF = build_models(MSG_LEN, CHANNELS, ENCODE_STRENGTH, DEVICE)
enc.load_state_dict(ckpt["encoder"])
decR.load_state_dict(ckpt["decoderR"])
decF.load_state_dict(ckpt["decoderF"])
enc.eval(); decR.eval(); decF.eval()
print("Models ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_tensor(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t   = transforms.ToTensor()(img)
    return NORM(t).unsqueeze(0).to(DEVICE)

def tensor_to_pil(t):
    arr = (DENORM(t).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def psnr(orig, wm):
    mse = ((orig - wm) ** 2).mean().item()
    return 10 * np.log10(4.0 / (mse + 1e-10))

def expected_hashes(caption, user_id):
    wm = make_watermark(caption, user_id)
    result = decode_watermark(wm)
    return result["caption_hex"], result["userid_hex"]

# ── Backend ───────────────────────────────────────────────────────────────────
def embed_watermark(image, caption, user_id):
    if image is None:
        return None, "Please upload an image."
    if not caption.strip() or not user_id.strip():
        return None, "Caption and User ID are required."

    img_t = pil_to_tensor(image)
    wm    = make_watermark(caption.strip(), user_id.strip()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        wm_img = enc(img_t, wm)

    p       = psnr(img_t, wm_img)
    out_pil = tensor_to_pil(wm_img).resize(image.size, Image.BILINEAR)
    cap_hex, uid_hex = expected_hashes(caption.strip(), user_id.strip())
    quality = "Invisible (Good)" if p >= 38 else "Borderline"

    info = (
        f"**Status:** Watermark embedded successfully\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| PSNR | `{p:.2f} dB` — {quality} |\n"
        f"| Caption Hash (96-bit) | `{cap_hex}` |\n"
        f"| UserID Hash (32-bit) | `{uid_hex}` |\n\n"
        f"Retain your Caption and User ID. They are required for ownership verification."
    )
    return out_pil, info


def verify_watermark(image, caption, user_id):
    if image is None:
        return "Please upload a watermarked image."
    if not caption.strip() or not user_id.strip():
        return "Caption and User ID are required to verify."

    img_t  = pil_to_tensor(image)
    wm_gt  = make_watermark(caption.strip(), user_id.strip())  # (128,) ±1 tensor

    with torch.no_grad():
        pred = decR(img_t)

    pred_cpu = pred.squeeze(0).cpu()
    result   = decode_watermark(pred_cpu)
    cap_hex, uid_hex = expected_hashes(caption.strip(), user_id.strip())

    # ── Per-segment bit accuracy (caption = bits 0:96, userid = bits 96:128) ──
    ba_cap = bit_accuracy(pred_cpu[:96],  wm_gt[:96])
    ba_uid = bit_accuracy(pred_cpu[96:],  wm_gt[96:])

    # ── Owner Identification (UserID hash, 32 bits) ──
    if ba_uid >= 0.85:
        uid_status  = "✅ OWNER CONFIRMED"
        uid_verdict = "UserID hash matches. Ownership is verified."
    elif ba_uid >= 0.70:
        uid_status  = "⚠️ UNCERTAIN"
        uid_verdict = "Weak match. Image may have been compressed or resized."
    else:
        uid_status  = "❌ OWNER MISMATCH"
        uid_verdict = "UserID hash does not match. This User ID did not watermark this image."

    # ── Caption Integrity (caption hash, 96 bits) ──
    if ba_cap >= 0.85:
        cap_status  = "✅ CAPTION INTACT"
        cap_verdict = "Caption hash matches. The caption has not been altered."
    elif ba_cap >= 0.70:
        cap_status  = "⚠️ UNCERTAIN"
        cap_verdict = "Weak match. Minor degradation may have affected the caption bits."
    else:
        cap_status  = "❌ CAPTION MISMATCH"
        cap_verdict = "Caption hash does not match. The caption may have been changed or is incorrect."

    info = (
        f"---\n"
        f"### 🔐 Owner Identification\n\n"
        f"**Result:** {uid_status}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| UserID Bit Accuracy | `{ba_uid*100:.1f}%` (random baseline = 50%) |\n"
        f"| Threshold | `85%` |\n"
        f"| Verdict | {uid_verdict} |\n\n"
        f"| Field | Expected | Decoded |\n|---|---|---|\n"
        f"| UserID Hash | `{uid_hex}` | `{result['userid_hex']}` |\n\n"
        f"---\n"
        f"### 📝 Caption Integrity\n\n"
        f"**Result:** {cap_status}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Caption Bit Accuracy | `{ba_cap*100:.1f}%` (random baseline = 50%) |\n"
        f"| Threshold | `85%` |\n"
        f"| Verdict | {cap_verdict} |\n\n"
        f"| Field | Expected | Decoded |\n|---|---|---|\n"
        f"| Caption Hash | `{cap_hex}` | `{result['caption_hex']}` |"
    )
    return info


def detect_forgery(image):
    if image is None:
        return None, "Please upload an image."

    img_t = pil_to_tensor(image)

    with torch.no_grad():
        prob_map = torch.sigmoid(decF(img_t))

    manip = prob_map.mean().item()
    p     = prob_map.squeeze().cpu().numpy()

    # ── Spatially-varying threshold heatmap ───────────────────────────────────
    # The model's center bias means the face region is ALWAYS elevated.
    # Previous attempts (Gaussian subtraction, modal baseline) all failed
    # because the face mound is nearly flat and too wide to subtract cleanly.
    #
    # User's insight: raise the threshold specifically in the center region.
    # "Center" is defined from the probability map itself — the high-probability
    # blob IS the face bias region.
    #
    #   Center/face region  → require p > 0.90 to appear red
    #                         (filters out the 0.55–0.75 face bias noise;
    #                          only genuine high-confidence forgery triggers it)
    #   Background region   → require p > 0.60
    #                         (background pixels rarely exceed 0.35,
    #                          so spliced-in patches stand out immediately)
    from scipy.ndimage import gaussian_filter as _gf
    p_sm       = _gf(p, sigma=3)          # light smooth to get a clean blob
    center_mask = p_sm > 0.50             # True = face/center bias region

    thresh = np.where(center_mask, 0.95, 0.60)   # spatially adaptive bar
    p_above = np.clip(p - thresh, 0, None)

    # SCALE=0.10 → pixel at p=0.90 starts showing; p=1.00 is fully red
    SCALE  = 0.10
    p_norm = np.clip(p_above / SCALE, 0, 1)

    heatmap        = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
    heatmap[:,:,0] = (p_norm * 255).clip(0, 255).astype(np.uint8)
    heatmap[:,:,2] = ((1 - p_norm) * 255).clip(0, 255).astype(np.uint8)
    heatmap_pil    = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    if manip > 0.35:
        status  = "HIGH RISK — Likely Forged"
        verdict = "Significant pixel-level manipulation detected."
    elif manip > 0.15:
        status  = "MODERATE RISK — Suspicious"
        verdict = "Possible manipulation. Manual inspection recommended."
    else:
        status  = "LOW RISK — Likely Authentic"
        verdict = "No significant manipulation detected."

    info = (
        f"**Integrity Result:** {status}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Manipulation Score | `{manip*100:.1f}%` |\n"
        f"| Verdict | {verdict} |\n\n"
        f"The heatmap indicates per-pixel manipulation probability. "
        f"Red regions indicate likely tampered areas. Blue regions indicate authentic areas."
    )
    return heatmap_pil, info


def forge_image(wm_image, orig_image, num_patches, patch_size_pct, anchor):
    """
    Splice `num_patches` rectangles from orig_image (or black) into wm_image,
    then run the forgery detector on the result.
    anchor: one of 9 labels like 'Top-Left', 'Center', etc.
    """
    import random

    if wm_image is None:
        return None, None, "Please upload a watermarked image."

    wm   = wm_image.convert("RGB")
    W, H = wm.size
    out  = wm.copy()

    # ── Anchor → (col, row) in 0-2 grid ──────────────────────────────────────
    anchor_map = {
        "Top-Left":      (0, 0), "Top-Center":    (1, 0), "Top-Right":     (2, 0),
        "Middle-Left":   (0, 1), "Center":         (1, 1), "Middle-Right":  (2, 1),
        "Bottom-Left":   (0, 2), "Bottom-Center": (1, 2), "Bottom-Right":  (2, 2),
    }
    col, row = anchor_map.get(anchor, (1, 1))

    # The placement zone is the chosen third of the image
    zone_x0 = (col * W) // 3
    zone_y0 = (row * H) // 3
    zone_x1 = ((col + 1) * W) // 3
    zone_y1 = ((row + 1) * H) // 3

    pw = max(10, int(W * patch_size_pct / 100))
    ph = max(10, int(H * patch_size_pct / 100))

    patch_infos = []
    for _ in range(int(num_patches)):
        # Top-left of the patch, clipped so patch stays inside the zone
        px = random.randint(zone_x0, max(zone_x0, zone_x1 - pw))
        py = random.randint(zone_y0, max(zone_y0, zone_y1 - ph))
        patch_infos.append((px, py, pw, ph))

    import PIL.ImageDraw as ImageDraw
    for (px, py, pw, ph) in patch_infos:
        if orig_image is not None:
            src = orig_image.convert("RGB").resize(wm.size, Image.BILINEAR)
            patch = src.crop((px, py, px + pw, py + ph))
        else:
            patch = Image.new("RGB", (pw, ph), (0, 0, 0))
        out.paste(patch, (px, py))

    # ── Run forgery detector on the forged image ──────────────────────────────
    heatmap_pil, detect_info = detect_forgery(out)

    patch_desc = "  ".join(
        f"Patch {i+1}: ({px},{py}) {pw}×{ph}px"
        for i, (px, py, pw, ph) in enumerate(patch_infos)
    )
    summary = (
        f"**Forge Summary**\n\n"
        f"| Setting | Value |\n|---|---|\n"
        f"| Patches applied | `{int(num_patches)}` |\n"
        f"| Patch size | `{pw}×{ph} px` ({patch_size_pct:.0f}% of image) |\n"
        f"| Placement zone | `{anchor}` |\n"
        f"| Fill source | `{'Original image region' if orig_image else 'Black (no original provided)'}` |\n\n"
        f"**Patch coordinates**\n\n"
        f"{patch_desc}\n\n"
        f"---\n\n"
        f"{detect_info}"
    )
    return out, heatmap_pil, summary


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #080b14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #c9cfe0 !important;
}

.gradio-container {
    max-width: 940px !important;
    margin: 0 auto !important;
    padding: 0 16px 40px !important;
}

/* ── Header ── */
.dm-header {
    border-bottom: 1px solid #1e2740;
    padding: 32px 0 24px;
    margin-bottom: 24px;
}
.dm-header .dm-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6c7faa;
    background: #0f1626;
    border: 1px solid #1e2740;
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 14px;
}
.dm-header h1 {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #e4e9f5 !important;
    letter-spacing: -0.8px !important;
    margin: 0 0 10px 0 !important;
    line-height: 1.1 !important;
}
.dm-header .dm-sub {
    font-size: 0.92rem;
    color: #6c7faa;
    line-height: 1.6;
    max-width: 620px;
}

/* ── Model info panel ── */
.dm-info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.dm-info-card {
    background: #0d1120;
    border: 1px solid #1e2740;
    border-radius: 8px;
    padding: 14px 16px;
}
.dm-info-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4a5a7a;
    margin-bottom: 6px;
}
.dm-info-card .value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #a8b8d8;
    font-family: 'JetBrains Mono', monospace;
}
.dm-info-card .sub {
    font-size: 0.72rem;
    color: #3a4a65;
    margin-top: 3px;
}

/* ── Architecture section ── */
.dm-arch {
    background: #0d1120;
    border: 1px solid #1e2740;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 24px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
}
.dm-arch-block {
    padding: 0 16px;
    border-right: 1px solid #1e2740;
}
.dm-arch-block:first-child { padding-left: 0; }
.dm-arch-block:last-child  { border-right: none; }
.dm-arch-block .arch-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3d6099;
    margin-bottom: 8px;
}
.dm-arch-block .arch-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #c0ccdf;
    margin-bottom: 4px;
}
.dm-arch-block .arch-desc {
    font-size: 0.78rem;
    color: #4a5a7a;
    line-height: 1.55;
}

/* ── Section divider ── */
.dm-section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #3d5080;
    border-bottom: 1px solid #1e2740;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Tabs ── */
.tab-nav {
    border-bottom: 1px solid #1e2740 !important;
    margin-bottom: 0 !important;
}
.tab-nav button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #4a5a7a !important;
    padding: 10px 22px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.2s !important;
}
.tab-nav button.selected {
    color: #8ba3cc !important;
    border-bottom: 2px solid #3d6099 !important;
}

/* ── Tab content wrapper ── */
.tabitem {
    background: #0d1120 !important;
    border: 1px solid #1e2740 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 20px !important;
}

/* ── Tab description ── */
.tab-desc {
    font-size: 0.82rem;
    color: #4a5a7a;
    margin-bottom: 18px;
    padding: 10px 14px;
    background: #080b14;
    border-left: 3px solid #1e3060;
    border-radius: 0 6px 6px 0;
}

/* ── Images ── */
.small-img > div {
    border: 1px solid #1e2740 !important;
    border-radius: 8px !important;
    background: #080b14 !important;
    overflow: hidden !important;
}
.small-img img {
    max-height: 200px !important;
    object-fit: contain !important;
}
.small-img .upload-container {
    background: #080b14 !important;
}

/* ── Inputs ── */
label > span {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #4a5a7a !important;
}
input[type="text"], textarea {
    background: #080b14 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 6px !important;
    color: #c0ccdf !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #2d4a7a !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(45,74,122,0.25) !important;
}

/* ── Button ── */
button.primary, .gr-button-primary {
    background: #1a3060 !important;
    border: 1px solid #2d4a7a !important;
    border-radius: 6px !important;
    color: #a8c0e8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 10px 22px !important;
    width: 100% !important;
    transition: background 0.2s, border-color 0.2s !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #1f3a75 !important;
    border-color: #3d5a8a !important;
}

/* ── Output markdown ── */
.out-md {
    background: #080b14 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    color: #8a9ab8 !important;
    min-height: 100px;
}
.out-md strong { color: #a8b8d8 !important; }
.out-md table  { width: 100%; border-collapse: collapse; margin: 10px 0; }
.out-md th {
    background: #0f1626;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #4a5a7a;
    padding: 7px 12px;
    border: 1px solid #1e2740;
    text-align: left;
}
.out-md td {
    padding: 7px 12px;
    border: 1px solid #1e2740;
    font-size: 0.83rem;
    color: #8a9ab8;
}
.out-md td:first-child, .out-md th:first-child {
    white-space: nowrap;
    width: 1%;
}
.out-md code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #0f1626 !important;
    color: #6a9fd8 !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #1e2740 !important;
}

/* ── Forge position grid ── */
.position-grid > .wrap {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 5px !important;
    margin-top: 6px !important;
}
.position-grid label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #0d1120 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 6px !important;
    padding: 7px 4px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #4a5a7a !important;
    cursor: pointer !important;
    transition: border-color 0.15s, color 0.15s !important;
    text-align: center !important;
}
.position-grid label:has(input:checked) {
    border-color: #3d6099 !important;
    color: #8ba3cc !important;
    background: #0f1a2e !important;
}
.position-grid input[type="radio"] { display: none !important; }

/* ── Footer ── */
.dm-footer {
    text-align: center;
    color: #2a3550;
    font-size: 0.74rem;
    letter-spacing: 0.3px;
    margin-top: 20px;
    padding: 12px 0;
    border-top: 1px solid #0f1626;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DualMark — Invisible Watermarking", css=css, theme=gr.themes.Base()) as demo:

    # Header
    gr.HTML("""
    <div class="dm-header">
        <div class="dm-badge">Research Prototype</div>
        <h1>DualMark</h1>
        <div class="dm-sub">
            A dual-decoder invisible watermarking system for digital images.
            Embeds a 128-bit hash imperceptibly into image pixels using a learned encoder,
            then recovers ownership metadata or detects pixel-level forgery through two independent decoders.
        </div>
    </div>
    """)

    # Model stats
    gr.HTML("""
    <div class="dm-info-grid">
        <div class="dm-info-card">
            <div class="label">Watermark Length</div>
            <div class="value">128 bits</div>
            <div class="sub">96-bit caption + 32-bit user ID</div>
        </div>
        <div class="dm-info-card">
            <div class="label">Encode Strength</div>
            <div class="value">0.1</div>
            <div class="sub">Max pixel delta: ~12.75/255</div>
        </div>
        <div class="dm-info-card">
            <div class="label">Input Resolution</div>
            <div class="value">128 x 128</div>
            <div class="sub">Images resized before inference</div>
        </div>
        <div class="dm-info-card">
            <div class="label">Training Data</div>
            <div class="value">CelebA-HQ</div>
            <div class="sub">40 epochs · batch size 8</div>
        </div>
    </div>
    """)

    # Architecture
    gr.HTML("""
    <div class="dm-arch">
        <div class="dm-arch-block">
            <div class="arch-title">Encoder</div>
            <div class="arch-name">Conv-UNet</div>
            <div class="arch-desc">
                Takes a cover image and a 128-bit message vector.
                Outputs a watermarked image via residual delta:
                image + strength * tanh(delta).
                Trained with MSE image quality loss.
            </div>
        </div>
        <div class="dm-arch-block">
            <div class="arch-title">Decoder R — Recovery</div>
            <div class="arch-name">Conv Classifier</div>
            <div class="arch-desc">
                Reads back the 128-bit watermark from a watermarked image.
                Robust to Gaussian blur, noise, brightness shifts,
                resize, salt-and-pepper, and pixel dropout.
            </div>
        </div>
        <div class="dm-arch-block">
            <div class="arch-title">Decoder F — Forgery</div>
            <div class="arch-name">Conv Segmenter</div>
            <div class="arch-desc">
                Produces a per-pixel manipulation probability map.
                Trained on synthetic splice attacks.
                Outputs a heatmap indicating tampered regions.
            </div>
        </div>
    </div>
    """)

    gr.HTML('<div class="dm-section-label">Interactive Demo</div>')

    with gr.Tabs():

        # ── Embed ─────────────────────────────────────────────────────────────
        with gr.TabItem("Embed Watermark"):
            gr.HTML('<div class="tab-desc">Upload an image and provide a caption and user identifier. The encoder will invisibly embed a 128-bit hash derived from these inputs into the image pixels.</div>')
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    embed_img_in  = gr.Image(type="pil", label="Input Image",
                                             elem_classes=["small-img"], height=200)
                    embed_caption = gr.Textbox(label="Caption",
                                               placeholder="e.g. Portrait session, Studio A, 2024-03-15")
                    embed_userid  = gr.Textbox(label="User ID",
                                               placeholder="e.g. photographer_id_001")
                    embed_btn     = gr.Button("Run Encoder", variant="primary")
                with gr.Column(scale=1, min_width=260):
                    embed_img_out = gr.Image(type="pil", label="Watermarked Output",
                                             elem_classes=["small-img"], height=200)
                    embed_info    = gr.Markdown(elem_classes=["out-md"])

            embed_btn.click(embed_watermark,
                            inputs=[embed_img_in, embed_caption, embed_userid],
                            outputs=[embed_img_out, embed_info])

        # ── Verify ────────────────────────────────────────────────────────────
        with gr.TabItem("Ownership & Caption Integrity"):
            gr.HTML('<div class="tab-desc">Upload a watermarked image and provide the original caption and user ID. Decoder R recovers the embedded 128-bit message and runs two independent checks: the <strong>32-bit UserID hash</strong> identifies the owner, and the <strong>96-bit Caption hash</strong> verifies that the caption has not been altered.</div>')
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    verify_img     = gr.Image(type="pil", label="Watermarked Image",
                                              elem_classes=["small-img"], height=200)
                    verify_caption = gr.Textbox(label="Caption",
                                                placeholder="Original caption used during embedding")
                    verify_userid  = gr.Textbox(label="User ID",
                                                placeholder="Original user ID used during embedding")
                    verify_btn     = gr.Button("Run Decoder R", variant="primary")
                with gr.Column(scale=1, min_width=260):
                    verify_result  = gr.Markdown(elem_classes=["out-md"])

            verify_btn.click(verify_watermark,
                             inputs=[verify_img, verify_caption, verify_userid],
                             outputs=[verify_result])

        # ── Forgery ───────────────────────────────────────────────────────────
        with gr.TabItem("Detect Forgery"):
            gr.HTML('<div class="tab-desc">Upload a watermarked image to inspect its pixel-level integrity. Decoder F produces a spatial probability map indicating regions that have been tampered with or spliced.</div>')
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    detect_img = gr.Image(type="pil", label="Image to Inspect",
                                          elem_classes=["small-img"], height=200)
                    detect_btn = gr.Button("Run Decoder F", variant="primary")
                with gr.Column(scale=1, min_width=260):
                    detect_heatmap = gr.Image(type="pil", label="Manipulation Heatmap",
                                              elem_classes=["small-img"], height=200)
                    detect_result  = gr.Markdown(elem_classes=["out-md"])

            detect_btn.click(detect_forgery,
                             inputs=[detect_img],
                             outputs=[detect_heatmap, detect_result])

        # ── Forge ─────────────────────────────────────────────────────────────
        with gr.TabItem("Forge Image"):
            gr.HTML('<div class="tab-desc">Simulate a splice attack by replacing rectangular regions of a watermarked image with areas from an original image (or black patches). The forgery detector runs automatically on the result so you can see how well it catches the tampering.</div>')
            with gr.Row():
                # ── Left column: inputs ───────────────────────────────────────
                with gr.Column(scale=1, min_width=260):
                    forge_wm_img   = gr.Image(type="pil", label="Watermarked Image (required)",
                                              elem_classes=["small-img"], height=180)
                    forge_orig_img = gr.Image(type="pil", label="Original Image (optional — leave blank for black patches)",
                                              elem_classes=["small-img"], height=180)

                    forge_num = gr.Slider(minimum=1, maximum=12, step=1, value=3,
                                          label="Number of Patches")
                    forge_size = gr.Slider(minimum=5, maximum=60, step=1, value=20,
                                           label="Patch Size (% of image dimension)")

                    gr.HTML('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#4a5a7a;margin-top:10px;margin-bottom:4px;">Placement Zone</div>')
                    forge_anchor = gr.Radio(
                        choices=[
                            "Top-Left",    "Top-Center",    "Top-Right",
                            "Middle-Left", "Center",        "Middle-Right",
                            "Bottom-Left", "Bottom-Center", "Bottom-Right",
                        ],
                        value="Center",
                        label="",
                        elem_classes=["position-grid"],
                    )
                    forge_btn = gr.Button("Generate Forged Image", variant="primary")

                # ── Right column: outputs ─────────────────────────────────────
                with gr.Column(scale=1, min_width=260):
                    forge_out_img  = gr.Image(type="pil", label="Forged Image",
                                              elem_classes=["small-img"], height=180)
                    forge_heatmap  = gr.Image(type="pil", label="Forgery Heatmap",
                                              elem_classes=["small-img"], height=180)
                    forge_result   = gr.Markdown(elem_classes=["out-md"])

            forge_btn.click(forge_image,
                            inputs=[forge_wm_img, forge_orig_img,
                                    forge_num, forge_size, forge_anchor],
                            outputs=[forge_out_img, forge_heatmap, forge_result])

    gr.HTML("""
    <div class="dm-footer">
        DualMark &nbsp;|&nbsp; Dual-Decoder Invisible Watermarking &nbsp;|&nbsp;
        Encoder + Decoder R + Decoder F &nbsp;|&nbsp; Trained on CelebA-HQ &nbsp;|&nbsp; Research Prototype
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
