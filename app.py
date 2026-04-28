"""
DualMark — Invisible Image Watermarking Demo
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

# ── Backend functions ─────────────────────────────────────────────────────────
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
    quality = "Invisible" if p >= 38 else "Borderline"

    info = (
        f"### Watermark Embedded\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| PSNR | `{p:.2f} dB` — {quality} |\n"
        f"| Caption hash | `{cap_hex}` |\n"
        f"| UserID hash | `{uid_hex}` |\n\n"
        f"> Save your Caption and User ID — you'll need them to verify later."
    )
    return out_pil, info


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
        icon    = "### Watermark VERIFIED"
        verdict = "Strong match — ownership confirmed"
    elif ba >= 0.70:
        icon    = "### Partial Match"
        verdict = "Weak match — image may have been compressed or resized"
    else:
        icon    = "### Watermark NOT Found"
        verdict = "No match — wrong Caption/UserID or watermark absent"

    info = (
        f"{icon}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Bit Accuracy | `{ba*100:.1f}%` |\n"
        f"| Verdict | {verdict} |\n\n"
        f"**Hash comparison:**\n\n"
        f"| | Expected | Decoded |\n|---|---|---|\n"
        f"| Caption | `{cap_hex}` | `{result['caption_hex']}` |\n"
        f"| UserID  | `{uid_hex}` | `{result['userid_hex']}` |"
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

    heatmap        = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
    heatmap[:,:,0] = (p * 255).clip(0, 255).astype(np.uint8)
    heatmap[:,:,2] = ((1 - p) * 255).clip(0, 255).astype(np.uint8)
    heatmap_pil    = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    if manip > 0.35:
        icon    = "### Likely FORGED"
        verdict = "High risk — significant manipulation detected"
    elif manip > 0.15:
        icon    = "### Suspicious"
        verdict = "Moderate risk — possible manipulation"
    else:
        icon    = "### Likely AUTHENTIC"
        verdict = "Low risk — image appears unmodified"

    info = (
        f"{icon}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Manipulation Score | `{manip*100:.1f}%` |\n"
        f"| Verdict | {verdict} |\n\n"
        f"> Heatmap: red = manipulated region, blue = authentic region"
    )
    return heatmap_pil, info


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}
.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 14px;
    padding: 26px 24px 18px;
    margin-bottom: 10px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.header-box h1 {
    color: #e8e8f0 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    margin: 0 0 5px 0 !important;
}
.header-box p {
    color: #9090b8 !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.small-img img {
    max-height: 200px !important;
    object-fit: contain !important;
    border-radius: 8px !important;
}
.out-md {
    background: #0f0f1e !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    font-size: 0.87rem !important;
    line-height: 1.65 !important;
    min-height: 80px;
}
.out-md code {
    background: #1e1e32 !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    color: #a5b4fc !important;
    font-size: 0.82rem !important;
}
.out-md table { width: 100%; border-collapse: collapse; margin: 6px 0; }
.out-md th, .out-md td {
    padding: 5px 10px;
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 0.84rem;
}
.run-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}
.footer-txt {
    text-align: center;
    color: #55557a;
    font-size: 0.76rem;
    margin-top: 10px;
    padding-bottom: 4px;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DualMark", css=css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div class="header-box">
        <h1>&#128274; DualMark</h1>
        <p>Invisible image watermarking &nbsp;&middot;&nbsp; Ownership verification &nbsp;&middot;&nbsp; Forgery detection</p>
    </div>
    """)

    with gr.Tabs():

        # ── Embed ─────────────────────────────────────────────────────────────
        with gr.TabItem("Embed"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    embed_img_in = gr.Image(type="pil", label="Original Image",
                                            elem_classes=["small-img"], height=200)
                    embed_caption = gr.Textbox(label="Caption",
                                               placeholder="e.g. Photo by Alice, Jan 2024")
                    embed_userid  = gr.Textbox(label="User ID",
                                               placeholder="e.g. alice_42")
                    embed_btn = gr.Button("Embed Watermark", variant="primary",
                                          elem_classes=["run-btn"])
                with gr.Column(scale=1, min_width=280):
                    embed_img_out = gr.Image(type="pil", label="Watermarked Image",
                                             elem_classes=["small-img"], height=200)
                    embed_info = gr.Markdown(elem_classes=["out-md"])

            embed_btn.click(embed_watermark,
                            inputs=[embed_img_in, embed_caption, embed_userid],
                            outputs=[embed_img_out, embed_info])

        # ── Verify ────────────────────────────────────────────────────────────
        with gr.TabItem("Verify"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    verify_img = gr.Image(type="pil", label="Watermarked Image",
                                          elem_classes=["small-img"], height=200)
                    verify_caption = gr.Textbox(label="Caption",
                                                placeholder="Same caption used during embedding")
                    verify_userid  = gr.Textbox(label="User ID",
                                                placeholder="Same user ID used during embedding")
                    verify_btn = gr.Button("Verify Ownership", variant="primary",
                                           elem_classes=["run-btn"])
                with gr.Column(scale=1, min_width=280):
                    verify_result = gr.Markdown(elem_classes=["out-md"])

            verify_btn.click(verify_watermark,
                             inputs=[verify_img, verify_caption, verify_userid],
                             outputs=[verify_result])

        # ── Forgery ───────────────────────────────────────────────────────────
        with gr.TabItem("Detect Forgery"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    detect_img = gr.Image(type="pil", label="Image to Inspect",
                                          elem_classes=["small-img"], height=200)
                    detect_btn = gr.Button("Detect Forgery", variant="primary",
                                           elem_classes=["run-btn"])
                with gr.Column(scale=1, min_width=280):
                    detect_heatmap = gr.Image(type="pil", label="Forgery Heatmap",
                                              elem_classes=["small-img"], height=200)
                    detect_result  = gr.Markdown(elem_classes=["out-md"])

            detect_btn.click(detect_forgery,
                             inputs=[detect_img],
                             outputs=[detect_heatmap, detect_result])

    gr.HTML('<div class="footer-txt">DualMark &nbsp;&middot;&nbsp; 128-bit dual-decoder invisible watermarking</div>')

if __name__ == "__main__":
    demo.launch()
