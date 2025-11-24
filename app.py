import os
import requests
import gradio as gr
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

print("🚀 Starting app…")

MODEL_REPO = "Satvi/tiny_t5"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------
# RELIABLE ONNX DOWNLOADER
# ---------------------------------------------------------
def force_download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        print(f"✔ Using cached {os.path.basename(dest)}")
        return

    print(f"⬇ Downloading {url} ...")
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"❌ Download failed ({r.status_code}) → {url}")

    with open(dest, "wb") as f:
        f.write(r.content)

    print(f"✔ Saved: {dest}")


def download_models():
    encoder_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/encoder.onnx"
    decoder_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/decoder.onnx"

    force_download(encoder_url, os.path.join(MODEL_DIR, "encoder.onnx"))
    force_download(decoder_url, os.path.join(MODEL_DIR, "decoder.onnx"))


# ---------------------------------------------------------
# LOAD TOKENIZER + MODELS
# ---------------------------------------------------------
print("🔧 Initializing models…")
download_models()

print("🔹 Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    use_fast=False,
    trust_remote_code=True
)

PAD = tokenizer.pad_token_id or 0
EOS = tokenizer.eos_token_id


# ---------------------------------------------------------
# ONNX SESSION WITH FIX FOR SHAPE MISMATCH
# ---------------------------------------------------------
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 🔥 FIX: Disable memory pattern reuse → prevents MatMul shape mismatch
sess_opts.enable_mem_pattern = False
sess_opts.enable_cpu_mem_arena = False
sess_opts.log_severity_level = 2

enc_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "encoder.onnx"),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"]
)
dec_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "decoder.onnx"),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"]
)

print("✔ ONNX sessions loaded safely")


# ---------------------------------------------------------
# CLEANING
# ---------------------------------------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()


# ---------------------------------------------------------
# FIXED TINY T5 ONNX GENERATOR
# ---------------------------------------------------------
def tiny_generate(text, max_len=120):
    prompt = "summarize: " + text

    # 🔥 Safe tokenizer length for tiny_t5
    tokens = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=256
    )

    input_ids = tokens["input_ids"].astype(np.int64)

    # 1) Encoder
    enc_out = enc_sess.run(None, {"input_ids": input_ids})[0]

    # 2) Decoder
    dec_ids = np.array([[PAD]], dtype=np.int64)
    generated = []

    for _ in range(max_len):
        logits = dec_sess.run(
            None,
            {
                "decoder_input_ids": dec_ids,      # 🔥 FIXED
                "encoder_hidden_states": enc_out
            }
        )[0]

        next_tok = int(np.argmax(logits[:, -1, :]))

        if next_tok == EOS:
            break

        generated.append(next_tok)

        dec_ids = np.concatenate(
            [dec_ids, np.array([[next_tok]], dtype=np.int64)],
            axis=1
        )

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------
# MAIN SUMMARIZER (TEXT ONLY)
# ---------------------------------------------------------
LENGTH = {
    "Short (100 words)": 120,
    "Medium (250 words)": 250,
    "Long (500 words)": 350
}


def summarize_text(input_text, length):
    if not input_text or input_text.strip() == "":
        return "❌ Please paste some text."

    text = clean_text(input_text)
    max_tokens = LENGTH[length]

    # 🔥 Safe chunking to avoid long-seq ONNX crashes
    if len(text) > 1500:
        chunks = [text[i:i+700] for i in range(0, len(text), 700)]
        parts = [tiny_generate(chunk, 80) for chunk in chunks]
        combined = " ".join(parts)
        return tiny_generate(combined, max_tokens)
    else:
        return tiny_generate(text, max_tokens)


# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------
with gr.Blocks(title="📄 Tiny T5 ONNX Text Summarizer") as app:
    gr.Markdown("## 📄 Tiny T5 ONNX Text Summarizer")
    gr.Markdown("Paste text below and click **Summarize**")

    input_box = gr.Textbox(
        label="Paste Text",
        placeholder="Paste or type your text here...",
        lines=12
    )

    length_input = gr.Dropdown(
        ["Short (100 words)", "Medium (250 words)", "Long (500 words)"],
        value="Medium (250 words)",
        label="Summary Length"
    )

    output = gr.Textbox(label="Summary", lines=10)
    btn = gr.Button("Summarize")

    btn.click(summarize_text, [input_box, length_input], output)


# ---------------------------------------------------------
# RENDER SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False
    )
