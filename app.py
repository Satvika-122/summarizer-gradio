# app.py — Render-ready Tiny T5 ONNX Summarizer
import os
import requests
import gradio as gr
import pdfplumber
import io
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ---------------------------
# CONFIG
# ---------------------------
MODEL_REPO = "Satvi/tiny_t5"
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "encoder.onnx": f"{HF_BASE}/encoder.onnx",
    "decoder.onnx": f"{HF_BASE}/decoder.onnx",
}

# ---------------------------
# STREAM-SAFE DOWNLOAD (no RAM spike)
# ---------------------------
def stream_download(url: str, dst_path: str, chunk=4 * 1024 * 1024):
    headers = {"User-Agent": "tiny-t5-onnx/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)

def download_if_needed():
    for fname, url in FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"📥 Downloading {fname} ...")
            stream_download(url, path)
            print(f"✅ Saved {path}")
        else:
            print(f"✔ {fname} exists")

download_if_needed()

# ---------------------------
# LOAD TOKENIZER
# ---------------------------
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)

PAD_ID = tokenizer.pad_token_id or 0
EOS_ID = tokenizer.eos_token_id

# ---------------------------
# ONNX SESSIONS
# ---------------------------
enc_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "encoder.onnx"),
                                providers=["CPUExecutionProvider"])
dec_sess = ort.InferenceSession(os.path.join(MODEL_DIR, "decoder.onnx"),
                                providers=["CPUExecutionProvider"])

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ---------------------------
# EXTRACT TEXT
# ---------------------------
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if file.name.endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text += t + " "
        except:
            return None
        return text
    return None

# ---------------------------
# ONNX GENERATION LOOP
# ---------------------------
def tiny_generate(text, max_len=120):
    prompt = "summarize: " + text
    tok = tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
    input_ids = tok["input_ids"].astype(np.int64)

    enc_out = enc_sess.run(None, {"input_ids": input_ids})[0]

    dec_ids = np.array([[PAD_ID]], dtype=np.int64)
    tokens = []

    for _ in range(max_len):
        logits = dec_sess.run(None, {
            "decoder_input_ids": dec_ids,
            "encoder_hidden_states": enc_out
        })[0]

        next_id = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(np.int64)
        dec_ids = np.concatenate([dec_ids, next_id], axis=1)

        nid = int(next_id[0, 0])
        if nid == EOS_ID:
            break
        tokens.append(nid)

    return tokenizer.decode(tokens, skip_special_tokens=True)

# ---------------------------
# MAIN LOGIC
# ---------------------------
LENGTH_MAP = {
    "100 words": 120,
    "250 words": 250,
    "500 words": 350,
}

def summarize_document(file, length):
    raw = extract_text(file)
    if not raw:
        return "❌ Could not extract text. PDF may be image-based."

    raw = clean_text(raw)
    max_len = LENGTH_MAP[length]

    chunks = [raw[i:i+800] for i in range(0, len(raw), 800)]
    partial = [tiny_generate(c, max_len=120) for c in chunks]

    combined = " ".join(partial)
    final = tiny_generate(combined, max_len=max_len)
    return final

# ---------------------------
# UI
# ---------------------------
app = gr.Interface(
    fn=summarize_document,
    inputs=[
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(["100 words", "250 words", "500 words"], value="250 words")
    ],
    outputs=gr.Textbox(label="Summary"),  # Changed from Markdown to Textbox
    title="📄 Tiny T5 ONNX Document Summarizer (Render Friendly)",
    description="Upload a PDF or TXT file to generate a summary"
)

# FIXED LAUNCH - Added share=False to prevent the localhost accessibility error
app.launch(server_name="0.0.0.0", server_port=10000, share=False)
