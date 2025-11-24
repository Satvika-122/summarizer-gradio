import os
import requests
import gradio as gr
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import time

print("🚀 Starting app…")

MODEL_REPO = "Satvi/tiny_t5"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# RELIABLE ONNX DOWNLOADER
# ---------------------------
def force_download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        print(f"✔ Using cached {os.path.basename(dest)}")
        return
    print(f"⬇ Downloading {url} ...")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise Exception(f"Download failed: HTTP {r.status_code} for {url}")
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"✔ Saved: {dest}")

def download_models():
    encoder_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/encoder.onnx"
    decoder_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/decoder.onnx"
    force_download(encoder_url, os.path.join(MODEL_DIR, "encoder.onnx"))
    force_download(decoder_url, os.path.join(MODEL_DIR, "decoder.onnx"))

# ---------------------------
# Create session options (shared)
# ---------------------------
def make_session_options():
    sess_opts = ort.SessionOptions()
    # keep optimizations but disable mem-pattern and cpu arena to avoid buffer reuse bugs
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = False
    sess_opts.enable_cpu_mem_arena = False
    sess_opts.log_severity_level = 2
    # small extra safety
    try:
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    except Exception:
        pass
    return sess_opts

# ---------------------------
# Initialize tokenizer + encoder session (decoder session will be created per-chunk)
# ---------------------------
print("🔧 Initializing models...")
download_models()

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    use_fast=False,
    trust_remote_code=True
)
PAD = tokenizer.pad_token_id or 0
EOS = tokenizer.eos_token_id

# encoder session created once
sess_opts = make_session_options()
enc_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "encoder.onnx"),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"]
)
print("✔ Encoder session loaded")

# We'll create decoder sessions on demand (per-chunk) to avoid buffer reuse across different decoder_input lengths.

# ---------------------------
# Cleaning helper
# ---------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()

# ---------------------------
# Helper to create a fresh decoder session
# ---------------------------
def create_decoder_session():
    sess_opts_local = make_session_options()
    dec = ort.InferenceSession(
        os.path.join(MODEL_DIR, "decoder.onnx"),
        sess_options=sess_opts_local,
        providers=["CPUExecutionProvider"]
    )
    return dec

# ---------------------------
# Tiny generate: creates a summary for a single (short) text chunk
# ---------------------------
def tiny_generate(text, max_len=60, tokenizer_max_len=128):
    """
    Generates up to max_len tokens for the given text chunk.
    Uses a fresh decoder session to avoid ONNX buffer-reuse shape mismatches.
    """
    prompt = "summarize: " + text

    # conservative tokenizer length for tiny model
    tokens = tokenizer(prompt, return_tensors="np", truncation=True, max_length=tokenizer_max_len)
    input_ids = tokens["input_ids"].astype(np.int64)

    # run encoder (single shared session)
    enc_out = enc_sess.run(None, {"input_ids": input_ids})[0]

    # create fresh decoder session for this generation to avoid buffer reuse problems
    dec_sess_local = create_decoder_session()

    # decoder start token (typically 0 for T5)
    decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dec_ids = np.array([[decoder_start_token_id]], dtype=np.int64)
    generated = []

    for _ in range(max_len):
        # FIXED: Use "input_ids" not "decoder_input_ids"
        logits = dec_sess_local.run(
            None,
            {
                "input_ids": dec_ids,  # ✅ FIXED
                "encoder_hidden_states": enc_out
            }
        )[0]

        next_tok = int(np.argmax(logits[:, -1, :]))
        if next_tok == EOS:
            break

        generated.append(next_tok)
        # append new token to dec_ids (grows by 1)
        dec_ids = np.concatenate([dec_ids, np.array([[next_tok]], dtype=np.int64)], axis=1)

    # release session (allow GC)
    try:
        del dec_sess_local
    except Exception:
        pass

    return tokenizer.decode(generated, skip_special_tokens=True)

# ---------------------------
# Summarizer: chunking + safe generation
# ---------------------------
# Conservative settings that work for tiny models on CPU
CHUNK_CHAR_SIZE = 500           # chunk size in characters
TOKENIZER_MAX_LEN = 128         # tokenizer max_length for encoder
SHORT_DECODER_TOKENS = 40       # tokens for "Short" summary
MEDIUM_DECODER_TOKENS = 80
LONG_DECODER_TOKENS = 120

LENGTH = {
    "Short (100 words)": SHORT_DECODER_TOKENS,
    "Medium (250 words)": MEDIUM_DECODER_TOKENS,
    "Long (500 words)": LONG_DECODER_TOKENS
}

def summarize_text(input_text, length):
    if not input_text or input_text.strip() == "":
        return "❌ Please paste some text."

    text = clean_text(input_text)

    # split into small chunks to keep encoder inputs short
    if len(text) > CHUNK_CHAR_SIZE:
        chunks = [text[i:i + CHUNK_CHAR_SIZE] for i in range(0, len(text), CHUNK_CHAR_SIZE)]
    else:
        chunks = [text]

    parts = []
    # for each chunk, generate a short summary (use smaller decode length)
    for chunk in chunks:
        part = tiny_generate(chunk, max_len= min(60, LENGTH[length]), tokenizer_max_len=TOKENIZER_MAX_LEN)
        if part and part.strip():
            parts.append(part)
        # small pause to give runtime breathing room (optional)
        time.sleep(0.01)

    combined = " ".join(parts).strip()
    if not combined:
        return "❌ Generation failed for all chunks."

    # final pass: summarize the combined parts into requested length
    final = tiny_generate(combined, max_len=LENGTH[length], tokenizer_max_len=TOKENIZER_MAX_LEN)
    return final or "❌ Final generation returned empty result."

# ---------------------------
# Gradio UI (text only)
# ---------------------------
with gr.Blocks(title="📄 Tiny T5 ONNX Text Summarizer (Stable)") as app:
    gr.Markdown("## 📄 Tiny T5 ONNX Text Summarizer — Stable")
    gr.Markdown("Paste text below and click **Summarize**. This build uses conservative limits and fresh decoder sessions to avoid ONNX shape errors.")

    input_box = gr.Textbox(label="Paste Text", placeholder="Paste text here...", lines=12)
    length_input = gr.Dropdown(["Short (100 words)", "Medium (250 words)", "Long (500 words)"], value="Medium (250 words)", label="Summary Length")
    output = gr.Textbox(label="Summary", lines=10)
    btn = gr.Button("Summarize")
    btn.click(summarize_text, [input_box, length_input], output)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)), share=False)
