import os
import gradio as gr
import requests
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import re
import time

print("🚀 Starting BART ONNX Summarizer...")

MODEL_DIR = "bart_onnx"
os.makedirs(MODEL_DIR, exist_ok=True)

# URLs for Xenova's BART-base ONNX repo
FILES = {
    "model.onnx": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/model.onnx",
    "config.json": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/config.json",
    "tokenizer.json": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/tokenizer_config.json",
    "vocab.json": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/Xenova/bart-base-onnx/resolve/main/merges.txt"
}

# ---------------------------
# DOWNLOAD ONNX + TOKENIZER FILES
# ---------------------------
def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"⬇ Downloading {os.path.basename(dest)}...")
    r = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"✔ Downloaded")

print("🔍 Checking ONNX files...")
for name, url in FILES.items():
    download_file(url, os.path.join(MODEL_DIR, name))

print("✔ All files ready")

# ---------------------------
# LOAD TOKENIZER
# ---------------------------
print("🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("✔ Tokenizer ready")


# ---------------------------
# LOAD ONNX RUNTIME SESSION
# ---------------------------
print("🔧 Loading ONNX Runtime session...")
sess_opts = ort.SessionOptions()
sess_opts.enable_mem_pattern = False
sess_opts.enable_cpu_mem_arena = False
sess_opts.log_severity_level = 2

session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"]
)

print("✔ ONNX model loaded")


# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(t):
    t = re.sub(r"\s+", " ", t or "")
    return t.strip()


# ---------------------------
# GENERATE SUMMARY USING BART ONNX
# ---------------------------
def generate_summary(text, max_len=150):
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Run ONNX model
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    summary_ids = outputs[0]
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    return summary


# ---------------------------
# CHUNKING FOR LONG TEXTS
# ---------------------------
CHUNK_SIZE = 1500

LENGTH_MAP = {
    "Short (100 words)": 80,
    "Medium (250 words)": 150,
    "Long (500 words)": 250
}

def summarize_text(text, length):
    if not text:
        return "❌ Please paste some text."

    text = clean_text(text)

    # If short → direct summarization
    if len(text) <= CHUNK_SIZE:
        return generate_summary(text, max_len=LENGTH_MAP[length])

    # Long text → chunking
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"📦 Chunk {i+1}/{len(chunks)}")
        part = generate_summary(chunk, max_len=120)
        partial_summaries.append(part)
        time.sleep(0.05)

    combined_text = " ".join(partial_summaries)

    # Final summary from all partial ones
    return generate_summary(combined_text, max_len=LENGTH_MAP[length])


# ---------------------------
# GRADIO UI
# ---------------------------
with gr.Blocks(title="📄 BART ONNX Summarizer") as app:
    gr.Markdown("## 📄 BART ONNX Document Summarizer")
    gr.Markdown("Paste your text below and click **Summarize**.")

    input_box = gr.Textbox(lines=12, label="Paste text")
    length_dd = gr.Dropdown(
        ["Short (100 words)", "Medium (250 words)", "Long (500 words)"],
        value="Medium (250 words)",
        label="Summary Length"
    )
    output_box = gr.Textbox(lines=10, label="Summary")

    summarize_btn = gr.Button("Summarize")
    summarize_btn.click(summarize_text, [input_box, length_dd], output_box)


# ---------------------------
# LAUNCH (Render)
# ---------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False
    )
