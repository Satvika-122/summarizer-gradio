import os
import gradio as gr
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import re
import time

print("🚀 Starting BART ONNX Summarizer...")

MODEL_DIR = "bart_onnx"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_REPO = "Xenova/distilbart-cnn-6-6-onnx"

FILES = [
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt"
]

print("⬇ Downloading ONNX + tokenizer files from HuggingFace Hub...")

for filename in FILES:
    local_path = os.path.join(MODEL_DIR, filename)
    if (not os.path.exists(local_path)) or os.path.getsize(local_path) < 500:
        print(f"⬇ Downloading: {filename}")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✔ Downloaded: {filename}")
    else:
        print(f"✔ Already exists: {filename}")

print("🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("✔ Tokenizer ready")

print("🔧 Loading ONNX Runtime model...")
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"]
)
print("✔ ONNX model loaded")

def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()

def generate_summary(text, max_len=150):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=1024
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    )

    summary_ids = outputs[0]
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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

    if len(text) <= CHUNK_SIZE:
        return generate_summary(text, LENGTH_MAP[length])

    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    partials = []

    for i, chunk in enumerate(chunks):
        print(f"📦 Chunk {i+1}/{len(chunks)}")
        partials.append(generate_summary(chunk, 120))
        time.sleep(0.05)

    combined = " ".join(partials)
    return generate_summary(combined, LENGTH_MAP[length])

with gr.Blocks(title="📄 BART ONNX Summarizer") as app:
    gr.Markdown("## 📄 BART ONNX Summarizer")
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

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False
    )
