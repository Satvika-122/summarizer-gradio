# app.py — Render-ready ONNX T5-small summarizer
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
MODEL_REPO = "Satvi/t5-small-onnx"                # <- your uploaded HF repo
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "encoder.onnx": f"{HF_BASE}/encoder.onnx",
    "decoder.onnx": f"{HF_BASE}/decoder.onnx",
}

# Example local file path from conversation (for reference)
example_uploaded_file = "/mnt/data/7daec814-addd-412d-a5b8-7795dd4414a2.png"

# ---------------------------
# HELPER: Stream-download big files (safe on low-RAM)
# ---------------------------
def stream_download(url: str, dst_path: str, chunk_size: int = 4 * 1024 * 1024):
    """Download a large file streaming to disk to avoid memory spikes."""
    headers = {"User-Agent": "hf-onxx-downloader/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

def download_if_needed():
    for fname, url in FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"📥 Downloading {fname} from HF ...")
            stream_download(url, path)
            print(f"✅ Saved {path}")
        else:
            print(f"✔ {fname} already exists at {path}")

# Ensure the ONNX files are present locally before creating sessions
download_if_needed()

# ---------------------------
# LOAD tokenizer (HF will download tokenizer files into cache)
# ---------------------------
print("🔹 Loading tokenizer from HF repo:", MODEL_REPO)
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)

# Determine a safe start token for decoder:
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
EOS_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

# ---------------------------
# CREATE ONNX SESSIONS (local files)
# ---------------------------
enc_path = os.path.join(MODEL_DIR, "encoder.onnx")
dec_path = os.path.join(MODEL_DIR, "decoder.onnx")

print("🔹 Creating ONNX sessions (local)...")
encoder_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

# Inspect output names (just for debug; not required)
# print("Encoder outputs:", [o.name for o in encoder_sess.get_outputs()])
# print("Decoder outputs:", [o.name for o in decoder_sess.get_outputs()])

# ---------------------------
# TEXT CLEANING + EXTRACTION
# ---------------------------
def clean_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("�", "")
    text = re.sub(r"(?<=[.,;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text(file):
    if isinstance(file, str):
        filename = file
        with open(file, "rb") as f:
            file_bytes = f.read()
    elif hasattr(file, "read"):
        filename = file.name
        file_bytes = file.read()
    elif isinstance(file, dict):
        filename = file["name"]
        file_bytes = file["data"]
    else:
        return None

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    if filename.endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception as e:
            print("PDF extract error:", e)
            return None
        return text
    return None

# ---------------------------
# ONNX INFERENCE (T5 decoder returns logits)
# ---------------------------
def onnx_summarize(text, max_len=150):
    # T5 usually expects a "summarize: " prefix for instruction-based models
    prompt = "summarize: " + text

    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Run encoder
    encoder_out = encoder_sess.run(None, {"input_ids": input_ids})[0]  # shape (batch, seq, dim)

    # Prepare decoder input ids (start with PAD token or decoder_start_token)
    decoder_input_ids = np.array([[PAD_ID]], dtype=np.int64)

    generated = []

    for step in range(max_len):
        # decoder expects (decoder_input_ids, encoder_hidden_states)
        outputs = decoder_sess.run(None, {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_out
        })

        # Depending on how decoder was exported, the logits output name may vary.
        # We assume the first output is logits with shape (batch, seq, vocab)
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]  # (batch, vocab)
        next_token_id = np.argmax(next_token_logits, axis=-1).reshape(1, 1).astype(np.int64)

        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=1)
        generated.append(int(next_token_id[0, 0]))

        if EOS_ID is not None and int(next_token_id[0,0]) == EOS_ID:
            break

    # decode (skip special tokens)
    decoded = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded

# ---------------------------
# MAIN DOCUMENT SUMMARIZER
# ---------------------------
LENGTH_MAP = {
    "100 words": (120, 50),
    "250 words": (350, 200),
    "500 words": (550, 350),
}

def summarize_document(file, word_limit):
    raw = extract_text(file)
    if not raw:
        return "❌ Could not extract text. The PDF may be image-based."

    raw = clean_text(raw)
    max_len, min_len = LENGTH_MAP.get(word_limit, (350, 200))

    # simple chunking by characters (you can improve)
    chunks = [raw[i:i+900] for i in range(0, len(raw), 900)]
    partial = []
    for chunk in chunks:
        s = onnx_summarize(chunk, max_len=150)
        partial.append(s)

    combined = " ".join(partial)
    final = onnx_summarize(combined, max_len=max_len)

    return final

# ---------------------------
# GRADIO UI
# ---------------------------
app = gr.Interface(
    fn=summarize_document,
    inputs=[
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(["100 words", "250 words", "500 words"], value="250 words", label="Summary Length"),
    ],
    outputs=gr.Markdown(label="Summary Output"),
    title="📄 ONNX T5-small Document Summarizer (Render-friendly)",
    description=f"Example uploaded image path (for debugging): {example_uploaded_file}"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
