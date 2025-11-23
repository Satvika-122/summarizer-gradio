import gradio as gr
import pdfplumber
import io
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# -------------------------------
# LOAD ONNX MODELS FROM HF HUB
# -------------------------------
import os
import requests

# -------------------------------
# DOWNLOAD ONNX FILES LOCALLY
# -------------------------------

MODEL_REPO = "Satvi/distilbart-onnx"
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"

MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "encoder.onnx": f"{HF_BASE}/encoder.onnx",
    "decoder.onnx": f"{HF_BASE}/decoder.onnx",
}

def download_if_needed():
    for filename, url in FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"📥 Downloading {filename}...")
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Saved: {path}")
        else:
            print(f"✔ {filename} already exists.")

download_if_needed()

# -------------------------------
# LOAD ONNX MODELS LOCALLY
# -------------------------------
encoder_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "encoder.onnx"),
    providers=["CPUExecutionProvider"]
)

decoder_sess = ort.InferenceSession(
    os.path.join(MODEL_DIR, "decoder.onnx"),
    providers=["CPUExecutionProvider"]
)

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("�", "")
    text = re.sub(r"(?<=[.,;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# PDF / TXT EXTRACTION
# -------------------------------
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
        except:
            return None

        return text

    return None

# -------------------------------
# WORD LIMITS
# -------------------------------
LENGTH_MAP = {
    "100 words": (120, 50),
    "250 words": (350, 200),
    "500 words": (550, 350),
}

# -------------------------------
# ONNX SUMMARIZER (token-by-token)
# -------------------------------
def onnx_summarize(text, max_len, min_len):
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]

    encoder_out = encoder_sess.run(
        ["last_hidden_state"],
        {"input_ids": input_ids}
    )[0]

    decoder_input_ids = np.array([[tokenizer.bos_token_id]])

    for _ in range(max_len):
        out = decoder_sess.run(
            ["hidden_states"],
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_out
            }
        )[0]

        next_token_logits = out[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1).reshape(1, 1)
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

# -------------------------------
# MAIN LOGIC (chunks + final pass)
# -------------------------------
def summarize_document(file, word_limit):
    raw = extract_text(file)
    if not raw:
        return "❌ Could not extract text. The PDF may be image-based."

    raw = clean_text(raw)

    max_len, min_len = LENGTH_MAP[word_limit]

    chunks = [raw[i:i+900] for i in range(0, len(raw), 900)]
    partial = []

    # First pass (chunk summaries)
    for chunk in chunks:
        summary = onnx_summarize(chunk, max_len=200, min_len=80)
        partial.append(summary)

    combined = " ".join(partial)

    # Second pass (final summary)
    final = onnx_summarize(combined, max_len=max_len, min_len=min_len)

    return final

# -------------------------------
# GRADIO UI
# -------------------------------
app = gr.Interface(
    fn=summarize_document,
    inputs=[
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(["100 words", "250 words", "500 words"], value="250 words"),
    ],
    outputs=gr.Markdown(label="Summary Output"),
    title="📄 ONNX Document Summarizer (Render-Friendly)",
)

app.launch(server_name="0.0.0.0", server_port=7860)
