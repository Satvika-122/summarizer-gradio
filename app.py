import os
import requests
import gradio as gr
import pdfplumber
import io
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import time

print("🚀 Starting application...")

MODEL_REPO = "Satvi/tiny_t5"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# DOWNLOAD UTILS
# ---------------------------
def download_file(url, dest):
    for attempt in range(3):
        try:
            print(f"📥 Downloading {os.path.basename(dest)}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

            print(f"✅ Downloaded {os.path.basename(dest)}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            time.sleep(2)
    return False


def download_models():
    files = {
        "encoder.onnx": f"https://huggingface.co/{MODEL_REPO}/resolve/main/encoder.onnx",
        "decoder.onnx": f"https://huggingface.co/{MODEL_REPO}/resolve/main/decoder.onnx",
    }

    for name, url in files.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            if not download_file(url, path):
                raise Exception(f"Failed to download {name}")
        else:
            print(f"✅ {name} exists")


# ---------------------------
# LOAD MODELS
# ---------------------------
print("🔧 Initializing models...")

try:
    download_models()

    # FIXED TOKENIZER LOADING FOR RENDER
    print("🔹 Loading tokenizer...")
    try:
        # First try with use_fast=False
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO,
            use_fast=False,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded with use_fast=False")
    except Exception as tokenizer_error:
        print(f"❌ First tokenizer attempt failed: {tokenizer_error}")
        # Fallback to T5Tokenizer specifically
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(MODEL_REPO)
        print("✅ Tokenizer loaded with T5Tokenizer fallback")

    print("🔹 Loading ONNX sessions...")
    enc_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "encoder.onnx"),
        providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "decoder.onnx"),
        providers=["CPUExecutionProvider"]
    )

    PAD = tokenizer.pad_token_id or 0
    EOS = tokenizer.eos_token_id

    print("✅ ONNX models loaded")

except Exception as e:
    print("❌ Model load error:", e)
    raise


# ---------------------------
# TEXT EXTRACT
# ---------------------------
def extract_text(file_obj):
    try:
        if file_obj.name.endswith(".txt"):
            return file_obj.read().decode("utf-8", errors="ignore")

        if file_obj.name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(file_obj.read())) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + " "
            return text
    except:
        return None


# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()


# ---------------------------
# SUMMARIZATION
# ---------------------------
def tiny_generate(text, max_len=120):
    try:
        prompt = "summarize: " + text

        tokens = tokenizer(prompt, return_tensors="np",
                           truncation=True, max_length=512)
        input_ids = tokens["input_ids"].astype(np.int64)

        enc_out = enc_sess.run(None, {"input_ids": input_ids})[0]

        dec_ids = np.array([[PAD]], dtype=np.int64)
        generated = []

        for _ in range(max_len):
            logits = dec_sess.run(None, {
                "input_ids": dec_ids,
                "encoder_hidden_states": enc_out
            })[0]

            next_tok = int(np.argmax(logits[:, -1, :]))

            if next_tok == EOS:
                break

            generated.append(next_tok)
            dec_ids = np.concatenate(
                [dec_ids, np.array([[next_tok]], dtype=np.int64)], axis=1
            )

        return tokenizer.decode(generated, skip_special_tokens=True)

    except Exception as e:
        print("❌ Generation error:", e)
        return f"Error: {e}"


# ---------------------------
# MAIN SUMMARIZER
# ---------------------------
LENGTH = {
    "Short (100 words)": 120,
    "Medium (250 words)": 250,
    "Long (500 words)": 350
}


def summarize_document(file, length):
    if not file:
        return "❌ Please upload a file"

    text = extract_text(file)
    if not text:
        return "❌ Unable to extract text"

    text = clean_text(text)
    max_len = LENGTH[length]

    if len(text) > 1200:
        chunks = [text[i:i+900] for i in range(0, len(text), 900)]
        parts = [tiny_generate(ch, 80) for ch in chunks]
        combined = " ".join(parts)
        return tiny_generate(combined, max_len)
    else:
        return tiny_generate(text, max_len)


# ---------------------------
# GRADIO UI
# ---------------------------
with gr.Blocks(title="📄 Tiny T5 ONNX Document Summarizer") as app:
    gr.Markdown("## 📄 Tiny T5 ONNX Document Summarizer")

    file_input = gr.File(label="Upload PDF or TXT",
                         file_types=[".pdf", ".txt"])
    length_input = gr.Dropdown(
        ["Short (100 words)", "Medium (250 words)", "Long (500 words)"],
        value="Medium (250 words)"
    )
    output = gr.Textbox(label="Summary", lines=10)

    btn = gr.Button("Summarize")
    btn.click(summarize_document, [file_input, length_input], output)


# ---------------------------
# RENDER FIX → LAUNCH SERVER
# ---------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),  # ← CHANGED THIS LINE
        share=False,
        quiet=False
    )
