# app.py — Render-ready Tiny T5 Summarizer
import gradio as gr
import pdfplumber
import io
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# LOAD TINY T5 (Very small!)
# ---------------------------
MODEL_NAME = "google/t5-efficient-tiny"

print("🔹 Loading Tiny T5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Force CPU
device = torch.device("cpu")
model = model.to(device)

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------
# EXTRACT TEXT FROM PDF OR TXT
# ---------------------------
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if file.name.endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + " "
        except:
            return None
        return text

    return None

# ---------------------------
# TINY T5 SUMMARIZATION
# ---------------------------
def tiny_summarize(text, max_len=150):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    ids = model.generate(
        inputs.input_ids,
        max_length=max_len,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ---------------------------
# MAIN SUMMARY LOGIC
# ---------------------------
LENGTH_MAP = {
    "100 words": 120,
    "250 words": 250,
    "500 words": 350,
}

def summarize_document(file, length):
    raw = extract_text(file)
    if not raw:
        return "❌ Error extracting text. PDF may be image-based."

    raw = clean_text(raw)

    max_len = LENGTH_MAP[length]

    # simple chunking to avoid long input
    chunks = [raw[i:i+900] for i in range(0, len(raw), 900)]
    partial_summaries = []

    for chunk in chunks:
        s = tiny_summarize(chunk, max_len=150)
        partial_summaries.append(s)

    combined = " ".join(partial_summaries)

    final_summary = tiny_summarize(combined, max_len=max_len)

    return final_summary

# ---------------------------
# GRADIO UI
# ---------------------------
app = gr.Interface(
    summarize_document,
    [
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(["100 words", "250 words", "500 words"], value="250 words"),
    ],
    gr.Markdown(),
    title="📄 Tiny T5 Document Summarizer (Render Friendly)"
)

app.launch(server_name="0.0.0.0", server_port=7860)
