import os
import gradio as gr
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import time

print("🚀 Starting application...")

MODEL_REPO = "Satvi/tiny_t5"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# LOAD MODELS
# ---------------------------
print("🔧 Initializing models...")

try:
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        use_fast=False,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded")

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

    print("✅ ONNX models ready")

except Exception as e:
    print("❌ Model load error:", e)
    raise


# ---------------------------
# CLEANING
# ---------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()


# ---------------------------
# SUMMARY GENERATION
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


def summarize_text(input_text, length):
    if not input_text or len(input_text.strip()) == 0:
        return "❌ Please paste some text."

    text = clean_text(input_text)
    max_len = LENGTH[length]

    print(f"📝 Received text of {len(text)} characters")

    if len(text) > 1200:
        chunks = [text[i:i+900] for i in range(0, len(text), 900)]
        parts = [tiny_generate(chunk, 80) for chunk in chunks]
        combined = " ".join(parts)
        return tiny_generate(combined, max_len)
    else:
        return tiny_generate(text, max_len)


# ---------------------------
# GRADIO UI (TEXT ONLY — NO FILE UPLOAD)
# ---------------------------
with gr.Blocks(title="📄 Tiny T5 ONNX Text Summarizer") as app:
    gr.Markdown("## 📄 Tiny T5 ONNX Text Summarizer")
    gr.Markdown("Paste your text below and generate a summary")

    input_text = gr.Textbox(
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
    btn.click(fn=summarize_text, inputs=[input_text, length_input], outputs=output)


# ---------------------------
# RENDER LAUNCH
# ---------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False,
        quiet=False
    )
