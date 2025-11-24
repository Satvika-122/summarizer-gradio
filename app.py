import os
import gradio as gr
import re
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import time

print("🚀 Starting app with Optimum ONNX Runtime...")

MODEL_REPO = "google/flan-t5-small"  # Using a reliable model
MODEL_DIR = "onnx_model_optimum"
os.makedirs(MODEL_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# Load Model & Tokenizer using Optimum
# ---------------------------
print("🔧 Loading model with Optimum (this handles ONNX properly)...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

# Load ONNX model with Optimum - it will export if needed
# This properly handles dynamic shapes
model = ORTModelForSeq2SeqLM.from_pretrained(
    MODEL_REPO,
    export=True,  # Auto-export to ONNX if not already
    cache_dir=MODEL_DIR,
    provider="CPUExecutionProvider"
)

print("✔ Model and tokenizer loaded successfully")

# ---------------------------
# Cleaning helper
# ---------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()

# ---------------------------
# Generate summary using Optimum
# ---------------------------
def generate_summary(text, max_length=150):
    """
    Generate summary using Optimum's ONNX model.
    This handles all the ONNX complexity internally.
    """
    # Prepare input with summarization prompt
    prompt = "summarize: " + text
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# ---------------------------
# Summarizer with chunking for long texts
# ---------------------------
CHUNK_SIZE = 1000  # characters per chunk

LENGTH_MAP = {
    "Short (100 words)": 80,
    "Medium (250 words)": 150,
    "Long (500 words)": 250
}

def summarize_text(input_text, length):
    if not input_text or input_text.strip() == "":
        return "❌ Please paste some text."

    text = clean_text(input_text)
    
    # If text is short, summarize directly
    if len(text) <= CHUNK_SIZE:
        try:
            return generate_summary(text, max_length=LENGTH_MAP[length])
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # For long texts, chunk and combine
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            summary = generate_summary(chunk, max_length=100)
            if summary.strip():
                summaries.append(summary)
            time.sleep(0.05)  # Small delay between chunks
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}")
            continue
    
    if not summaries:
        return "❌ Failed to generate summary."
    
    # Combine chunk summaries
    combined = " ".join(summaries)
    
    # Final summary pass
    try:
        final_summary = generate_summary(combined, max_length=LENGTH_MAP[length])
        return final_summary
    except Exception as e:
        return f"❌ Error in final summary: {str(e)}"

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(title="📄 T5 ONNX Text Summarizer") as app:
    gr.Markdown("## 📄 T5 ONNX Text Summarizer (Optimum)")
    gr.Markdown("Paste text below and click **Summarize**. Uses Optimum for proper ONNX runtime.")

    input_box = gr.Textbox(
        label="Paste Text",
        placeholder="Paste your text here...",
        lines=12
    )
    length_input = gr.Dropdown(
        ["Short (100 words)", "Medium (250 words)", "Long (500 words)"],
        value="Medium (250 words)",
        label="Summary Length"
    )
    output = gr.Textbox(label="Summary", lines=10)
    btn = gr.Button("Summarize", variant="primary")
    
    btn.click(summarize_text, [input_box, length_input], output)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    ) 
