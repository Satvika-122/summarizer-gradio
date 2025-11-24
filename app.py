# app.py — Fixed for Render deployment
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

# ---------------------------
# CONFIG
# ---------------------------
MODEL_REPO = "Satvi/tiny_t5"
MODEL_DIR = "onnx_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# DOWNLOAD UTILS
# ---------------------------
def download_file(url, dest):
    """Download file with retry logic"""
    for attempt in range(3):
        try:
            print(f"📥 Downloading {os.path.basename(dest)} (attempt {attempt + 1})...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded {os.path.basename(dest)}")
            return True
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return False

def download_models():
    """Download ONNX models if they don't exist"""
    files = {
        "encoder.onnx": f"https://huggingface.co/{MODEL_REPO}/resolve/main/encoder.onnx",
        "decoder.onnx": f"https://huggingface.co/{MODEL_REPO}/resolve/main/decoder.onnx",
    }
    
    for filename, url in files.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest_path):
            success = download_file(url, dest_path)
            if not success:
                raise Exception(f"Failed to download {filename}")
        else:
            print(f"✅ {filename} already exists")

# ---------------------------
# MODEL LOADING
# ---------------------------
print("🔧 Initializing models...")

try:
    # Download models first
    download_models()
    
    # Load tokenizer
    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    
    # Load ONNX sessions
    print("🔹 Loading ONNX sessions...")
    enc_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "encoder.onnx"),
        providers=['CPUExecutionProvider']
    )
    dec_sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "decoder.onnx"), 
        providers=['CPUExecutionProvider']
    )
    
    PAD_ID = tokenizer.pad_token_id or 0
    EOS_ID = tokenizer.eos_token_id
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise e

# ---------------------------
# TEXT PROCESSING
# ---------------------------
def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(file_obj):
    """Extract text from PDF or TXT file"""
    try:
        if file_obj.name.endswith('.txt'):
            return file_obj.read().decode('utf-8', errors='ignore')
        
        elif file_obj.name.endswith('.pdf'):
            text = ""
            with pdfplumber.open(io.BytesIO(file_obj.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return text
            
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
    
    return None

# ---------------------------
# SUMMARIZATION
# ---------------------------
def tiny_generate(text, max_len=120):
    """Generate summary using ONNX model"""
    try:
        prompt = "summarize: " + text
        tokens = tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].astype(np.int64)

        # Encoder forward pass
        enc_output = enc_sess.run(None, {"input_ids": input_ids})[0]

        # Decoder generation loop
        dec_ids = np.array([[PAD_ID]], dtype=np.int64)
        generated_tokens = []

        for _ in range(max_len):
            logits = dec_sess.run(None, {
                "decoder_input_ids": dec_ids,
                "encoder_hidden_states": enc_output
            })[0]

            next_token = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(np.int64)
            dec_ids = np.concatenate([dec_ids, next_token], axis=1)

            token_id = int(next_token[0, 0])
            if token_id == EOS_ID:
                break
            generated_tokens.append(token_id)

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"❌ Error during summarization: {str(e)}"

# ---------------------------
# MAIN FUNCTION
# ---------------------------
LENGTH_MAP = {
    "Short (100 words)": 120,
    "Medium (250 words)": 250, 
    "Long (500 words)": 350,
}

def summarize_document(file, length):
    """Main summarization function"""
    if file is None:
        return "❌ Please upload a file."
    
    raw_text = extract_text(file)
    if not raw_text:
        return "❌ Could not extract text. The PDF may be image-based or corrupted."
    
    raw_text = clean_text(raw_text)
    max_len = LENGTH_MAP[length]
    
    # For very long documents, chunk them
    if len(raw_text) > 1000:
        chunks = [raw_text[i:i+800] for i in range(0, len(raw_text), 800)]
        chunk_summaries = []
        
        for chunk in chunks:
            if len(chunk) > 50:  # Only summarize substantial chunks
                summary = tiny_generate(chunk, max_len=80)
                chunk_summaries.append(summary)
        
        if chunk_summaries:
            combined = " ".join(chunk_summaries)
            final_summary = tiny_generate(combined, max_len=max_len)
        else:
            final_summary = tiny_generate(raw_text[:1000], max_len=max_len)
    else:
        final_summary = tiny_generate(raw_text, max_len=max_len)
    
    return final_summary

# ---------------------------
# GRADIO UI WITH BLOCKS (FIXES THE SCHEMA ERROR)
# ---------------------------
with gr.Blocks(title="📄 Tiny T5 ONNX Document Summarizer") as app:
    gr.Markdown("# 📄 Tiny T5 ONNX Document Summarizer")
    gr.Markdown("Upload a PDF or TXT file to generate an AI-powered summary")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".txt"],
                type="filepath"
            )
            length_dropdown = gr.Dropdown(
                choices=["Short (100 words)", "Medium (250 words)", "Long (500 words)"],
                value="Medium (250 words)",
                label="Summary Length"
            )
            submit_btn = gr.Button("Generate Summary", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Summary",
                lines=8,
                max_lines=12,
                placeholder="Your summary will appear here..."
            )
    
    # Examples
    gr.Examples(
        examples=[],
        inputs=[file_input],
        label="Example Documents (add your own files here)"
    )
    
    # Connect the button
    submit_btn.click(
        fn=summarize_document,
        inputs=[file_input, length_dropdown],
        outputs=output_text
    )

# ---------------------------
# LAUNCH APP (CRITICAL FIX)
# ---------------------------
if __name__ == "__main__":
    # This is the key fix for Render
    app.launch(
        server_name="0.0.0.0",
        server_port=10000,
        share=False,  # Must be False for Render
        show_error=True,
        quiet=True  # Reduces verbose logging
    )
