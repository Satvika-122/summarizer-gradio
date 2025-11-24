import os
import gradio as gr
import requests
import re

print("🚀 Starting Summarizer...")

# Hugging Face Inference API - no local models, no PyTorch
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()

LENGTH_MAP = {
    "Short (100 words)": 100,
    "Medium (250 words)": 150,
    "Long (500 words)": 250
}

def summarize_text(text, length):
    if not text:
        return "❌ Please paste some text."

    text = clean_text(text)
    
    try:
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": LENGTH_MAP[length],
                "min_length": 30,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, json=payload)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]['summary_text']
        elif 'error' in result:
            return f"❌ Model is loading, please wait... (try again in 30 seconds)"
        else:
            return "❌ Could not generate summary"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="📄 Text Summarizer") as app:
    gr.Markdown("## 📄 Text Summarizer")
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
