import gradio as gr
import pdfplumber
from transformers import pipeline
import io
import re

# ----------------------------------------------------------
# LIGHTWEIGHT SUMMARIZER (GOOD FOR RENDER 512MB LIMIT)
# ----------------------------------------------------------
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=-1
)

# ----------------------------------------------------------
# CLEAN TEXT
# ----------------------------------------------------------
def clean_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("�", "")
    text = re.sub(r"(?<=[.,;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------------------------------------
# EXTRACT TEXT FROM PDF / TXT (NO OCR)
# ----------------------------------------------------------
def extract_text(file):
    # File provided as string path
    if isinstance(file, str):
        filename = file
        with open(file, "rb") as f:
            file_bytes = f.read()
    # Normal uploaded file object
    elif hasattr(file, "read"):
        filename = file.name
        file_bytes = file.read()
    # Dict mode
    elif isinstance(file, dict):
        filename = file["name"]
        file_bytes = file["data"]
    else:
        return None

    # TXT extraction
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    # PDF extraction
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

# ----------------------------------------------------------
# WORD LIMIT CONFIG
# ----------------------------------------------------------
LENGTH_MAP = {
    "100 words": (120, 50),
    "250 words": (350, 200),
    "500 words": (550, 350)
}

# ----------------------------------------------------------
# MAIN SUMMARIZATION FUNCTION
# ----------------------------------------------------------
def summarize_document(file, word_limit):
    raw_text = extract_text(file)

    if not raw_text or raw_text.strip() == "":
        return "❌ Could not extract text. This PDF may be image-based."

    raw_text = clean_text(raw_text)

    max_len, min_len = LENGTH_MAP[word_limit]

    try:
        # Chunking for model limits
        chunks = [raw_text[i:i+900] for i in range(0, len(raw_text), 900)]
        partial = []

        for chunk in chunks:
            summary = summarizer(
                chunk,
                max_length=min(max_len, 400),
                min_length=min(min_len, 150),
                do_sample=False
            )
            partial.append(summary[0]["summary_text"])

        # Final combined summary
        combined = " ".join(partial)

        final = summarizer(
            combined,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return final[0]["summary_text"]

    except Exception as e:
        return f"❌ Summarization failed: {str(e)}"

# ----------------------------------------------------------
# GRADIO UI
# ----------------------------------------------------------
app = gr.Interface(
    fn=summarize_document,
    inputs=[
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(
            ["100 words", "250 words", "500 words"],
            value="250 words",
            label="Select Summary Length"
        )
    ],
    outputs=gr.Markdown(label="Summary Output (Full Text)"),
    title="📄 Document Summarizer (Lightweight – No OCR)",
)

# For Render deployment
app.launch(server_name="0.0.0.0", server_port=7860)
