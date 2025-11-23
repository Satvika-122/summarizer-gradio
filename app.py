import gradio as gr
import pdfplumber
from transformers import pipeline
from pdf2image import convert_from_bytes
import easyocr
import io
import re

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=-1
)

ocr_reader = easyocr.Reader(['en'], gpu=False)

def clean_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("�", "")
    text = re.sub(r"(?<=[.,;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_ocr(file_bytes):
    pages = convert_from_bytes(file_bytes)
    full_text = ""
    for page in pages:
        result = ocr_reader.readtext(page, detail=0)
        full_text += " ".join(result) + "\n"
    return full_text

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
        extracted_text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        extracted_text += t + "\n"
        except:
            extracted_text = ""

        if len(extracted_text.strip()) < 30:
            extracted_text = extract_text_ocr(file_bytes)

        return extracted_text

    return None

LENGTH_MAP = {
    "100 words": (120, 50),
    "250 words": (350, 200),
    "500 words": (550, 350)
}

def summarize_document(file, word_limit):
    raw_text = extract_text(file)
    if not raw_text or raw_text.strip() == "":
        return "❌ Could not extract text. File may be empty or image-only."

    raw_text = clean_text(raw_text)

    max_len, min_len = LENGTH_MAP[word_limit]

    try:
        chunks = [raw_text[i:i+900] for i in range(0, len(raw_text), 900)]
        partial = []

        for chunk in chunks:
            out = summarizer(
                chunk,
                max_length=min(max_len, 400),
                min_length=min(min_len, 150),
                do_sample=False
            )
            partial.append(out[0]["summary_text"])

        combined = " ".join(partial)

        final = summarizer(
            combined,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return final[0]["summary_text"]

    except Exception as e:
        return "❌ Summarization failed: " + str(e)

app = gr.Interface(
    fn=summarize_document,
    inputs=[
        gr.File(label="Upload PDF or TXT"),
        gr.Dropdown(["100 words", "250 words", "500 words"], value="250 words", label="Summary Length")
    ],
    outputs=gr.Markdown(label="Summary Output"),
    title="📄 Intelligent Document Summarizer with OCR",
)

app.launch(server_name="0.0.0.0", server_port=7860)
