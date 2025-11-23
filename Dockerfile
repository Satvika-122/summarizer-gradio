FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by pdfplumber + sentencepiece
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    poppler-utils \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
