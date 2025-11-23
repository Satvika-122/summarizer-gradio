FROM python:3.10-slim

WORKDIR /app

# Install pdfplumber dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    poppler-utils \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
