FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir gradio \
    && pip install --no-cache-dir transformers==4.28.1 \
    && pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
