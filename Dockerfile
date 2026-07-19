FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Сначала копируем requirements.txt (редко меняется)
COPY requirements.txt .

# 2. Устанавливаем зависимости (кешируется)
RUN pip install --no-cache-dir -r requirements.txt

# 3. ПОТОМ копируем код (меняется часто)
COPY . .

RUN mkdir -p /app/uploads /app/results

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]