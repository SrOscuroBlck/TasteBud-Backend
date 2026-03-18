FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (includes Playwright browser deps)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers + OS-level deps in one step
RUN playwright install --with-deps chromium

RUN mkdir -p /app/data/faiss_indexes /app/uploads/menus

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
