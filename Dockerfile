FROM python:3.11-slim

WORKDIR /app
COPY requirements.worker.txt .
RUN pip install --no-cache-dir -r requirements.worker.txt

# Download spaCy + sentence-transformers models at build time
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
CMD ["python", "-m", "worker.consumer"]