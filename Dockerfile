FROM python:3.12-slim

WORKDIR /app

# System dependency for lxml/beautifulsoup4's lxml parser and pymupdf's
# native extension; kept minimal on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Zero-config defaults (fake embeddings, stub LLM, in-memory Chroma) so the
# container runs standalone with no external services. Override via
# environment variables (see .env.example) to point at real local models.
ENV RAG_EMBEDDING_BACKEND=fake \
    RAG_LLM_BACKEND=stub \
    RAG_VECTOR_STORE_BACKEND=chroma

EXPOSE 8000

CMD ["uvicorn", "ragengine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
