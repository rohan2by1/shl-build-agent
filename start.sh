#!/usr/bin/env bash
set -o errexit

echo "1. Generating Vector Embeddings..."
python ingest.py

echo "2. Starting FastAPI Server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT