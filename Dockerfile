FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model \
    HF_HUB_OFFLINE=1

WORKDIR /app

# CPU-only torch wheel (much smaller than the default CUDA build), then the rest.
COPY requirements-api.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements-api.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

# Weights are not baked into the image: mount them at run time with
#   docker run -v "$(pwd)/model:/app/model" -p 8000:8000 receipt-api
EXPOSE 8000
CMD ["uvicorn", "receipt.api:app", "--host", "0.0.0.0", "--port", "8000"]
