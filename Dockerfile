# syntax=docker/dockerfile:1.7
# Three images from one file:  --target api | worker | ui
#   ui      Streamlit only (no ML stack)
#   api     FastAPI gateway + CPU-only torch + sentence-transformers
#   worker  Celery worker + Whisper, OpenCV, pypdf
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf
WORKDIR /app
RUN useradd --create-home --uid 10001 dam \
 && mkdir -p /models/hf /work \
 && chown -R dam:dam /models /work

# ---------------------------------------------------------------- UI
FROM base AS ui
COPY requirements/ui.txt requirements/
RUN pip install -r requirements/ui.txt
COPY app/__init__.py app/ui.py app/
USER dam
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["sh", "-c", "streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0 --server.maxUploadSize=${MAX_UPLOAD_MB:-200} --browser.gatherUsageStats=false"]

# ---------------------------------------------------------------- shared ML layer
FROM base AS ml
ARG TORCH_VERSION=2.14.0
# CPU wheels: the default PyPI torch pulls ~3 GB of CUDA libraries these containers never use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION}
COPY requirements/common.txt requirements/
RUN pip install -r requirements/common.txt

# ---------------------------------------------------------------- API gateway
FROM ml AS api
COPY requirements/api.txt requirements/
RUN pip install -r requirements/api.txt
COPY app/ app/
USER dam
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- ingestion worker
FROM ml AS worker
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements/worker.txt requirements/
RUN pip install -r requirements/worker.txt
COPY app/ app/
USER dam
ENV WORK_DIR=/work
HEALTHCHECK --interval=30s --timeout=20s --start-period=60s \
  CMD celery -A app.processor.celery_app inspect ping -d "celery@$HOSTNAME" --timeout 10 > /dev/null || exit 1
CMD ["sh", "-c", "celery -A app.processor.celery_app worker -Q ${INGESTION_QUEUE:-ingestion} --concurrency=${WORKER_CONCURRENCY:-2} --loglevel=info"]
