#!/usr/bin/env bash
# Starts throwaway Qdrant/Postgres/MinIO, runs the integration tests against them, tears them down.
# The first run downloads the embedding and Whisper models (~800 MB) into ~/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose -f docker-compose.test.yml up -d --wait
trap 'docker compose -f docker-compose.test.yml down -v >/dev/null 2>&1' EXIT

source scripts/integration-env.sh
python -m pytest -m integration -p no:cacheprovider "$@"
