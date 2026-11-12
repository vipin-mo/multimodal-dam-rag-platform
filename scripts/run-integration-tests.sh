#!/usr/bin/env bash
# Starts throwaway Qdrant/Postgres/MinIO, runs the integration tests against them, tears them down.
# The first run downloads the embedding and Whisper models (~800 MB) into ~/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")/.."

# Explicit -p: COMPOSE_PROJECT_NAME in .env would otherwise put these throwaway services (and the
# `down -v` below) into the main stack's project.
docker compose -p dam-test -f docker-compose.test.yml up -d --wait
trap 'docker compose -p dam-test -f docker-compose.test.yml down -v >/dev/null 2>&1' EXIT

source scripts/integration-env.sh
python -m pytest -m integration -p no:cacheprovider "$@"
