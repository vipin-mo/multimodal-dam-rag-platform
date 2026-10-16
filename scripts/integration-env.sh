# Environment for tests/integration against docker-compose.test.yml (sourced by CI and the runner).
export DATABASE_URL=postgresql://dam:dam-test-only@localhost:15432/dam
export QDRANT_HOST=localhost QDRANT_PORT=16333
export MINIO_ENDPOINT=http://localhost:19000 MINIO_PUBLIC_ENDPOINT=http://localhost:19000
export MINIO_ACCESS_KEY=dam-test-access MINIO_SECRET_KEY=dam-test-only-secret MINIO_BUCKET=dam-it
export CELERY_BROKER_URL=memory:// CELERY_RESULT_BACKEND=cache+memory://
export JWT_SECRET=integration-test-secret-key-0123456789abcdef
export ADMIN_PASSWORD=integration-admin CAPTION_IMAGES=false WORK_DIR=${TMPDIR:-/tmp}/dam-it-work
