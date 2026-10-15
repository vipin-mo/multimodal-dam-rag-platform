#!/usr/bin/env bash
# Creates .env from .env.example with random values for every empty secret.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then
  echo ".env already exists; not overwriting." >&2
  exit 1
fi
rand() { openssl rand -hex "${1:-24}"; }
while IFS= read -r line; do
  if [[ "$line" =~ ^(JWT_SECRET)=[[:space:]]*(#.*)?$ ]]; then
    echo "JWT_SECRET=$(rand 32)"
  elif [[ "$line" =~ ^([A-Z_]*PASSWORD)=[[:space:]]*(#.*)?$ ]]; then
    echo "${BASH_REMATCH[1]}=$(rand 12)"
  else
    echo "$line"
  fi
done < .env.example > .env
chmod 600 .env
echo "Wrote .env. Your admin login is: $(grep ^ADMIN_USERNAME= .env | cut -d= -f2) / $(grep ^ADMIN_PASSWORD= .env | cut -d= -f2)"
