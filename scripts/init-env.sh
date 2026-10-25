#!/usr/bin/env bash
# Creates .env from .env.example with random values for every empty secret.
#   ./scripts/init-env.sh              admin + a read-mostly viewer account (handy for demoing roles)
#   ./scripts/init-env.sh --no-viewer  admin only
set -euo pipefail
cd "$(dirname "$0")/.."
create_viewer=true
[[ "${1:-}" == "--no-viewer" ]] && create_viewer=false
if [[ -f .env ]]; then
  echo ".env already exists; not overwriting." >&2
  exit 1
fi
rand() { openssl rand -hex "${1:-24}"; }
while IFS= read -r line; do
  if [[ "$line" =~ ^(JWT_SECRET)=[[:space:]]*(#.*)?$ ]]; then
    echo "JWT_SECRET=$(rand 32)"
  elif [[ "$line" =~ ^VIEWER_PASSWORD= ]] && ! $create_viewer; then
    echo "VIEWER_PASSWORD="
  elif [[ "$line" =~ ^([A-Z_]*PASSWORD)=[[:space:]]*(#.*)?$ ]]; then
    echo "${BASH_REMATCH[1]}=$(rand 12)"
  else
    echo "$line"
  fi
done < .env.example > .env
chmod 600 .env
login() { echo "$(grep "^$1_USERNAME=" .env | cut -d= -f2) / $(grep "^$1_PASSWORD=" .env | cut -d= -f2)"; }
echo "Wrote .env (mode 600). Accounts created on first start:"
echo "  admin:  $(login ADMIN)"
if $create_viewer; then
  echo "  viewer: $(login VIEWER)   (search, ask, summaries; no uploads or deletes. Skip with --no-viewer)"
else
  echo "  no viewer account (VIEWER_PASSWORD is empty)"
fi
