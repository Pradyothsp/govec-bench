#!/usr/bin/env bash
# Tears down this project's Docker footprint to reclaim disk space: stops and
# removes the govec-bench containers, every image this compose file builds or
# pulls (govec, govec-scalar, chromadb/chroma, qdrant/qdrant), and all of this
# project's volumes -- both the ones declared in docker-compose.yml
# (chroma_data, qdrant_data) and any orphaned ones left over from a prior
# crashed run, matched by Compose's project-label rather than by name so
# nothing project-scoped is missed. Scoped to this project only -- doesn't
# touch any other project's containers/images/volumes.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT="govec-bench"

echo "Disk usage before cleanup:"
docker system df

echo
echo "Stopping and removing containers + declared volumes..."
docker compose down -v --remove-orphans

echo
echo "Removing images (govec, govec-scalar, chroma, qdrant)..."
docker compose down --rmi all >/dev/null 2>&1 || true
#docker image rm -f chromadb/chroma:latest qdrant/qdrant:latest 2>/dev/null || true

echo
echo "Removing any remaining ${PROJECT} volumes..."
leftover_volumes=$(docker volume ls -q --filter "label=com.docker.compose.project=${PROJECT}")
if [[ -n "${leftover_volumes}" ]]; then
  docker volume rm -f "${leftover_volumes}"
else
  echo "  none found"
fi

echo
echo "Disk usage after cleanup:"
docker system df
