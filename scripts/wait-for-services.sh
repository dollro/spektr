#!/usr/bin/env bash
set -euo pipefail

TIMEOUT=30
INTERVAL=2
ELAPSED=0

QDRANT_PORT="${QDRANT_PORT:-6333}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-cocoindex}"

echo "Waiting for services to become healthy..."

while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    QDRANT_OK=false
    NEO4J_OK=false
    POSTGRES_OK=false

    if curl -sf "http://localhost:${QDRANT_PORT}/healthz" > /dev/null 2>&1; then
        QDRANT_OK=true
    fi

    if curl -sf "http://localhost:${NEO4J_HTTP_PORT}" > /dev/null 2>&1; then
        NEO4J_OK=true
    fi

    if pg_isready -h localhost -p "$POSTGRES_PORT" -U "$POSTGRES_USER" > /dev/null 2>&1; then
        POSTGRES_OK=true
    fi

    if [ "$QDRANT_OK" = true ] && [ "$NEO4J_OK" = true ] && [ "$POSTGRES_OK" = true ]; then
        echo "All services are healthy."
        exit 0
    fi

    echo "Waiting... (${ELAPSED}s / ${TIMEOUT}s)" \
        "qdrant=$QDRANT_OK neo4j=$NEO4J_OK postgres=$POSTGRES_OK"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Timeout: services not ready after ${TIMEOUT}s."
exit 1
