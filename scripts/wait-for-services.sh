#!/usr/bin/env bash
set -euo pipefail

TIMEOUT=30
INTERVAL=2
ELAPSED=0

echo "Waiting for services to become healthy..."

while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    QDRANT_OK=false
    NEO4J_OK=false
    POSTGRES_OK=false

    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        QDRANT_OK=true
    fi

    if curl -sf http://localhost:7474 > /dev/null 2>&1; then
        NEO4J_OK=true
    fi

    if pg_isready -h localhost -p 5432 -U cocoindex > /dev/null 2>&1; then
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
