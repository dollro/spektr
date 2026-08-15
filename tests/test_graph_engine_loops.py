"""The shared graph engine must survive being used from more than one loop.

`get_graph_engine()` caches a process-wide singleton, but its Neo4j async
driver binds its socket to whichever event loop is live when it is built.
pytest-asyncio gives each test a fresh loop, so the second test to touch the
engine used to fail with "got Future attached to a different loop".
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.integration
def test_engine_search_works_across_event_loops() -> None:
    from ingestion.graph_engine import get_graph_engine

    engine = get_graph_engine()

    # First loop builds and binds the driver.
    asyncio.run(engine.search("database migration", limit=1))

    # Second, unrelated loop. The cached driver must not still be bound to the
    # first one.
    asyncio.run(engine.search("database migration", limit=1))
