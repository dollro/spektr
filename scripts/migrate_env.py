"""Migrate a pre-CocoIndex-v1 `.env.prod` to the current schema.

The v0->v1 upgrade removed the PostgreSQL service and added the LMDB state
path, the S3 trigger tuning knobs and a concurrency cap. An env file written
for the old stack therefore carries dead variables and is missing new ones;
the pipeline will not tell you about either -- it just runs with defaults.

This script rewrites such a file: it drops the dead variables, appends the
missing ones with their documented defaults, retunes two values that are
stale in `.env.example` itself, and validates that everything required is
present. Comments, ordering and untouched lines are preserved.

Deliberately stdlib-only, unlike the rest of `scripts/`. It is meant to run
on a production VM that has Docker but no `uv` and no project virtualenv:

    python3 scripts/migrate_env.py .env.prod             # report only
    python3 scripts/migrate_env.py .env.prod --write     # rewrite + .bak
    python3 scripts/migrate_env.py .env.prod -o new.env  # write elsewhere

Exits 1 if the migrated file would still be invalid, so it is safe to gate a
deploy on it. Values are never printed -- only variable names.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Removed by the CocoIndex v0->v1 migration. Nothing reads these any more:
# the ledger, memo cache and component tree moved into an LMDB directory.
DEAD_VARS = frozenset(
    {
        "DATABASE_URL",
        "COCOINDEX_DATABASE_URL",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_PORT",
        "POSTGRES_HOST",
    }
)

# Appended when absent, as `NAME=default  # comment`. Order is preserved.
NEW_VARS: tuple[tuple[str, str, str], ...] = (
    ("COCOINDEX_DB_PATH", "state/cocoindex.db", "LMDB state directory (replaces Postgres)"),
    (
        "PIPELINE_MAX_CONCURRENT_FILES",
        "4",
        "caps max_inflight_components; CocoIndex defaults to 1024",
    ),
    ("S3_PREFIX", "", "optional key prefix restricting the bucket scan"),
    ("S3_SQS_DEBOUNCE_SECONDS", "5", "coalesce an event burst into one catch-up run"),
    ("S3_FULL_SCAN_INTERVAL_HOURS", "24", "safety-net sweep for missed events"),
)

# Values that `.env.example` still ships stale. Rewritten only when the file
# holds exactly the outdated value -- a deliberate override is left alone.
RETUNE: tuple[tuple[str, str, str, str], ...] = (
    (
        "JINA_DENSE_DIMENSIONS",
        "512",
        "2048",
        "code default is 2048; 512 silently halves recall",
    ),
    (
        "LLM_MODEL",
        "claude-sonnet-4-20250514",
        "claude-sonnet-5",
        "code default moved to claude-sonnet-5",
    ),
)

# Test-suite overrides. Set in production they point the stack at throwaway
# collections, which looks exactly like an empty knowledge base.
SUSPECT_VARS = frozenset({"QDRANT_DENSE_COLLECTION", "QDRANT_MULTIVEC_COLLECTION"})

REQUIRED = ("NEO4J_PASSWORD", "MCP_PUBLIC_DOMAIN", "MCP_API_KEY", "LLM_API_KEY")

# Which embedding key each provider needs.
PROVIDER_KEYS = {
    "jina": "JINA_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_ASSIGN = re.compile(r"^(\s*)(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def parse(lines: list[str]) -> dict[str, str]:
    """Map variable name -> raw value for every assignment in the file."""
    found: dict[str, str] = {}
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        match = _ASSIGN.match(line.rstrip("\n"))
        if match:
            found[match.group(2)] = _strip_value(match.group(3))
    return found


def _strip_value(raw: str) -> str:
    """Drop an inline comment and surrounding quotes from an assignment RHS."""
    value = raw.strip()
    if value[:1] not in {'"', "'"}:
        value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def transform(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (new lines, dropped var names, retuned descriptions)."""
    out: list[str] = []
    dropped: list[str] = []
    retuned: list[str] = []

    for line in lines:
        match = _ASSIGN.match(line.rstrip("\n"))
        if not match or line.lstrip().startswith("#"):
            out.append(line)
            continue
        indent, name, raw = match.groups()
        if name in DEAD_VARS:
            dropped.append(name)
            continue
        replacement = _retune(name, _strip_value(raw))
        if replacement is not None:
            new_value, note = replacement
            out.append(f"{indent}{name}={new_value}\n")
            retuned.append(f"{name}: {note}")
            continue
        out.append(line)

    return out, dropped, retuned


def _retune(name: str, value: str) -> tuple[str, str] | None:
    for var, stale, fresh, note in RETUNE:
        if name == var and value == stale:
            return fresh, f"{stale} -> {fresh} ({note})"
    return None


def append_missing(lines: list[str], present: set[str]) -> tuple[list[str], list[str]]:
    """Append a block for every NEW_VARS entry the file does not already set."""
    missing = [(n, d, c) for n, d, c in NEW_VARS if n not in present]
    if not missing:
        return lines, []

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    block = [
        "\n",
        "# ---------------------------------------------------------------------------\n",
        "# Added by scripts/migrate_env.py (CocoIndex v1 upgrade)\n",
        "# ---------------------------------------------------------------------------\n",
    ]
    for name, default, comment in missing:
        # Docker Compose only strips an inline comment when a value precedes
        # it: `VAR=  # note` sets VAR to the literal "# note". Verified against
        # compose v2. So empty defaults take the comment on the line above.
        if comment and not default:
            block.append(f"# {comment}\n")
            block.append(f"{name}=\n")
        elif comment:
            block.append(f"{name}={default}  # {comment}\n")
        else:
            block.append(f"{name}={default}\n")
    return lines + block, [name for name, _, _ in missing]


def validate(env: dict[str, str]) -> list[str]:
    """Return human-readable problems that would break the stack at boot."""
    problems: list[str] = []
    for name in REQUIRED:
        if not env.get(name):
            problems.append(f"{name} is missing or empty")

    provider = env.get("EMBEDDING_PROVIDER", "jina").strip().lower()
    key = PROVIDER_KEYS.get(provider)
    if key is None:
        problems.append(
            f"EMBEDDING_PROVIDER={provider!r} is not one of {sorted(PROVIDER_KEYS)}"
        )
    elif not env.get(key):
        problems.append(f"EMBEDDING_PROVIDER={provider} but {key} is missing or empty")

    source = env.get("DOCUMENT_SOURCE", "local").strip().lower()
    if source == "s3" and not env.get("S3_BUCKET_NAME"):
        problems.append(
            "DOCUMENT_SOURCE=s3 but S3_BUCKET_NAME is missing (settings will reject it)"
        )
    if source == "sharepoint":
        missing = [
            n for n in ("SHAREPOINT_TENANT_ID", "SHAREPOINT_CLIENT_ID") if not env.get(n)
        ]
        problems.extend(f"DOCUMENT_SOURCE=sharepoint but {n} is missing" for n in missing)

    return problems


def report(section: str, items: list[str], bullet: str = "-") -> None:
    if items:
        print(f"\n{section}")
        for item in items:
            print(f"  {bullet} {item}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("env_file", type=Path, help="the old .env.prod to migrate")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of in place")
    parser.add_argument(
        "--write", action="store_true", help="rewrite in place, keeping a .bak"
    )
    args = parser.parse_args()

    if not args.env_file.is_file():
        print(f"error: {args.env_file} does not exist", file=sys.stderr)
        return 2

    original = args.env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    migrated, dropped, retuned = transform(original)
    migrated, added = append_missing(migrated, set(parse(migrated)))

    env = parse(migrated)
    problems = validate(env)
    suspect = sorted(SUSPECT_VARS & set(env))

    print(f"{args.env_file}: {len(original)} lines in, {len(migrated)} out")
    report("Dropped (dead since CocoIndex v1):", sorted(set(dropped)))
    report("Added (missing, using documented defaults):", added)
    report("Retuned (stale value from .env.example):", retuned)
    report(
        "Review these -- test-suite overrides, remove them in production:", suspect, bullet="!"
    )
    report("PROBLEMS -- the stack will not come up correctly:", problems, bullet="x")

    if not (dropped or added or retuned):
        print("\nNothing to migrate; the file already matches the current schema.")

    destination = args.output or (args.env_file if args.write else None)
    if destination is None:
        print("\nDry run. Re-run with --write (in place, keeps a .bak) or -o FILE.")
    else:
        if destination == args.env_file:
            backup = args.env_file.with_suffix(args.env_file.suffix + ".bak")
            shutil.copy2(args.env_file, backup)
            print(f"\nBacked up to {backup}")
        destination.write_text("".join(migrated), encoding="utf-8")
        destination.chmod(0o600)
        print(f"Wrote {destination} (mode 600)")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
