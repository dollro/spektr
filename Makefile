.PHONY: docs-serve docs-build

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build
