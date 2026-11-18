.PHONY: install test lint run

install:
	uv venv --python 3.12 .venv
	UV_TORCH_BACKEND=cpu uv pip install --python .venv/bin/python -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/mypy

run:
	.venv/bin/trustmat all
