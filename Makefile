.PHONY: test format lint check

test:
	poetry run pytest

coverage:
	poetry run pytest --cov=connectfour --cov-report=term-missing

format:
	uv run black connectfour tests
	uv run isort connectfour tests

lint:
	poetry run mypy connectfour
	poetry run black --check connectfour tests
	poetry run isort --check-only connectfour tests

check: format lint test