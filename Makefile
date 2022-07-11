.PHONY: all lint diff format test

all: format lint test

lint:
	python3 -m isort -c vespa/ tests/
	python3 -m black --check vespa/ tests/
	python3 -m flake8 vespa/ tests/
	python3 -m mypy vespa/ tests/

diff:
	python3 -m isort --diff vespa/ tests/
	python3 -m black --diff vespa/ tests/

format:
	python3 -m isort vespa/ tests/
	python3 -m black vespa/ tests/

test:
	pip3 install .
	coverage run -m pytest -v
	coverage report -m

