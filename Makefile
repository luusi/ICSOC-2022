.PHONY: clean clean-test clean-pyc clean-build docs

clean: clean-build clean-pyc clean-test # remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr coverage.xml

lint-all: black isort lint static ## run all linters

lint: ## check style with flake8
	flake8 icsoc_2022 tests

static: ## static type checking with mypy
	mypy icsoc_2022 tests

isort: ## sort import statements with isort
	isort icsoc_2022 tests

isort-check: ## check import statements order with isort
	isort --check-only icsoc_2022 tests

black: ## apply black formatting
	black icsoc_2022 tests

black-check: ## check black formatting
	black --check --verbose icsoc_2022 tests

pylint: ## run pylint
	pylint icsoc_2022 tests

test: ## run tests quickly with the default Python
	pytest tests --doctest-modules icsoc_2022 tests/ \
        --cov=icsoc_2022 \
        --cov-report=xml \
        --cov-report=html \
        --cov-report=term

docs: ## generate MkDocs HTML documentation, including API docs
	mkdocs build --clean
	$(BROWSER) site/index.html

servedocs: docs ## compile the docs watching for changes
	mkdocs build --clean
	python -c 'print("###### Starting local server. Press Control+C to stop server ######")'
	mkdocs serve
