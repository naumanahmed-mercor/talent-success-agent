.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests run_local

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

######################
# LOCAL TESTING
######################

# Run agent locally with conversation ID
# Usage: make run_local CONV_ID=215471618006513
# Options: CONV_ID (required), FULL_CONV=true, NO_DRY_RUN=true
run_local:
ifndef CONV_ID
	@echo "Error: CONV_ID is required"
	@echo "Usage: make run_local CONV_ID=215471618006513"
	@echo "Options:"
	@echo "  FULL_CONV=true     - Use full conversation (default: first message only)"
	@echo "  NO_DRY_RUN=true    - Actually write to Intercom (default: dry run)"
	@exit 1
endif
	@PYTHONPATH=src:$$PYTHONPATH python scripts/run_local.py $(CONV_ID) \
		$(if $(FULL_CONV),--full-conversation,) \
		$(if $(NO_DRY_RUN),--no-dry-run,)

######################
# TESTS
######################

test:
	python -m pytest $(TEST_FILE)

integration_tests:
	python -m pytest tests/integration_tests 

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'Local Testing:'
	@echo '  make run_local CONV_ID=<id>        - run agent locally (dry run, first message)'
	@echo '  make run_local CONV_ID=<id> FULL_CONV=true  - run with full conversation'
	@echo '  make run_local CONV_ID=<id> NO_DRY_RUN=true - run with actual Intercom writes'
	@echo ''
	@echo 'Testing:'
	@echo '  make test                          - run unit tests'
	@echo '  make tests                         - run unit tests'
	@echo '  make test TEST_FILE=<test_file>    - run all tests in file'
	@echo '  make test_watch                    - run unit tests in watch mode'
	@echo ''
	@echo 'Code Quality:'
	@echo '  make format                        - run code formatters'
	@echo '  make lint                          - run linters'

