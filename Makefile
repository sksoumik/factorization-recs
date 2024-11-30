SHELL := /bin/bash
CONDA_ENV_NAME := recs   ## conda environment name

# https://www.gnu.org/software/make/manual/make.html#Call-Function
confirm := read -r -p "⚠  Are you sure? [y/N] " response && [[ "$$response" =~ ^([yY][eE][sS]|[yY])$$ ]]

help: ## Print help for each target
	$(info Available commands:)
	$(info ==========================================================================================)
	$(info )
	@grep '^[[:alnum:]_-]*:.* ##' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS=":.* ## "}; {printf "%-25s %s\n", $$1, $$2};'

setup-conda: ## Creates and sets up conda environment with requirements.txt
	@echo "Creating conda environment: $(CONDA_ENV_NAME)"
	conda create -n $(CONDA_ENV_NAME) python=3.11 -y
	@echo "Activating conda environment and installing dependencies..."
	conda run -n $(CONDA_ENV_NAME) pip install -r requirements.txt
	@echo "Conda environment setup complete. Activate it with: conda activate $(CONDA_ENV_NAME)"

clean-conda: ## Removes the conda environment
	conda deactivate
	@echo "Removing conda environment: $(CONDA_ENV_NAME)"
	conda env remove -n $(CONDA_ENV_NAME)


copy-env: ## Copies .env to .env.bak and creates a new one from .env.example
	@echo "Your may lose .env.bak"
	@if $(call confirm); then \
		cp .env .env.bak || true ; \
		cp .env.example .env ; \
	fi


setup-pre-commit: ## Installs pre-commit-hook
	@echo "Installing pre-commit-hook"
	poetry run pre-commit install


setup: setup-conda setup-pre-commit ## Sets up local-development environment

run: ## Runs the service locally using poetry

start: ## Starts the service using docker

stop: ## Stops docker containers
	docker compose down --remove-orphans

clean: ## Cleans up the local-development environment except .env
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -f .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} \;
	find . -name .pytest_cache -type d -prune -exec rm -rf {} \;
	rm -rf .vscode
	rm -rf .venv
	rm -f poetry.lock
	rm -rf experiment_results
	rm -rf logs
	rm -rf output

#################################################################################
# Formatting checks #############################################################

check-isort: ## Checks if .py files are formatted with isort
	@echo "Checking isort formatting(without update)"
	isort --check --diff .

check-black: ## Checks if .py files are formatted with black
	@echo "Checking black formatting(without change)"
	black --line-length 79 --check --diff .

check-format: check-isort check-black

#################################################################################
# Formatting fixes ##############################################################

format-isort: ## Fixes .py files with isort
	@echo "Fixing isort formatting issues"
	isort .

format-black: ## Fixes .py files with black
	@echo "Fixing black formatting issues"
	black . --line-length 88

format-unused-imports: ## Fixes unused imports and unused variables
	@echo "Removing unused imports"
	autoflake -i --remove-all-unused-imports --recursive .

format: format-isort format-black format-unused-imports

#################################################################################
# Linting checks ################################################################

lint-flake8: ## Checks if .py files follow flake8
	@echo "Checking flake8 errors"
	flake8 --max-line-length=88 --extend-ignore=E203,W503

lint-pylint: ## Checks if .py files follow pylint
	@echo "Checking pylint errors"
	pylint .

lint-pylint-with-report-txt: ## Checks if .py files follow pylint and generates pylint-output.txt
	@echo "Checking pylint errors and generating pylint-output.txt"
	set -o pipefail && pylint . | tee pylint-output.txt

check-lint: lint-flake8 lint-pylint ## Checks all linting issues

#################################################################################
# mypy test ##########################################################

test-pre-commit: ## Runs pre-commit tests without committing
	@echo "Running pre-commit tests"
	pre-commit run -a

test-pre-push: ## Runs pre-push tests without pushing
	@echo "Running pre-push tests"
	pre-commit run -a --hook-stage push

test-mypy: ## Runs mypy tests
	@echo "Running mypy tests"
	mypy .

test: test-mypy

check-missing-type:
	@echo "Checking for missing function parameter type and return type"
	mypy . --disallow-untyped-defs
