.PHONY: help install tidy lint tests build release-test release

help: Makefile
	@printf "Note: this Makefile is intended for development. Use pip to install pheres as a user\n\n"
	@printf "Available targets:\n"
	@printf "  setup        : setup env: install depencencies, and then pheres in editable mode.\n"
	@printf "                 Only run once.\n"
	@printf "  tidy         : tidy up the source code with isort and black\n"
	@printf "  lint         : lint and typecheck the code with pylint and mypy\n"
	@printf "  tests        : typecheck and test the source code\n"
	@printf "  build        : builds distribution archives\n"
	@printf "  release-test : release on testpypi (test server)\n"
	@printf "  release      : release on pypi\n"

setup: Makefile
	pip install -U -r requirements.txt
	pip install -U -e .

tidy: Makefile
	isort --profile=black src tests
	black src tests

lint: Makefile
	pylint src tests
	mypy --show-error-code src tests

tests: Makefile
#   mypy src tests
#   pytest -v --hypothesis-show-statistics
	pytest -v

build: Makefile test
	python setup.py sdist bdist_wheel

release-test: Makefile build
	python -m twine upload --repository testpypi dist/*

release: Makefile build
	python -m twine upload dist/*