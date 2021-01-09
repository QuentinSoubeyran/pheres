.PHONY: help install tidy lint tests build release-test release

help: Makefile
	@printf "Note: this Makefile is intended for development. Use pip to install pheres as a user\n\n"
	@printf "Available targets:\n"
	@printf "  setup        : install depencencies, and then pheres in editable mode\n"
	@printf "  tidy         : tidy up the source code\n"
	@printf "  tests        : typecheck and test the source code\n"
	@printf "  build        : builds distribution archives\n"
	@printf "  release-test : release on testpypi (test server)\n"
	@printf "  release      : release on pypi\n"

setup: Makefile
	pip install -U -r requirements.txt
	pip install -U -e .

tidy: Makefile
	isort src tests
	black src tests

lint: Makefile
	pylint src/pheres tests
#   mypy src tests

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