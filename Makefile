.PHONY: help install tidy lint tests build release-test release

help: Makefile
	@printf "This makefile is intended for development\n\n"
	@printf "Available targets:\n"
	@printf "  install : install depencencies; then, pheres in editable mode\n"
	@printf "  tidy    : tidy up the source source code\n"
	@printf "  test    : typecheck and test source code\n"
	@printf "  build   : builds distribution archives\n"

install: Makefile
	pip install -U -r requirements.txt
	pip install -U -e .

tidy: Makefile
	isort src tests
	black src tests

lint: Makefile
	pylint src/pheres tests
	#mypy src tests

tests: Makefile
	#mypy src tests
	#pytest -v --hypothesis-show-statistics
	pytest

build: Makefile test
	python setup.py sdist bdist_wheel

release-test: Makefile build
	python -m twine upload --repository testpypi dist/*

release: Makefile build
	python -m twine upload dist/*