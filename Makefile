

.PHONY: tidy tests

tidy: Makefile
	isort src tests
	black src tests

tests: Makefile tidy
	mypy src tests
	pytest
