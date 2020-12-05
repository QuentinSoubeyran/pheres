# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=-a -c . -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOCDIR		  = ../docs
_ADDITIONAL_TARGETS =Additional targets:\n  \033[94m%-11s\033[0m to make HTML documentation in the dir used by github ($(DOCDIR))\n  \033[94m%-11s\033[0m to clean the github documentation

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@printf "$(_ADDITIONAL_TARGETS)" github github-clean

.PHONY: help Makefile github-clean github

github-clean:
	rm -rf "$(DOCDIR)"

github: github-clean html
	cp -rT "$(BUILDDIR)/html/" "$(DOCDIR)/"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
