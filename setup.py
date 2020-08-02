import setuptools

from pheres.metadata import (
    __name__ as name,
    __version__ as version,
    py_version,
    __author__ as author,
    project_url
)

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    # Package
    name=name.capitalize(),
    version=version,
    packages=setuptools.find_packages(),
    python_requires=f">={py_version}",

    # Metadata
    author=author,
    author_email="45202794+QuentinSoubeyran@users.noreply.github.com",
    description="Extension to the builtin json module",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=project_url,
    keywords="json",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ]
)