import setuptools

from pheres import metadata

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    # Package
    name=metadata.name.capitalize(),
    version=metadata.version,
    packages=setuptools.find_packages(),
    python_requires=f">={metadata.py_version}",
    # Metadata
    author=metadata.author,
    author_email=metadata.email,
    description="Extension to the builtin json module",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=metadata.project_url,
    keywords="json",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)
