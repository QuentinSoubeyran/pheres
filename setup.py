import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    # Package
    name="pheres",
    version="1.0a2",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    # Metadata
    author="Quentin Soubeyran",
    license="MIT",
    author_email="45202794+QuentinSoubeyran@users.noreply.github.com",
    description="Extension to the builtin json module",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/QuentinSoubeyran/pheres",
    keywords="json",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Typing :: Typed"
    ],
)
