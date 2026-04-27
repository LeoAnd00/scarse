from pathlib import Path
import sys
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

install_requires=[
    l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    if l.strip() and not l.startswith("--")  # Ignore empty lines and options like --extra-index-url
]

setup(
    name="scarse",
    version="0.0.1",
    author="Leo Andrekson, Rocío Mercado, Robin Rydbergh, Michaela Wenzel",
    author_email="leo.andrekson@chalmers.se, rocom@chalmers.se, robin.rydbergh@chalmers.se, wenzelm@chalmers.se",
    description="A package for training SCARSE on small sample sizes of peptide sequences that can later be used to predict peptide properties of unseen peptides. Making SCARSE perfectly suited for AI-infused peptide engineering.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/LeoAnd00/scarse",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.12.10",
    packages=find_packages(),
    install_requires=install_requires,
)