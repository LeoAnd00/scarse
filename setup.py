from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent.resolve()

with open(HERE / "README.md", "r", encoding="utf-8") as f:
    description = f.read()

install_requires = [
    l.strip() for l in (HERE / "requirements.txt").read_text("utf-8").splitlines()
    if l.strip() and not l.startswith("--")
]

setup(
    name="scarse",
    version="1.0.0",
    author="Leo Andrekson, Robin Rydbergh, Rocío Mercado, Michaela Wenzel",
    author_email="leo.andrekson@chalmers.se, robin.rydbergh@chalmers.se, rocom@chalmers.se, wenzelm@chalmers.se",
    description="A package for training SCARSE on small sample sizes of peptide sequences that can later be used to predict peptide properties of unseen peptides. Making SCARSE perfectly suited for AI-infused peptide engineering.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/LeoAnd00/scarse",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=install_requires,
)