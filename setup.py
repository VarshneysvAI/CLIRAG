import os
from setuptools import setup, find_packages

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="clirag",
    version="0.1.0",
    author="Elite Systems Engineer",
    description="CLI Retrieval-Augmented Generation (CLIRAG) - 100% Offline Edge AI Analysis Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clirag",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "duckdb>=0.10.0",
        "kuzu>=0.0.11",
        "pymupdf>=1.23.0",
        "reportlab>=4.0.0", # Added for stress testing script
        # "llama-cpp-python>=0.2.0",  # Requires C++ build tools, omit for standard mock installation
        # "spacy",
        # "gliner"
    ],
    entry_points={
        "console_scripts": [
            "clirag=clirag.main:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
)
