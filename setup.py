from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from package without importing
about = {}
version_path = Path("src/whisper_batch/__init__.py")
with open(version_path, "r", encoding="utf-8") as f:
    exec(f.read(), about)

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper-batch-enhanced",
    version=about["__version__"],
    description="Batch transcription tool using OpenAI Whisper with enhanced CUDA error handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__email__"],
    url="https://github.com/always-tinkering/whisper-batch-enhanced",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "openai-whisper>=20230314",
        "ffmpeg-python>=0.2.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "whisper-batch=whisper_batch.main:main",
        ],
    },
)