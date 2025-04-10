from setuptools import setup, find_packages

setup(
    name="whisper-batch",
    version="1.0.0",
    description="A robust tool for batch processing and transcribing audio and video files using OpenAI's Whisper model",
    author="WhisperBatch Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/whisper-batch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "openai-whisper>=20230918",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pathlib>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.1.0",
            "flake8>=4.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "whisper-batch=whisper_batch.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="whisper, transcription, audio, video, batch, gpu, cuda, speech-to-text",
) 