from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive_gap_splitter",
    version="0.1.0",
    author="Tobias Canavesi",
    author_email="tobiascanavesi@gmail.com",
    description="A package for adaptive gap-based data splitting using percentile thresholds.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobiascanavesi/adaptive_gap_splitter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
        ],
    },
)