#!/usr/bin/env python3
"""
Setup script for StockBot2
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stockbot2",
    version="2.0.0",
    author="ProgramingPro2",
    description="A Deep Reinforcement Learning Trading Bot",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ProgramingPro2/StockBot2.0",
    project_urls={
        "Bug Reports": "https://github.com/ProgramingPro2/StockBot2.0/issues",
        "Source": "https://github.com/ProgramingPro2/StockBot2.0",
        "Documentation": "https://github.com/ProgramingPro2/StockBot2.0/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stockbot-automation=market_automation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="trading, reinforcement-learning, deep-learning, finance, alpaca, stocks",
)
