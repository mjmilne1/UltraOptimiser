"""
UltraOptimiser - Advanced Portfolio Optimization Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ultraoptimiser",
    version="2.0.0",
    author="mjmilne1",
    description="Advanced portfolio optimization with multi-objective Lagrangian optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjmilne1/UltraOptimiser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",# Create setup.py for pip installation
$setupPy = @'
"""
UltraOptimiser - Advanced Portfolio Optimization Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ultraoptimiser",
    version="2.0.0",
    author="mjmilne1",
    description="Advanced portfolio optimization with multi-objective Lagrangian optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjmilne1/UltraOptimiser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "yfinance>=0.2.28",
        "plotly>=5.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ultraoptimiser=core.optimizer_v2:main",
        ],
    },
)
