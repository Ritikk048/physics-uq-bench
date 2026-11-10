from setuptools import setup, find_packages

setup(
    name="physics-uq-bench",
    version="0.1.0",
    author="Ritik Kumar",
    author_email="hrithikk048@gmail.com",
    description="Benchmark suite for uncertainty quantification methods on physics-driven tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ritikkumar/physics-uq-bench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "gpytorch>=1.11",
        "optuna>=3.5.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "mcmc": ["pymc>=5.9.0", "arviz>=0.17.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
