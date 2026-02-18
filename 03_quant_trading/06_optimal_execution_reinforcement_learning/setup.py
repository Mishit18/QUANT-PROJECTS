"""Setup configuration for optimal execution package."""

from setuptools import setup, find_packages

setup(
    name='optimal-execution-rl',
    version='1.0.0',
    description='Optimal execution with Almgren-Chriss and constrained RL',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'gymnasium>=0.29.0',
        'torch>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
)
