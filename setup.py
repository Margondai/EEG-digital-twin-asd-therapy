#!/usr/bin/env python3
"""
Setup script for EEG-Driven Digital Twin Models for ASD Language Therapy
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [req for req in requirements if not req.startswith('#') and req.strip()]

setup(
    name="eeg-digital-twin-asd-therapy",
    version="1.0.0",
    description="EEG-Driven Digital Twin Models for AI and VR-Based Language Therapy in Children with Autism Spectrum Disorder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Nikita Islam, Ancuta Margondai",
    author_email="Ni836085@ucf.edu, Ancuta.Margondai@ucf.edu",
    maintainer="Nikita Islam",
    maintainer_email="Ni836085@ucf.edu",
    
    # Project URLs
    url="https://github.com/yourusername/eeg-digital-twin-asd-therapy",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/eeg-digital-twin-asd-therapy/issues",
        "Source": "https://github.com/yourusername/eeg-digital-twin-asd-therapy",
        "Documentation": "https://github.com/yourusername/eeg-digital-twin-asd-therapy/docs",
        "Conference Paper": "https://modsimworld.org/papers/2025/",
    },
    
    # Package information
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Additional package data
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    include_package_data=True,
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "eeg-asd-therapy=main:main",
            "train-digital-twin=scripts.train_model:main",
            "run-therapy-simulation=scripts.run_simulation:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        "EEG", "digital-twin", "autism", "ASD", "language-therapy", 
        "virtual-reality", "VR", "artificial-intelligence", "AI",
        "neurofeedback", "neuroadaptive", "reinforcement-learning",
        "biomedical-engineering", "human-computer-interaction"
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "ipywidgets>=7.6.0",
        ],
        "advanced": [
            "opencv-python>=4.5.0",
            "pyaudio>=0.2.11",
            "pynput>=1.7.6",
        ]
    },
    
    # Project status
    zip_safe=False,
)
