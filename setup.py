from setuptools import setup, find_packages

setup(
    name="DistillNLP",
    version="1.0.0",
    description="Ein professionelles Projekt zur Implementierung von Knowledge Distillation für Textklassifikation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Benjamin",
    author_email="your email",
    url="https://github.com/Benjamin2099/DistillNLP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.2",
        "scipy>=1.7.1",
        "nltk>=3.6.5",
        "spacy>=3.2.0",
        "transformers>=4.15.0",
        "sentencepiece>=0.1.96",
        "pandas>=1.3.3",
        "pyyaml>=5.4.1",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "flask>=2.0.2",
        "flask-restful>=0.3.9",
        "loguru>=0.5.3",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train_teacher=src.training:train_teacher_main",
            "train_student=src.training:train_student_main",
            "start_api=app:run_api"
        ],
    },
)
