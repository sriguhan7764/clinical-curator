from setuptools import setup, find_packages

setup(
    name="clinical-curator",
    version="1.0.0",
    description="NIH ChestX-ray14 multi-label classification system",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "pillow>=10.0",
        "opencv-python-headless>=4.8",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "python-multipart>=0.0.9",
        "fpdf2>=2.7",
    ],
    extras_require={
        "hpo": ["optuna>=3.5"],
        "dev": ["pytest", "ruff", "mypy"],
    },
)
