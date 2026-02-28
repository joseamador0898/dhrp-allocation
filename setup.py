from setuptools import setup, find_packages

setup(
    name="llm-dhrp",
    version="1.0.0",
    description="Language-Informed Differentiable Hierarchical Risk Parity",
    author="Jose L. Amador",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "torch",
        "cvxpy",
        "requests",
        "pandas-datareader",
        "python-dotenv",
        "fredapi",
        "seaborn",
        "matplotlib",
    ],
    extras_require={
        "llm": ["transformers>=4.46", "tokenizers>=0.20", "huggingface-hub>=0.26"],
    },
)
