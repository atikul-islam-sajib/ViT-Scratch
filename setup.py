from setuptools import setup, find_packages

setup(
    name="ViT",
    version="0.0.1",
    description="A deep learning project for Transformer-based tasks using ViT",
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/ViT-Scratch",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.2",
        "pandas==1.3.3",
        "torch==1.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="TransformerScratch, machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/ViT-Scratch/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/ViT-Scratch",
        "Source Code": "https://github.com/atikul-islam-sajib/ViT-Scratch",
    },
)
