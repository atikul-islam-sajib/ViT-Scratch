from setuptools import setup, find_packages


def requirements():
    with open("./requirements.txt", "r") as file:
        return file.read().splitlines()


setup(
    name="ViT",
    version="0.0.1",
    description="A deep learning project that is build for Transformer based task using ViT",
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/ViT-Scratch.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    install_requires=requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="TransformerScratch : machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/ViT-Scratch.git/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/ViT-Scratch.git",
        "Source Code": "https://github.com/atikul-islam-sajib/ViT-Scratch.git",
    },
)
