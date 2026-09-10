from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MULTIMODAL-DEPRESSION",
    version="0.1",
    author="Annisaa",
    packages=find_packages(),
    install_requires = requirements,
)