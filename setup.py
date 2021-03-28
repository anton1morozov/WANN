from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

setup(
    name="wann",
    version="0.1.0",
    packages=find_packages(),
    install_requires=lines
)
