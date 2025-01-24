import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CGCoT",
    version="0.0.1",
    author="Patrick Y. Wu, Jonathan Nagler, Joshua A. Tucker, and Solomon Messing",
    author_email="patrickwu@american.edu",
    description="Package for Concept-Guided Chain-of-Thought",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SMAPPNYU/CGCoT",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)