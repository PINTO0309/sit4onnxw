#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("sit4onnxw/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = [\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(
    name="sit4onnxw",
    version=version,
    author="Katsuya Hyodo",
    author_email="",
    description="Simple Inference Test for ONNX Runtime Web",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PINTO0309/sit4onnxw",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
        "onnx>=1.12.0",
        "click>=8.0.0",
        "requests>=2.25.0",
        "selenium>=4.0.0",
        "webdriver-manager>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "sit4onnxw=sit4onnxw.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)