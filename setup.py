from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algebraic_optimization_py",
    version="0.1.0",
    packages=find_packages(include=['algebraic_optimization_py*']),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "sympy>=1.9"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Python implementation of Algebraic Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AlgebraicOptimization.py",
    license="MIT",
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.yaml', '*.yml'],
    },
)
