from setuptools import setup, find_packages

setup(
    name="algebraic_optimization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "sympy>=1.9"
    ],
    author="",
    description="Python implementation of Algebraic Optimization",
    license="MIT",
    python_requires='>=3.8'
)
