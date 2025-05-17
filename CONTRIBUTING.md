# Contributing to AlgebraicOptimization.py

We welcome contributions to AlgebraicOptimization.py! Here's how you can help:

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. When reporting a bug, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your Python version and operating system

## Development Setup

1. Fork the repository and clone it locally
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Code Style

We use `black` for code formatting and `flake8` for linting. Before submitting a pull request, please run:

```bash
black .
flake8
```

## Testing

Run the test suite with:

```bash
pytest
```

We aim for good test coverage. Please add tests for any new functionality.

## Pull Requests

1. Create a new branch for your changes
2. Make your changes and ensure tests pass
3. Update documentation as needed
4. Submit a pull request with a clear description of your changes

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.
