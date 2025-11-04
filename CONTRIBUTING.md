# Contributing to UltraOptimiser

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/UltraOptimiser.git`
3. Create a virtual environment: `python -m venv venv`
4. Install in development mode: `pip install -e .[dev]`

## Code Style

- Use Black for formatting: `black .`
- Use flake8 for linting: `flake8 .`
- Use mypy for type checking: `mypy core/`

## Testing

Run tests with: `pytest tests/ -v`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Push to your fork
6. Create a Pull Request

## Code Review Criteria

- All tests pass
- Code coverage > 80%
- No flake8 errors
- Type hints included
- Documentation updated
