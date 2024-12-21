
# Contributing to RDQN-vectorized

1. Fork the repository
2. Clone your fork locally
3. Set up your development environment using either conda or Docker

### Setting up the Development Environment

#### Using Conda
```bash
conda env create -f environment.yml
conda activate dqn_env
```

#### Using Docker
```bash
docker build -t rdqn-vectorized .
docker run --gpus all rdqn-vectorized
```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing
Before submitting a pull request:
1. Run any existing tests
2. Add tests for new functionality
3. Ensure all tests pass

### Pull Request Process

1. Create a new branch for your feature/fix
2. Make your changes
3. Update documentation if needed
4. Submit a pull request with a clear description of changes

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add feature" or "Fix bug")
- Reference issues if applicable

## Project Structure

```
RDQN-vectorized/
├── modules/           # Core implementation modules
├── assets/           # Media assets
├── logs/             # Training logs
└── experiments/      # Experiment results
```

## Areas for Contribution
- Performance optimizations
- Documentation improvements
- Bug fixes
- New features from the to-do list in README.md

## Questions or Need Help?
Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
