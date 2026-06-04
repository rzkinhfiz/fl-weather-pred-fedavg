# Contributing to FL-Weather

First off, thank you for considering contributing to the FL-Weather project! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**
* **Include your environment details:**
  ```bash
  python --version
  pip list | grep flwr
  pip list | grep torch
  nvidia-smi  # If using GPU
  ```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and the proposed behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Python style guide (see below)
* End all files with a newline
* Avoid platform-dependent code
* Document new code with docstrings
* Add tests for new functionality

---

## Development Setup

### 1. Fork & Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/flweatherpred.git
cd flweatherpred
git remote add upstream https://github.com/ORIGINAL_OWNER/flweatherpred.git
```

### 2. Create Development Environment

```bash
# Create conda environment
conda create -n fl-weather-dev python=3.11
conda activate fl-weather-dev

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy isort
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

### 4. Make Changes

Follow the style guide (see below) and write tests for your changes.

### 5. Test Your Changes

```bash
# Run unit tests
python -m pytest

# Check code style
black src/ --check
flake8 src/
mypy src/

# Or use the combined script:
bash scripts/lint.sh  # If available
```

### 6. Format Code

```bash
# Auto-format with Black
black src/

# Sort imports
isort src/

# Check and fix with flake8
flake8 src/ --show-source
```

### 7. Commit & Push

```bash
# Create meaningful commits
git add .
git commit -m "Add feature: meaningful description"

# Push to your fork
git push origin feature/your-feature-name
```

### 8. Create Pull Request

On GitHub:
1. Create pull request from your fork to main repository
2. Fill in the PR template
3. Reference related issues (#123)
4. Wait for review and address feedback

---

## Style Guide

### Python Code Style

We use **Black** for code formatting and follow **PEP 8**:

```python
# ✓ Good
def calculate_mse(predictions, targets):
    """Calculate mean squared error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        MSE loss value
    """
    import numpy as np
    return np.mean((predictions - targets) ** 2)


# ✗ Bad
def calc_mse(pred,tgt):
    return np.mean((pred-tgt)**2)
```

### Naming Conventions

```python
# Classes: PascalCase
class WeatherLSTM:
    pass

# Functions/methods: snake_case
def prepare_data(filepath):
    pass

# Constants: UPPER_SNAKE_CASE
BATCH_SIZE = 32
NUM_CLIENTS = 15

# Private methods: _leading_underscore
def _validate_input(data):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(model, dataloader, num_epochs):
    """Train the LSTM model.
    
    Args:
        model: PyTorch LSTM model
        dataloader: DataLoader with training data
        num_epochs: Number of training epochs
        
    Returns:
        Trained model with updated weights
        
    Raises:
        ValueError: If num_epochs <= 0
        TypeError: If model is not a torch.nn.Module
        
    Example:
        >>> model = WeatherLSTM()
        >>> trained_model = train_model(model, train_loader, 5)
    """
    pass
```

### Import Organization

```python
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party libraries
import numpy as np
import torch
import pandas as pd

# 3. Local modules
from src.config import BATCH_SIZE
from src.utils import prepare_data
```

Sort with `isort`:
```bash
isort src/
```

---

## Testing Guidelines

### Write Tests

Place tests in a `tests/` directory:

```python
# tests/test_model.py
import pytest
import torch
from src.model import WeatherLSTM


def test_lstm_initialization():
    """Test LSTM model initialization."""
    model = WeatherLSTM()
    assert model is not None
    
    
def test_lstm_forward_pass():
    """Test LSTM forward pass with sample input."""
    model = WeatherLSTM()
    input_tensor = torch.randn(32, 5, 4)  # batch, seq, features
    
    with torch.no_grad():
        output = model(input_tensor)
        
    assert output.shape == (32, 1)
    

def test_lstm_parameter_extraction():
    """Test parameter extraction for federated learning."""
    model = WeatherLSTM()
    params = model.get_model_parameters()
    
    assert isinstance(params, list)
    assert len(params) > 0
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/
```

---

## Documentation

### Update Documentation

If you're adding features, update relevant documentation:

1. **Docstrings**: Add/update function docstrings
2. **README.md**: Update if user-facing changes
3. **Docs**: Update in [docs/](docs/) if architectural changes
4. **CHANGELOG**: Add entry (if exists)

### Documentation Style

```markdown
# Clear Heading

Short description of the feature.

## Subheading

Detailed explanation with examples:

\`\`\`python
# Code example
model = WeatherLSTM()
\`\`\`

### Key Points

- Point 1
- Point 2
- Point 3
```

---

## Commit Messages

Use clear, descriptive commit messages:

```bash
# ✓ Good
git commit -m "Add non-IID data analysis to evaluation module"
git commit -m "Fix GPU memory leak in training loop"
git commit -m "Improve LSTM convergence with gradient clipping"

# ✗ Bad
git commit -m "fix stuff"
git commit -m "update code"
git commit -m "changes"
```

### Commit Message Format

```
Type: Brief description (50 chars max)

Detailed explanation if needed (72 char line limit).
Explain the problem and the solution.

Closes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests

---

## Issue Labels

When submitting issues or PRs, we use these labels:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: This will not be worked on

---

## Additional Notes

### Project Principles

1. **Privacy First**: Changes must maintain data privacy guarantees
2. **Performance**: Consider impact on training time and memory
3. **Compatibility**: Support Python 3.10+ and PyTorch 2.1+
4. **Testing**: All changes must have corresponding tests
5. **Documentation**: All public APIs must be documented

### Areas for Contribution

We're actively seeking contributions in:

- **Performance Optimization**: Improve training speed
- **New Federated Algorithms**: Implement FedProx, FedAdam, etc.
- **Scalability**: Support for 100+ clients
- **Visualization**: Better training metrics display
- **Documentation**: Clarify existing docs, add examples
- **Testing**: Improve test coverage
- **Deployment**: Docker optimization, Kubernetes support

---

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Project README (major contributions)
- Release notes (significant changes)

---

## Questions?

Feel free to open an issue for any questions or reach out to the maintainers.

Thank you for contributing! 🎉

---

**Last Updated**: June 4, 2026  
**Version**: 1.0
