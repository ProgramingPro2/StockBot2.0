# Contributing to StockBot2

Thank you for your interest in contributing to StockBot2! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use clear, descriptive titles** for bug reports
3. **Include system information**: OS, Python version, GPU details
4. **Provide reproduction steps** with minimal examples
5. **Include relevant logs** and error messages

### Suggesting Features

1. **Search existing feature requests** first
2. **Explain the use case** and problem it solves
3. **Provide implementation ideas** if you have them
4. **Consider backwards compatibility**

### Code Contributions

#### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/ProgrammingPro2/StockBot2.0.git
cd StockBot2.0

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Test your changes**:

   ```bash
   # Test basic functionality
   python3 main.py --mode test --episodes 3

   # Test training (short)
   python3 main.py --mode train --epochs 5
   ```

3. **Update documentation** if needed

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: New feature description"
   git commit -m "Fix: Bug description"
   git commit -m "Update: Documentation changes"
   ```

#### Pull Request Process

1. **Update README.md** if you've changed functionality
2. **Test functionality** with the commands above
3. **Include description** of changes in PR
4. **Reference related issues** with `Fixes #123`
5. **Request review** from maintainers

## 📝 Code Style Guidelines

### Python Style

- Follow **PEP 8** standards
- Use **Black** for code formatting
- Use **isort** for import sorting
- Maximum line length: **88 characters**

### Documentation

- **Docstrings** for all public functions
- **Type hints** where appropriate
- **Clear variable names**
- **Inline comments** for complex logic

### Example Code Style

```python
def calculate_portfolio_value(
    balance: float,
    shares: Dict[str, float],
    prices: Dict[str, float]
) -> float:
    """
    Calculate total portfolio value including cash and positions.

    Args:
        balance: Available cash balance
        shares: Dictionary of stock symbols to share quantities
        prices: Dictionary of stock symbols to current prices

    Returns:
        Total portfolio value in dollars

    Raises:
        ValueError: If prices are missing for held positions
    """
    total_value = balance

    for symbol, quantity in shares.items():
        if quantity > 0:
            if symbol not in prices:
                raise ValueError(f"Missing price for {symbol}")
            total_value += quantity * prices[symbol]

    return total_value
```

## 🧪 Testing Guidelines

### Manual Testing

Test your changes with these commands:

```bash
# Test basic functionality (safe, no trading)
python3 main.py --mode test --episodes 5

# Test training (short session)
python3 main.py --mode train --epochs 10

# Test optimization (quick)
python3 main.py --mode optimize --trials 5

# Test configuration loading
python3 -c "from config import *; print('Config OK')"
```

## 🔒 Security Guidelines

### API Keys and Secrets

- **Never commit** API keys or secrets
- Use **environment variables** for sensitive data
- Add secrets to `.gitignore`
- Use **paper trading** for development

### Code Security

- **Validate inputs** from external sources
- **Handle errors gracefully** without exposing internals
- **Use secure defaults** in configuration
- **Review dependencies** for vulnerabilities

## 📚 Documentation Standards

### README Updates

- Keep **Quick Start** section current
- Update **feature lists** for new functionality
- Include **examples** for new commands
- Update **troubleshooting** section

### Code Documentation

- **Function docstrings** with Args/Returns/Raises
- **Class docstrings** explaining purpose
- **Module docstrings** for file overview
- **Inline comments** for complex algorithms

### Configuration Documentation

- Document **all config parameters**
- Include **valid value ranges**
- Provide **example configurations**
- Explain **performance implications**

## 🐛 Debugging Guidelines

### Debug Mode

Enable debug logging for development:

```bash
python3 main.py --mode train --epochs 5 --debug
```

### Common Issues

1. **GPU Memory**: Reduce batch size
2. **Training Slow**: Check GPU utilization
3. **API Errors**: Verify credentials and limits
4. **Process Crashes**: Check system resources

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

# Good logging examples
logger.info(f"Starting training with {episodes} episodes")
logger.warning(f"Low memory detected: {memory_gb:.1f}GB available")
logger.error(f"Failed to load model: {e}")
logger.debug(f"Portfolio value: ${portfolio_value:.2f}")
```

## 🚀 Performance Guidelines

### Optimization Priorities

1. **Correctness first**: Ensure accuracy before optimizing
2. **Profile before optimizing**: Measure actual bottlenecks
3. **GPU utilization**: Maximize hardware usage
4. **Memory efficiency**: Minimize memory footprint

### Performance Testing

```python
import time
import psutil

def benchmark_function():
    """Benchmark critical functions for performance."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Your function here
    result = expensive_function()

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Memory: {(end_memory - start_memory) / 1024 / 1024:.1f}MB")

    return result
```

## 📞 Getting Help

### Before Asking for Help

1. **Check documentation** and README
2. **Search existing issues** and discussions
3. **Try debugging steps** listed above
4. **Prepare minimal reproduction** example

### Where to Ask

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Review**: Pull request feedback

### Providing Context

Include in your help request:

- **System information**: OS, Python version, GPU
- **Configuration**: Relevant config parameters
- **Steps to reproduce**: Minimal example
- **Expected vs actual**: What should happen vs what does
- **Logs and errors**: Relevant error messages

## 🎯 Project Goals

### Primary Objectives

- **Educational Value**: Help people learn algorithmic trading
- **Production Ready**: Reliable enough for real use
- **Performance**: Efficient use of modern hardware
- **Safety**: Protect users from financial losses

### Code Quality Standards

- **Maintainable**: Easy to understand and modify
- **Testable**: Comprehensive test coverage
- **Documented**: Clear documentation and examples
- **Robust**: Graceful error handling and recovery

Thank you for contributing to StockBot2! 🚀
