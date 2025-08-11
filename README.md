# 🤖 StockBot2 - Intelligent Automated Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Alpaca](https://img.shields.io/badge/Alpaca-API-blue.svg)](https://alpaca.markets)

A sophisticated deep reinforcement learning system for automated stock trading with intelligent market cycle management, GPU acceleration, and comprehensive automation features.

## ✨ Key Features

### 🎯 **Intelligent Automation**

- **Market-Aware Operation**: Automatically detects market hours and adjusts behavior
- **Weekend Training Cycles**: Continuous model improvement during off-hours
- **Friday Optimization**: Hyperparameter tuning after market close
- **Systemd Integration**: Boot-on-startup with automatic restart capabilities
- **Health Monitoring**: Real-time system resource tracking and alerts

### 🧠 **Advanced AI**

- **Deep Reinforcement Learning**: DDPG (Deep Deterministic Policy Gradients) agent
- **GPU Acceleration**: CUDA-optimized with automatic CPU fallback
- **Experience Replay**: Sophisticated memory management for stable learning
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Mixed Precision Training**: Optimized for performance and memory efficiency

### 📊 **Professional Trading**

- **Live Trading**: Alpaca Markets API integration
- **Paper Trading**: Safe testing environment
- **Multi-Asset Support**: Configurable stock symbol lists (15+ stocks)
- **Risk Management**: Built-in position limits and safety checks
- **Historical Data**: Yahoo Finance integration for training data

### 🔧 **Enterprise Features**

- **Comprehensive Logging**: Rotating logs with performance metrics
- **Error Recovery**: Robust failure handling and automatic restart
- **Clean Code Architecture**: Modular, maintainable design
- **Configuration Management**: Flexible config system with validation
- **Performance Monitoring**: Real-time training and trading metrics

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (tested up to 3.12)
- **Git**
- **Alpaca Markets account** (for live trading)
- **CUDA-compatible GPU** (optional, CPU supported)
- **Linux/Unix system** (for automation features)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ProgramingPro2/StockBot2.0.git
   cd StockBot2-Experimental
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Alpaca API keys
   ```

### Quick Start Commands

#### Training Mode

```bash
python main.py --mode train --epochs 100
```

#### Optimization Mode

```bash
python main.py --mode optimize --trials 20
```

#### Testing Mode

```bash
python main.py --mode test --episodes 10
```

#### Live Trading (Paper)

```bash
python main.py --mode live
```

#### Automation System

```bash
python market_automation.py
```

**Note**: The `--mode` argument is required. Available modes: `train`, `test`, `live`, `optimize`

### Automation Setup

1. **Install as a service**

   ```bash
   sudo bash scripts/install_service.sh
   ```

2. **Enable and start**

   ```bash
   sudo systemctl enable stockbot-automation
   sudo systemctl start stockbot-automation
   ```

3. **Monitor the system**

   ```bash
   # Check logs
   journalctl -u stockbot-automation -f

   # View status
   sudo systemctl status stockbot-automation
   ```

## 📁 Project Structure

```
StockBot2-Experimental/
├── src/
│   └── stockbot/                 # Main package
│       ├── __init__.py           # Package initialization
│       ├── drl_agent.py          # Deep RL agent implementation (18KB)
│       ├── stock_env.py          # Trading environment (51KB)
│       ├── alpaca_trade_system.py # Alpaca trading integration (11KB)
│       ├── performance_monitor.py # Performance monitoring (5.4KB)
│       ├── replay_buffer.py      # Experience replay buffer (11KB)
│       ├── noise.py              # Action noise implementation (1.2KB)
│       ├── utils.py              # Utility functions (3.9KB)
│       └── system_utils.py       # System utilities (10KB)
├── docs/                         # Documentation
│   ├── AUTOMATION_GUIDE.md       # Automation setup guide (4.5KB)
│   └── CONTRIBUTING.md           # Contribution guidelines (7KB)
├── scripts/                      # Installation and setup scripts
│   ├── install_service.sh        # Service installation (2.9KB)
│   ├── stockbot-automation.service # Systemd service file (1KB)
│   └── start_market_automation.sh # Startup script (223B)
├── main.py                       # Main entry point (35KB)
├── market_automation.py          # Automation system (25KB)
├── config.py                     # Configuration management (6.9KB)
├── requirements.txt              # Python dependencies (15 packages)
├── setup.py                      # Package setup (2.3KB)
├── README.md                     # Project overview
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── env.example                   # Environment variables template
├── stocks.json                   # Stock symbols configuration (15 stocks, overrides config.py)
├── hyperparameters.json          # Optimized hyperparameters
├── app.log                       # Application logs
├── final_values_distribution.png # Training visualization
├── test_results.png              # Test results visualization
├── data_cache/                   # Cached market data (git-ignored)
├── models/                       # Trained model files (git-ignored)
├── checkpoints/                  # Training checkpoints (git-ignored)
├── logs/                         # Log files (git-ignored)
└── .venv/                        # Virtual environment (git-ignored)
```

### Core Components

#### `src/stockbot/` - Main Package

- **`drl_agent.py`**: Deep reinforcement learning agent with DDPG algorithm
- **`stock_env.py`**: Trading environment with market simulation
- **`alpaca_trade_system.py`**: Real trading system integration
- **`performance_monitor.py`**: Training and trading metrics tracking
- **`replay_buffer.py`**: Experience replay with GPU tensor support
- **`noise.py`**: Ornstein-Uhlenbeck noise for exploration
- **`utils.py`**: Helper functions and SSL configuration
- **`system_utils.py`**: System health checks and validation

#### Entry Points

- **`main.py`**: Main application with training, testing, and live trading modes
- **`market_automation.py`**: Automated trading system with market-aware scheduling

#### Configuration Files

- **`config.py`**: Main configuration with validation and defaults
- **`stocks.json`**: List of 15 stocks to trade (overrides config.py defaults)
- **`hyperparameters.json`**: Optimized hyperparameters from training (auto-loaded)
- **`env.example`**: Template for environment variables

#### Scripts

- **`install_service.sh`**: Systemd service installation
- **`stockbot-automation.service`**: Systemd service definition
- **`start_market_automation.sh`**: Startup script

#### Documentation

- **`AUTOMATION_GUIDE.md`**: Detailed automation setup instructions
- **`CONTRIBUTING.md`**: Development and contribution guidelines

## ⚙️ Configuration

### Basic Configuration

Edit `config.py` to customize:

```python
# Trading parameters
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "TSLA", "NVDA", "AMD"]  # Default in config.py
# Note: stocks.json overrides this with 15 stocks
INITIAL_BALANCE = 1000.0
LOOKBACK_WINDOW = 168
TRADING_FREQUENCY = "hourly"
MAX_SHARES_PER_TRADE = 5
ALLOW_SHORT_SELLING = False

# Training parameters
TRAINING_EPISODES = 500
TEST_EPISODES = 50
REWARD_SCALING_FACTOR = 0.01
```

### API Setup (Optional - for live trading)

Create `.env` file:

```bash
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADING=true  # Use paper trading for safety
```

**Note**: API keys are optional for training/testing (uses Yahoo Finance data). Required only for live trading.

### Optimized Hyperparameters

The system includes pre-optimized hyperparameters in `hyperparameters.json` (auto-loaded):

```json
{
  "learning_rate": 5.46e-5,
  "batch_size": 1024,
  "gamma": 0.951,
  "noise_sigma": 0.187,
  "noise_theta": 0.129,
  "reward_scaling_factor": 29.36
}
```

**Note**: These values override the defaults in `config.py` when the file exists.

## 🧪 Testing & Development

### Quick Test

```bash
# Test the system (simulated trading, no real money)
python main.py --mode test --episodes 5

# Train a model (simulated trading with historical data)
python main.py --mode train --epochs 50

# Optimize hyperparameters
python main.py --mode optimize --trials 10

# Run with time limits instead of episode counts
python main.py --mode train --time-limit 2.0  # 2 hours
python main.py --mode test --time-limit 0.5   # 30 minutes
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run with console scripts
stockbot-automation

# Run automation directly
python market_automation.py
```

### System Health Check

```bash
# Validate configuration
python -c "from config import validate_config; validate_config()"

# Check system resources
python -c "from src.stockbot.system_utils import comprehensive_system_check; comprehensive_system_check()"

# Force GPU usage (will error if not available)
python main.py --mode test --episodes 1 --gpu
```

## 📊 Performance & Monitoring

### Training Performance

- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Memory Management**: Efficient replay buffer with GPU tensors
- **Progress Tracking**: Real-time training metrics and visualization
- **Checkpointing**: Automatic model saving and recovery
- **Mixed Precision**: Optimized for performance and memory efficiency

### System Monitoring

- **Health Checks**: CPU, memory, and GPU usage monitoring
- **Logging**: Comprehensive logging with rotation (10MB max, 5 backups)
- **Error Recovery**: Automatic restart on failures
- **Performance Metrics**: Training and trading performance tracking

### Visualization

- **Training Progress**: Real-time loss and reward plots
- **Test Results**: Performance visualization and analysis
- **Value Distribution**: Final model value function analysis

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

### Development Tools

- **Ruff**: Code linting and formatting
- **Type Hints**: Comprehensive type annotations
- **Docstrings**: Detailed documentation
- **Error Handling**: Robust exception management

## 📚 Documentation

- **[Automation Guide](docs/AUTOMATION_GUIDE.md)**: Detailed setup for automated trading
- **[Contributing Guidelines](docs/CONTRIBUTING.md)**: How to contribute to the project
- **Code Documentation**: Comprehensive docstrings and type hints

## 🔒 Security & Safety

### Paper Trading First

- Always start with paper trading (`ALPACA_PAPER_TRADING=true`)
- Test thoroughly before live trading
- Monitor performance closely

### Risk Management

- Built-in position limits (5 shares per trade by default)
- Maximum shares per trade restrictions
- Short selling disabled by default (`ALLOW_SHORT_SELLING = False`)
- Comprehensive error handling and validation
- Paper trading mode for safe testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real money.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ProgramingPro2/StockBot2.0/issues)
- **Documentation**: Check the [docs/](docs/) directory
- **Logs**: Check `app.log` and `logs/` directory for detailed information

## 📈 Performance Metrics

The system includes comprehensive performance monitoring:

- **Training Metrics**: Loss curves, reward progression, episode statistics
- **Trading Metrics**: Portfolio value, returns, Sharpe ratio, drawdown
- **System Metrics**: CPU/GPU usage, memory consumption, response times
- **Market Metrics**: Position tracking, trade execution, slippage analysis

## 🔄 Automation Features

### Market-Aware Scheduling

- **Market Hours**: 9:30 AM - 4:00 PM EST/EDT
- **Weekend Training**: Continuous model improvement during off-hours
- **Friday Optimization**: Post-market hyperparameter tuning
- **Health Monitoring**: Automatic restart on failures
- **Time-Based Limits**: Configurable time limits for training/optimization

### Service Integration

- **Systemd Service**: Boot-on-startup capability
- **Log Rotation**: Automatic log management
- **Process Monitoring**: Health checks and recovery
- **Resource Management**: Memory and CPU optimization
