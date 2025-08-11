# 🤖 StockBot2 - Intelligent Automated Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

A sophisticated deep reinforcement learning system for automated stock trading with intelligent market cycle management, GPU acceleration, and comprehensive automation features.

## ✨ Key Features

### 🎯 **Intelligent Automation**

- **Market-Aware Operation**: Automatically detects market hours and adjusts behavior
- **Weekend Training Cycles**: Continuous model improvement during off-hours
- **Friday Optimization**: Hyperparameter tuning after market close
- **Systemd Integration**: Boot-on-startup with automatic restart capabilities

### 🧠 **Advanced AI**

- **Deep Reinforcement Learning**: DDPG (Deep Deterministic Policy Gradients) agent
- **GPU Acceleration**: CUDA-optimized with automatic CPU fallback
- **Experience Replay**: Sophisticated memory management for stable learning
- **Hyperparameter Optimization**: Automated tuning using Optuna

### 📊 **Professional Trading**

- **Live Trading**: Alpaca Markets API integration
- **Paper Trading**: Safe testing environment
- **Multi-Asset Support**: Configurable stock symbol lists
- **Risk Management**: Built-in position limits and safety checks

### 🔧 **Enterprise Features**

- **Comprehensive Logging**: Rotating logs with performance metrics
- **Health Monitoring**: System resource tracking and alerts
- **Clean Code Architecture**: Modular, maintainable design
- **Error Recovery**: Robust failure handling and automatic restart

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Alpaca Markets account (for live trading)
- CUDA-compatible GPU (optional, CPU supported)

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
│       ├── drl_agent.py          # Deep RL agent implementation
│       ├── stock_env.py          # Trading environment
│       ├── alpaca_trade_system.py # Alpaca trading integration
│       ├── performance_monitor.py # Performance monitoring
│       ├── replay_buffer.py      # Experience replay buffer
│       ├── noise.py              # Action noise implementation
│       ├── utils.py              # Utility functions
│       └── system_utils.py       # System utilities
├── docs/                         # Documentation
│   ├── AUTOMATION_GUIDE.md       # Automation setup guide
│   └── CONTRIBUTING.md           # Contribution guidelines
├── scripts/                      # Installation and setup scripts
│   ├── install_service.sh        # Service installation
│   ├── start_market_automation.sh # Startup script
│   └── stockbot-automation.service # Systemd service file
├── examples/                     # Example configurations and usage
├── main.py                       # Main entry point
├── market_automation.py          # Automation system
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # Project overview
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── env.example                   # Environment variables template
├── stocks.json                   # Stock symbols configuration
└── hyperparameters.json          # Optimized hyperparameters
```

### Key Directories

#### `src/stockbot/`

Core package containing all the main components:

- **DRL Agent**: Deep reinforcement learning implementation
- **Trading Environment**: Market simulation and state management
- **Alpaca Integration**: Real trading system integration
- **Performance Monitoring**: Training and trading metrics
- **Utilities**: Helper functions and system checks

#### `docs/`

Project documentation and guides:

- **Automation Guide**: How to set up automated trading
- **Contributing Guidelines**: How to contribute to the project

#### `scripts/`

Installation and deployment scripts:

- **Service Installation**: Systemd service setup
- **Startup Scripts**: Automated startup procedures

#### `examples/`

Example configurations and usage patterns

### Configuration Files

- **`config.py`**: Main configuration management
- **`hyperparameters.json`**: Optimized hyperparameters from training
- **`stocks.json`**: List of stocks to trade
- **`env.example`**: Template for environment variables

### Entry Points

- **`main.py`**: Main application entry point
- **`market_automation.py`**: Automated trading system

### Generated Directories (Git-ignored)

- **`data_cache/`**: Cached market data
- **`models/`**: Trained model files
- **`checkpoints/`**: Training checkpoints
- **`.venv/`**: Virtual environment

## ⚙️ Configuration

### Basic Configuration

Edit `config.py` to customize:

- Stock symbols to trade
- Initial balance and risk parameters
- Training hyperparameters

### API Setup (Optional - for live trading)

Create `.env` file:

```bash
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADING=true  # Use paper trading for safety
```

### Key Configuration Files

- **Stocks**: Edit `stocks.json` to change trading symbols
- **Hyperparameters**: Optimized values are in `hyperparameters.json`
- **Environment**: Configure API keys in `.env`

## 🧪 Testing & Development

### Quick Test

```bash
# Test the system (safe, no trading)
python main.py --mode test --episodes 5

# Train a model (uses historical data only)
python main.py --mode train --epochs 50
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run with console scripts
stockbot-automation
```

## 📊 Performance & Monitoring

### Training Performance

- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Memory Management**: Efficient replay buffer with GPU tensors
- **Progress Tracking**: Real-time training metrics and visualization
- **Checkpointing**: Automatic model saving and recovery

### System Monitoring

- **Health Checks**: CPU, memory, and GPU usage monitoring
- **Logging**: Comprehensive logging with rotation
- **Error Recovery**: Automatic restart on failures
- **Performance Metrics**: Training and trading performance tracking

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📚 Documentation

- **[Automation Guide](docs/AUTOMATION_GUIDE.md)**: Detailed setup for automated trading
- **[Contributing Guidelines](docs/CONTRIBUTING.md)**: How to contribute to the project
- **Code Documentation**: Comprehensive docstrings and type hints

## 🔒 Security & Safety

### Paper Trading First

- Always start with paper trading (`ALPACA_PAPER_TRADING=true`)
- Test thoroughly before live trading
- Monitor performance closely

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real money.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ProgramingPro2/StockBot2.0/issues)
- **Documentation**: Check the [docs/](docs/) directory
