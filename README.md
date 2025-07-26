# StockBot2-Experimental

A sophisticated algorithmic trading system that uses Deep Reinforcement Learning (DRL) to make autonomous trading decisions. This experimental version features advanced GPU optimization, hyperparameter tuning, and robust live trading capabilities through the Alpaca API.

## 🚀 Key Features

- **Deep Reinforcement Learning Agent**: DDPG (Deep Deterministic Policy Gradient) architecture with Actor-Critic networks
- **GPU Acceleration**: Optimized for RTX 3090 with mixed precision training and XLA JIT compilation
- **Live Trading Integration**: Seamless connection to Alpaca API for paper/live trading
- **Hyperparameter Optimization**: Automated tuning using Optuna for optimal performance
- **Robust Data Handling**: Multi-source data fetching with caching and fallback mechanisms
- **Automated Execution**: Cron-based automation for daily trading and weekly training
- **Advanced State Representation**: Enhanced market state encoding with historical data
- **Risk Management**: Built-in position limits and wash trade prevention

## 📁 Project Structure

### Core Components

| File                     | Purpose                           | Key Features                                                                           |
| ------------------------ | --------------------------------- | -------------------------------------------------------------------------------------- |
| `main.py`                | Main application entry point      | Orchestrates training, testing, optimization, and live trading modes                   |
| `drl_agent.py`           | Deep Reinforcement Learning agent | DDPG implementation with Actor-Critic networks, experience replay, and target networks |
| `stock_env.py`           | Trading environment simulation    | Market simulation with realistic trading constraints and reward calculation            |
| `alpaca_trade_system.py` | Live trading interface            | Real-time order execution and portfolio management via Alpaca API                      |
| `config.py`              | Configuration management          | Centralized settings, API credentials, and hyperparameter management                   |

### Supporting Modules

| File                    | Purpose                      | Key Features                                                                     |
| ----------------------- | ---------------------------- | -------------------------------------------------------------------------------- |
| `replay_buffer.py`      | Experience replay buffer     | Efficient memory management for DRL training with TensorFlow dataset integration |
| `noise.py`              | Exploration noise generation | Ornstein-Uhlenbeck process for improved exploration during training              |
| `utils.py`              | Utility functions            | Cache invalidation and data management utilities                                 |
| `data_fetch_testing.py` | Data fetching utilities      | Testing and validation tools for Alpaca API data retrieval                       |

### Configuration & Data

| File                   | Purpose               | Contents                                                           |
| ---------------------- | --------------------- | ------------------------------------------------------------------ |
| `config.py`            | Central configuration | API keys, trading parameters, model hyperparameters                |
| `stocks.json`          | Trading universe      | List of stock symbols to trade (27 major stocks)                   |
| `requirements.txt`     | Dependencies          | Python packages including TensorFlow, Alpaca-py, Optuna            |
| `market_automation.sh` | Automation script     | Cron job setup for automated trading and training                  |
| `setup_env.py`         | Environment setup     | Interactive script to configure API credentials and validate setup |

## 🧠 Architecture Overview

### Deep Reinforcement Learning Agent (`drl_agent.py`)

The DRL agent implements a DDPG architecture with the following components:

- **Actor Network**: Determines trading actions (buy/sell/hold) for each stock
- **Critic Network**: Evaluates the quality of actions and estimates Q-values
- **Target Networks**: Stable learning through soft target updates
- **Experience Replay**: Efficient memory management with TensorFlow datasets
- **Ornstein-Uhlenbeck Noise**: Improved exploration during training

### Trading Environment (`stock_env.py`)

The environment provides a realistic simulation of stock trading:

- **Multi-Symbol Trading**: Supports trading multiple stocks simultaneously
- **Historical Data Integration**: Uses Alpaca API and yfinance as fallback
- **State Representation**: Encodes current prices, positions, balance, and historical data
- **Reward Calculation**: Portfolio value changes with transaction costs
- **Trading Constraints**: Position limits and realistic market conditions

### Live Trading System (`alpaca_trade_system.py`)

Handles real-time trading execution:

- **Real-Time Data**: Fetches current market data from Alpaca
- **Order Execution**: Converts agent decisions into market orders
- **Portfolio Tracking**: Monitors positions and account balance
- **Error Handling**: Robust retry mechanisms and error recovery

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Alpaca trading account (paper or live)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd StockBot2-Experimental
   ```

2. **Create virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   ```bash
   # Run the setup script to configure your environment
   python setup_env.py

   # Or manually create .env file:
   # cp .env.example .env
   # Edit .env with your Alpaca API credentials
   ```

## 🎯 Usage Modes

### Training Mode

```bash
python main.py --mode train --epochs 1000 --gpu
```

- Trains the DRL agent on historical data
- Saves best models to `./models/`
- Includes early stopping and checkpointing

### Testing Mode

```bash
python main.py --mode test --episodes 50
```

- Evaluates trained agent on unseen data
- Generates performance visualizations
- Creates `test_results.png` and `final_values_distribution.png`
- Use `--episodes` to control number of test episodes (default: 100)

### Hyperparameter Optimization

```bash
python main.py --mode optimize --trials 100
```

- Uses Optuna for automated hyperparameter tuning
- Optimizes learning rate, batch size, gamma, noise parameters
- Saves best parameters to `hyperparameters.json`

### Live Trading

```bash
python main.py --mode live
```

- Connects to Alpaca API for real-time trading
- Executes agent decisions in live market
- Includes robust error handling and retry logic

## 🤖 Automation

The `market_automation.sh` script provides automated execution:

### Setup Automation

```bash
./market_automation.sh install
```

### Cron Jobs Created

- **Weekday Trading**: 9 AM daily for live trading
- **Friday Training**: 4:30 PM for weekly model retraining
- **Saturday Optimization**: 10 PM for hyperparameter tuning
- **Reboot Recovery**: 5 minutes after system restart

### Manual Execution

```bash
./market_automation.sh live      # Start live trading
./market_automation.sh training  # Start training
./market_automation.sh optimize  # Start optimization
```

## 📊 Configuration

### Trading Parameters (`config.py`)

- **Initial Balance**: $1,000 starting capital
- **Stock Universe**: 27 major stocks (AAPL, MSFT, GOOGL, etc.)
- **Lookback Window**: 168 time steps of historical data
- **Max Shares**: 5 shares per trade limit
- **Trading Frequency**: Hourly trading decisions

### Model Parameters

- **State Dimension**: Dynamic based on number of stocks
- **Action Dimension**: One action per stock
- **Learning Rate**: 0.0001 (optimizable)
- **Batch Size**: 4096 (optimized for RTX 3090)
- **Memory Size**: 100,000 experiences

## 📈 Performance Monitoring

### Logging

- Comprehensive logging to `app.log`
- Real-time performance metrics
- Error tracking and debugging information

### Visualizations

- Portfolio value over time plots
- Final value distribution histograms
- Training progress with loss curves

### Model Checkpoints

- Automatic model saving during training
- Best model preservation
- Checkpoint recovery capabilities

## 🔧 Advanced Features

### GPU Optimization

- Mixed precision training for RTX 3090
- XLA JIT compilation
- Memory growth management
- Efficient data prefetching

### Data Management

- Intelligent caching system
- Parallel data fetching
- Fallback data sources (yfinance)
- Cache invalidation strategies

### Risk Management

- Position size limits
- Wash trade prevention
- Transaction cost modeling
- Portfolio diversification

## ⚠️ Important Notes

### Disclaimer

This project is for **educational and experimental purposes only**. Algorithmic trading involves significant financial risks. The creators are not responsible for any financial losses.

### Paper Trading

The system defaults to paper trading mode. Enable live trading only after thorough testing and understanding of the risks.

### Model Limitations

- Historical performance doesn't guarantee future results
- Market conditions change, requiring regular retraining
- The agent may not perform well in all market conditions

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce batch size in `config.py`
2. **API Rate Limits**: Implement longer delays between requests
3. **Data Fetching Errors**: Check API credentials and network connectivity
4. **Training Instability**: Adjust learning rate or noise parameters

### Debug Mode

Enable detailed logging by modifying the logging level in individual modules.

## 📚 Dependencies

- **TensorFlow**: Deep learning framework
- **Alpaca-py**: Trading API client
- **Optuna**: Hyperparameter optimization
- **Pandas/NumPy**: Data manipulation
- **Matplotlib**: Visualization
- **Python-dotenv**: Environment management

## 🤝 Contributing

This is an experimental project. Contributions are welcome but should be thoroughly tested before integration.

## 📄 License

This project is provided as-is for educational purposes. Use at your own risk.
