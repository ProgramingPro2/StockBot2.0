# StockBot2 - Automated Trading with Deep Reinforcement Learning

A streamlined deep reinforcement learning system for automated stock trading using DDPG (Deep Deterministic Policy Gradients).

## Features

- **Unified Training Pipeline**: Single training mode with monitoring and checkpointing
- **GPU Acceleration**: Optimized for CUDA with fallback to CPU
- **Live Trading**: Integration with Alpaca Markets API
- **Automated Training**: Scheduled training sessions with monitoring
- **Data Caching**: Efficient data management with yfinance and Alpaca

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional - for live trading)
Create a `.env` file:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER_TRADING=true
```

### 3. Train the Model
```bash
# Basic training
python3 main.py --mode train --epochs 100

# Test trained model
python3 main.py --mode test
```

### 4. Automated Training
```bash
# Run automated training sessions
python3 auto_train.py --epochs 50 --sessions 10 --interval 60

# Run single training session
python3 auto_train.py --run-once --epochs 25
```

## Usage Examples

### Manual Training
```bash
# Train for 100 epochs
python3 main.py --mode train --epochs 100

# Test current model
python3 main.py --mode test

# Live trading (requires API keys)
python3 main.py --mode live
```

### Automated Training
```bash
# Standard automation: 50 epochs every hour, max 10 sessions
python3 auto_train.py

# Custom configuration
python3 auto_train.py --epochs 25 --sessions 20 --interval 30

# Quick single session
python3 auto_train.py --run-once --epochs 10
```

## Configuration

Key parameters in `config.py`:
- `STOCK_SYMBOLS`: List of stocks to trade
- `INITIAL_BALANCE`: Starting portfolio balance
- `TRAINING_EPISODES`: Episodes per training session
- `LEARNING_RATE`: Neural network learning rate
- `BATCH_SIZE`: Training batch size

## File Structure

```
StockBot2-Experimental/
├── main.py              # Main training/testing entry point
├── auto_train.py        # Automated training scheduler
├── config.py            # Configuration parameters
├── stock_env.py         # Trading environment
├── drl_agent.py         # DDPG agent implementation
├── replay_buffer.py     # Experience replay buffer
├── noise.py             # Exploration noise
├── utils.py             # Utility functions
├── performance_monitor.py # GPU/performance monitoring
├── alpaca_trade_system.py # Live trading system
└── requirements.txt     # Dependencies
```

## Monitoring

Training progress is logged to:
- Console output
- `app.log` (training logs)
- `auto_train.log` (automation logs)

Performance metrics include:
- Episode rewards
- Portfolio value
- Training losses
- GPU utilization

## Live Trading

⚠️ **Paper Trading Recommended**: Use `ALPACA_PAPER_TRADING=true` for testing.

The system supports live trading through Alpaca Markets:
1. Set up API credentials in `.env`
2. Run `python3 main.py --mode live`
3. Monitor trades through Alpaca dashboard

## Tips

- Start with paper trading to validate strategies
- Use automated training for hands-off model improvement
- Monitor GPU usage for optimal performance
- Regular checkpoints prevent training loss