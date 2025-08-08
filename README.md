# StockBot2 - Intelligent Market Automation System

A comprehensive deep reinforcement learning system for automated stock trading with intelligent market cycle management.

## 🚀 Features

- **Market-Aware Automation**: Automatically runs live trading during market hours
- **Smart Optimization**: Hyperparameter optimization after Friday market close
- **Continuous Training**: Automated training during weekends and off-hours
- **GPU Acceleration**: Optimized for CUDA with fallback to CPU
- **Live Trading**: Integration with Alpaca Markets API
- **Process Management**: Automatic restart and monitoring of all processes

## 🎯 Quick Start

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

### 3. Setup Market Automation

```bash
# Setup startup script and systemd service
python3 market_automation.py --setup-startup

# Enable system startup (optional)
sudo cp stockbot2-automation.service /etc/systemd/system/
sudo systemctl enable stockbot2-automation
```

### 4. Start Market Automation

```bash
# Run market automation (recommended)
python3 market_automation.py

# Or start manually
./start_market_automation.sh
```

## 📊 Usage Examples

### Manual Operations

```bash
# Train model
python3 main.py --mode train --epochs 100

# Test model
python3 main.py --mode test --episodes 10

# Live trading (requires API keys)
python3 main.py --mode live

# Hyperparameter optimization
python3 main.py --mode optimize --trials 20
```

### Market Automation Lifecycle

**Weekdays (Market Hours 9:30 AM - 4:00 PM EST):**

- ✅ Live trading active
- ✅ Real-time market monitoring
- ✅ Automatic trade execution

**Friday After Close:**

- 🔄 Hyperparameter optimization starts
- 📈 Model improvement cycle begins

**Weekends/Off-Hours:**

- 🎓 Continuous training sessions
- 📊 Performance analysis
- 🔧 Model refinement

## ⚙️ Configuration

### Market Automation Config (`automation_config.json`)

```json
{
  "training_epochs": 50,
  "training_sessions": 5,
  "optimization_trials": 20,
  "live_trading_enabled": true,
  "auto_restart": true,
  "check_interval": 60
}
```

### Trading Parameters (`config.py`)

- `STOCK_SYMBOLS`: List of stocks to trade
- `INITIAL_BALANCE`: Starting portfolio balance
- `TRAINING_EPISODES`: Episodes per training session
- `LEARNING_RATE`: Neural network learning rate
- `BATCH_SIZE`: Training batch size

## 📁 File Structure

```
StockBot2-Experimental/
├── main.py                    # Main training/testing entry point
├── market_automation.py       # Market automation system
├── start_market_automation.sh # Startup script
├── config.py                  # Configuration parameters
├── stock_env.py              # Trading environment
├── drl_agent.py              # DDPG agent implementation
├── replay_buffer.py          # Experience replay buffer
├── noise.py                  # Exploration noise
├── utils.py                  # Utility functions
├── performance_monitor.py    # GPU/performance monitoring
├── alpaca_trade_system.py   # Live trading system
├── automation_config.json    # Automation settings
└── requirements.txt          # Dependencies
```

## 📈 Monitoring & Logs

### Log Files

- `app.log` - Main application logs
- `market_automation.log` - Automation system logs
- `training_performance.png` - Performance graphs

### Real-time Monitoring

```bash
# Watch automation logs
tail -f market_automation.log

# Watch application logs
tail -f app.log

# Check system status
python3 market_automation.py --status
```

### Performance Metrics

- Episode rewards and portfolio value
- Training losses (actor/critic)
- GPU utilization and memory usage
- Market status and trading activity

## 🔄 Market Automation Features

### Intelligent Scheduling

- **Market Hours Detection**: Automatically detects market open/close
- **Friday Optimization**: Runs hyperparameter optimization after Friday close
- **Weekend Training**: Continuous model improvement during off-hours
- **Process Management**: Automatic restart of failed processes

### Safety Features

- **Graceful Shutdown**: Proper cleanup on system restart
- **Error Recovery**: Automatic restart of failed processes
- **Resource Monitoring**: GPU and memory usage tracking
- **Checkpoint Management**: Automatic backup of best models

## 🎯 Live Trading

⚠️ **Paper Trading Recommended**: Use `ALPACA_PAPER_TRADING=true` for testing.

### Setup Live Trading

1. Set up API credentials in `.env`
2. Enable live trading in `automation_config.json`
3. Start market automation: `python3 market_automation.py`
4. Monitor through Alpaca dashboard

### Trading Safety

- Paper trading mode for testing
- Automatic stop-loss and position limits
- Real-time portfolio monitoring
- Market hours enforcement

## 🚀 Tips & Best Practices

### Getting Started

- Start with paper trading to validate strategies
- Use market automation for hands-off operation
- Monitor logs regularly for system health
- Set up alerts for critical events

### Performance Optimization

- Monitor GPU usage for optimal performance
- Regular checkpoints prevent training loss
- Use hyperparameter optimization for model improvement
- Schedule training during off-hours

### System Maintenance

- Regular log rotation and cleanup
- Monitor disk space for checkpoints
- Update API credentials securely
- Backup best models regularly

## 🔧 Troubleshooting

### Common Issues

- **GPU Memory**: Reduce batch size if out of memory
- **API Limits**: Check Alpaca rate limits
- **Process Crashes**: Check logs for error details
- **Market Data**: Verify internet connection

### Debug Commands

```bash
# Check system health
python3 main.py --mode test --episodes 1

# Test market automation
python3 market_automation.py --config debug_config.json

# Monitor processes
ps aux | grep python3
```
