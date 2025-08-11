# 🤖 StockBot2.0 Automation Guide

## ⚡ Quick Setup

### 1. Install as System Service

```bash
sudo ./install_service.sh
```

### 2. Start Automation

```bash
# Manual start
python3 market_automation.py

# System service (auto-starts on boot)
sudo systemctl start stockbot-automation
sudo systemctl status stockbot-automation
```

## 🎯 Time-Based Configuration

### Practical Time Limits (config.py)

```python
AUTOMATION_CONFIG = {
    # Enable time-based controls
    "use_time_based_training": True,
    "use_time_based_optimization": True,

    # Friday after market close
    "training_schedule": {
        "friday_optimization": {
            "time_limit_hours": 6,    # Soft limit
            "max_time_hours": 8       # Hard limit
        },

        # Weekend training sessions
        "weekend_training": {
            "time_limit_hours": 3,    # Per session
            "max_time_hours": 4,      # Hard limit
            "sessions": 6             # Max sessions
        }
    },

    # System safety
    "health_checks": {
        "min_disk_space_gb": 2.0,
        "min_memory_gb": 4.0,
        "max_cpu_usage_percent": 85,
        "max_gpu_memory_percent": 90
    }
}
```

## 📊 Weekend Schedule

**Total Weekend Usage: ~24 hours (40% utilization)**

| Day            | Activity             | Duration | Hard Limit |
| -------------- | -------------------- | -------- | ---------- |
| Friday 4:00 PM | Optimization         | 6 hours  | 8 hours    |
| Saturday       | Training Session 1-3 | 3h each  | 4h each    |
| Sunday         | Training Session 4-6 | 3h each  | 4h each    |

**Benefits:**

- ✅ **Reliable completion** before Monday market open
- ✅ **Conservative resource usage** (40% of available time)
- ✅ **Built-in safety margins** with hard limits
- ✅ **No overruns** that interfere with live trading

## 🔧 Manual Controls

### Time-Based Training

```bash
# 3-hour training session
python3 main.py --mode train --time-limit 3 --max-time 4

# 6-hour optimization
python3 main.py --mode optimize --time-limit 6 --max-time 8
```

### Episode-Based Training (Legacy)

```bash
# Disable time-based controls in config.py
"use_time_based_training": False,
"use_time_based_optimization": False,

# Then run normally
python3 main.py --mode train --epochs 500
python3 main.py --mode optimize --trials 100
```

## 📈 Monitoring

### Real-Time Status

```bash
# View automation logs
tail -f automation.log

# Check system service
sudo journalctl -u stockbot-automation -f

# Manual status check
python3 -c "
from market_automation import MarketAutomation
automation = MarketAutomation()
automation.status_report()
"
```

### Expected Output

```
=== Market Automation Status ===
Market Status: closed
Market Open: False
Current Time: 2025-08-09 13:28:28
⏰ Weekend time remaining: 44.0h until Monday 9:30 AM
🔄 Live Trading: Stopped
🎯 Optimization: Running
🎓 Training: Stopped
⏱️ Training Mode: Time-based control enabled
🕐 Optimization Mode: Time-based control enabled
```

## 🛡️ Safety Features

### Automatic Safeguards

- **Hard Time Limits**: Processes stop automatically
- **Resource Monitoring**: CPU, memory, disk space checks
- **Weekend Time Calculation**: Ensures completion before Monday
- **Health Checks**: Validates system state before starting
- **Graceful Shutdown**: Completes current episodes before stopping

### Manual Override

```bash
# Stop all processes
pkill -f "python3 main.py"

# Restart automation
sudo systemctl restart stockbot-automation
```

## 🎯 Best Practices

1. **Monitor disk space** - Models and logs can grow large
2. **Check GPU temperature** - Long training sessions generate heat
3. **Review logs regularly** - automation.log and systemd journal
4. **Test time limits** - Use short limits for testing (--time-limit 0.1)
5. **Backup models** - Save successful models before major changes

## 🚨 Troubleshooting

### Common Issues

```bash
# Permission issues
sudo chown -R $USER:$USER /home/jpearce/Documents/Python/StockBot2-Experimental

# GPU memory issues
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Service not starting
sudo systemctl status stockbot-automation
sudo journalctl -u stockbot-automation --no-pager
```

### Reset Everything

```bash
# Stop all processes
sudo systemctl stop stockbot-automation
pkill -f "python3 main.py"

# Clear logs
> automation.log
> app.log

# Restart
sudo systemctl start stockbot-automation
```

---

**🎉 Your StockBot2.0 is now fully automated with intelligent time management!**
