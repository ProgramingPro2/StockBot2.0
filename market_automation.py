#!/usr/bin/env python3
"""
Market Automation System for StockBot2
Handles startup, live trading, optimization, and training cycles
"""

import os
import sys
import time
import schedule
import logging
import subprocess
import signal
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketAutomation:
    def __init__(self):
        self.live_trading_process = None
        self.optimization_process = None
        self.training_process = None
        self.is_running = True
        self.market_status = "unknown"
        
        # Market hours (EST/EDT)
        self.market_open = "09:30"
        self.market_close = "16:00"
        
        # Load configuration
        self.config = self.load_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def load_config(self):
        """Load automation configuration"""
        config_file = Path("automation_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default configuration
        return {
            "training_epochs": 50,
            "training_sessions": 5,
            "optimization_trials": 20,
            "live_trading_enabled": True,
            "auto_restart": True,
            "check_interval": 60  # seconds
        }
    
    def save_config(self):
        """Save current configuration"""
        config_file = Path("automation_config.json")
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check time
        current_time = now.strftime("%H:%M")
        return self.market_open <= current_time <= self.market_close
    
    def is_friday_after_close(self):
        """Check if it's Friday after market close"""
        now = datetime.now()
        if now.weekday() == 4:  # Friday
            current_time = now.strftime("%H:%M")
            return current_time > self.market_close
        return False
    
    def kill_process(self, process):
        """Safely kill a subprocess"""
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            logger.info("Process terminated")
    
    def start_live_trading(self):
        """Start live trading process"""
        if not self.config["live_trading_enabled"]:
            logger.info("Live trading disabled in config")
            return
            
        if self.live_trading_process and self.live_trading_process.poll() is None:
            logger.info("Live trading already running")
            return
            
        try:
            logger.info("Starting live trading...")
            self.live_trading_process = subprocess.Popen([
                "bash", "-c", 
                "source .venv/bin/activate && python3 main.py --mode live"
            ], cwd=os.getcwd())
            logger.info(f"Live trading started with PID: {self.live_trading_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
    
    def stop_live_trading(self):
        """Stop live trading process"""
        if self.live_trading_process:
            logger.info("Stopping live trading...")
            self.kill_process(self.live_trading_process)
            self.live_trading_process = None
    
    def run_optimization(self):
        """Run hyperparameter optimization"""
        if self.optimization_process and self.optimization_process.poll() is None:
            logger.info("Optimization already running")
            return
            
        try:
            logger.info("Starting hyperparameter optimization...")
            self.optimization_process = subprocess.Popen([
                "bash", "-c", 
                f"source .venv/bin/activate && python3 main.py --mode optimize --trials {self.config['optimization_trials']}"
            ], cwd=os.getcwd())
            logger.info(f"Optimization started with PID: {self.optimization_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
    
    def run_training(self):
        """Run training sessions"""
        if self.training_process and self.training_process.poll() is None:
            logger.info("Training already running")
            return
            
        try:
            logger.info("Starting training sessions...")
            self.training_process = subprocess.Popen([
                "bash", "-c", 
                f"source .venv/bin/activate && python3 auto_train.py --epochs {self.config['training_epochs']} --sessions {self.config['training_sessions']} --interval 120"
            ], cwd=os.getcwd())
            logger.info(f"Training started with PID: {self.training_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
    
    def check_processes(self):
        """Check and restart failed processes"""
        # Check live trading
        if self.live_trading_process and self.live_trading_process.poll() is not None:
            logger.warning("Live trading process died, restarting...")
            self.live_trading_process = None
            if self.is_market_open():
                self.start_live_trading()
        
        # Check optimization
        if self.optimization_process and self.optimization_process.poll() is not None:
            logger.info("Optimization completed")
            self.optimization_process = None
        
        # Check training
        if self.training_process and self.training_process.poll() is not None:
            logger.info("Training completed")
            self.training_process = None
    
    def market_status_check(self):
        """Check market status and take appropriate actions"""
        was_open = self.market_status == "open"
        is_open = self.is_market_open()
        
        if is_open and not was_open:
            logger.info("Market opened - starting live trading")
            self.market_status = "open"
            self.start_live_trading()
            
        elif not is_open and was_open:
            logger.info("Market closed - stopping live trading")
            self.market_status = "closed"
            self.stop_live_trading()
            
            # If it's Friday after close, start optimization
            if self.is_friday_after_close():
                logger.info("Friday after close - starting optimization")
                self.run_optimization()
        
        elif not is_open and self.market_status == "closed":
            # Weekend or after hours - run training if no optimization is running
            if not self.optimization_process or self.optimization_process.poll() is not None:
                if not self.training_process or self.training_process.poll() is not None:
                    logger.info("Market closed - starting training")
                    self.run_training()
    
    def status_report(self):
        """Print current status"""
        logger.info("=== Market Automation Status ===")
        logger.info(f"Market Status: {self.market_status}")
        logger.info(f"Market Open: {self.is_market_open()}")
        logger.info(f"Live Trading: {'Running' if self.live_trading_process and self.live_trading_process.poll() is None else 'Stopped'}")
        logger.info(f"Optimization: {'Running' if self.optimization_process and self.optimization_process.poll() is None else 'Stopped'}")
        logger.info(f"Training: {'Running' if self.training_process and self.training_process.poll() is None else 'Stopped'}")
        logger.info(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down market automation...")
        self.stop_live_trading()
        
        if self.optimization_process:
            self.kill_process(self.optimization_process)
        
        if self.training_process:
            self.kill_process(self.training_process)
        
        logger.info("Market automation stopped")
    
    def run(self):
        """Main automation loop"""
        logger.info("Starting Market Automation System")
        logger.info(f"Configuration: {self.config}")
        
        # Initial market status check
        self.market_status_check()
        
        # Schedule regular checks
        schedule.every(self.config["check_interval"]).seconds.do(self.market_status_check)
        schedule.every(5).minutes.do(self.check_processes)
        schedule.every(30).minutes.do(self.status_report)
        
        logger.info("Market automation running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.shutdown()

def setup_startup_script():
    """Create startup script for system boot"""
    startup_script = """#!/bin/bash
# Market Automation Startup Script
cd /home/jpearce/Documents/Python/StockBot2-Experimental
source .venv/bin/activate
python3 market_automation.py >> market_automation.log 2>&1 &
echo $! > market_automation.pid
"""
    
    script_path = Path("start_market_automation.sh")
    with open(script_path, 'w') as f:
        f.write(startup_script)
    
    os.chmod(script_path, 0o755)
    logger.info(f"Startup script created: {script_path}")
    
    # Instructions for systemd service
    systemd_service = f"""[Unit]
Description=StockBot2 Market Automation
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'jpearce')}
WorkingDirectory=/home/jpearce/Documents/Python/StockBot2-Experimental
ExecStart=/home/jpearce/Documents/Python/StockBot2-Experimental/start_market_automation.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("stockbot2-automation.service")
    with open(service_path, 'w') as f:
        f.write(systemd_service)
    
    logger.info(f"Systemd service file created: {service_path}")
    logger.info("To enable startup: sudo cp stockbot2-automation.service /etc/systemd/system/ && sudo systemctl enable stockbot2-automation")

def main():
    parser = argparse.ArgumentParser(description="Market Automation System")
    parser.add_argument("--setup-startup", action="store_true", help="Setup startup script and systemd service")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.setup_startup:
        setup_startup_script()
        return
    
    automation = MarketAutomation()
    automation.run()

if __name__ == "__main__":
    main()
