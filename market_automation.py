#!/usr/bin/env python3
"""
Market Automation System for StockBot2
Handles startup, live trading, optimization, and training cycles
Designed for continuous operation as a systemd service
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
import traceback

# Ensure we're in the correct directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Import automation config from main config
sys.path.append(str(script_dir))
from config import AUTOMATION_CONFIG

# Setup comprehensive logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler with rotation
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    'market_automation.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info(f"Starting Market Automation System from {script_dir}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

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
        """Load automation configuration from config.py"""
        try:
            # Use configuration from config.py
            config = AUTOMATION_CONFIG.copy()
            
            # Check for local override file
            config_file = Path("automation_config.json")
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        local_overrides = json.load(f)
                        config.update(local_overrides)
                        logger.info("Applied local configuration overrides")
                except Exception as e:
                    logger.error(f"Error loading local config overrides: {e}")
            
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fallback to minimal config
            return {
                "training_epochs": 300,
                "training_sessions": 8,
                "optimization_trials": 50,
                "live_trading_enabled": True,
                "auto_restart": True,
                "check_interval": 60
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
    
    def is_weekend(self):
        """Check if it's weekend (Saturday or Sunday)"""
        now = datetime.now()
        return now.weekday() in [5, 6]  # Saturday = 5, Sunday = 6
    
    def _health_check(self):
        """Perform health checks before starting processes"""
        try:
            # Check if virtual environment exists
            venv_python = script_dir / ".venv" / "bin" / "python3"
            if not venv_python.exists():
                logger.error(f"Virtual environment not found: {venv_python}")
                return False
            
            # Check if main script exists
            main_script = script_dir / "main.py"
            if not main_script.exists():
                logger.error(f"Main script not found: {main_script}")
                return False
            
            # Check disk space (require at least 1GB free)
            disk_usage = psutil.disk_usage(str(script_dir))
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1.0:
                logger.error(f"Insufficient disk space: {free_gb:.2f}GB free")
                return False
            
            # Check memory (require at least 2GB free)
            memory = psutil.virtual_memory()
            free_gb = memory.available / (1024**3)
            if free_gb < 2.0:
                logger.warning(f"Low memory: {free_gb:.2f}GB available")
            
            logger.info("Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
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
        
        # Health check before starting
        if not self._health_check():
            logger.error("Health check failed, not starting live trading")
            return
            
        try:
            logger.info("Starting live trading...")
            # Use absolute paths to ensure reliability
            venv_python = script_dir / ".venv" / "bin" / "python3"
            main_script = script_dir / "main.py"
            
            self.live_trading_process = subprocess.Popen([
                str(venv_python), str(main_script), "--mode", "live"
            ], 
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1")
            )
            logger.info(f"Live trading started with PID: {self.live_trading_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            logger.error(traceback.format_exc())
    
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
        
        if not self._health_check():
            logger.error("Health check failed, not starting optimization")
            return
            
        try:
            logger.info("Starting hyperparameter optimization...")
            venv_python = script_dir / ".venv" / "bin" / "python3"
            main_script = script_dir / "main.py"
            
            self.optimization_process = subprocess.Popen([
                str(venv_python), str(main_script), "--mode", "optimize", 
                "--trials", str(self.config['optimization_trials'])
            ], 
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1")
            )
            logger.info(f"Optimization started with PID: {self.optimization_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            logger.error(traceback.format_exc())
    
    def run_training(self):
        """Run training sessions"""
        if self.training_process and self.training_process.poll() is None:
            logger.info("Training already running")
            return
        
        if not self._health_check():
            logger.error("Health check failed, not starting training")
            return
            
        try:
            logger.info("Starting training sessions...")
            venv_python = script_dir / ".venv" / "bin" / "python3"
            main_script = script_dir / "main.py"
            
            self.training_process = subprocess.Popen([
                str(venv_python), str(main_script), "--mode", "train", 
                "--epochs", str(self.config['training_epochs'])
            ], 
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1")
            )
            logger.info(f"Training started with PID: {self.training_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            logger.error(traceback.format_exc())
    
    def run_weekend_training(self):
        """Run intensive weekend training with optimized parameters"""
        if self.training_process and self.training_process.poll() is None:
            logger.info("Weekend training already running")
            return
        
        if not self._health_check():
            logger.error("Health check failed, not starting weekend training")
            return
        
        # Use weekend-specific training parameters
        weekend_epochs = self.config.get("training_schedule", {}).get("weekend_training", {}).get("episodes_per_session", 300)
        
        try:
            logger.info(f"Starting intensive weekend training with {weekend_epochs} episodes...")
            venv_python = script_dir / ".venv" / "bin" / "python3"
            main_script = script_dir / "main.py"
            
            self.training_process = subprocess.Popen([
                str(venv_python), str(main_script), "--mode", "train", 
                "--epochs", str(weekend_epochs)
            ], 
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1")
            )
            logger.info(f"Weekend training started with PID: {self.training_process.pid}")
            
            # Log estimated completion time
            estimated_hours = self.config.get("training_schedule", {}).get("weekend_training", {}).get("estimated_time_per_session_hours", 2.2)
            completion_time = datetime.now() + timedelta(hours=estimated_hours)
            logger.info(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Failed to start weekend training: {e}")
            logger.error(traceback.format_exc())
    
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
                    # Check if we should run intensive weekend training
                    if self.is_weekend() and self.config.get("weekend_intensive_training", False):
                        logger.info("Weekend detected - starting intensive training session")
                        self.run_weekend_training()
                    else:
                        logger.info("Market closed - starting regular training")
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
        """Main automation loop with robust error handling"""
        logger.info("Starting Market Automation System")
        logger.info(f"Configuration: {self.config}")
        
        # Initial health check
        if not self._health_check():
            logger.error("Initial health check failed, exiting")
            return
        
        # Initial market status check
        try:
            self.market_status_check()
        except Exception as e:
            logger.error(f"Initial market status check failed: {e}")
        
        # Schedule regular checks with error handling
        def safe_market_check():
            try:
                self.market_status_check()
            except Exception as e:
                logger.error(f"Market status check failed: {e}")
                logger.error(traceback.format_exc())
        
        def safe_process_check():
            try:
                self.check_processes()
            except Exception as e:
                logger.error(f"Process check failed: {e}")
                logger.error(traceback.format_exc())
        
        def safe_status_report():
            try:
                self.status_report()
            except Exception as e:
                logger.error(f"Status report failed: {e}")
                logger.error(traceback.format_exc())
        
        schedule.every(self.config["check_interval"]).seconds.do(safe_market_check)
        schedule.every(5).minutes.do(safe_process_check)
        schedule.every(30).minutes.do(safe_status_report)
        
        logger.info("Market automation running. Press Ctrl+C to stop.")
        
        # Main loop with error recovery
        error_count = 0
        max_errors = 10
        
        try:
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(1)
                    error_count = 0  # Reset error count on successful iteration
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in main loop (count: {error_count}): {e}")
                    logger.error(traceback.format_exc())
                    
                    if error_count >= max_errors:
                        logger.critical(f"Too many errors ({max_errors}), shutting down")
                        break
                    
                    # Progressive backoff on errors
                    time.sleep(min(error_count * 5, 60))
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}")
            logger.critical(traceback.format_exc())
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
