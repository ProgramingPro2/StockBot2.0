#!/usr/bin/env python3
"""
Automated training script for StockBot2
Handles scheduled training, monitoring, and model management
"""

import os
import sys
import time
import schedule
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrainer:
    def __init__(self, epochs_per_session=50, max_sessions=10):
        self.epochs_per_session = epochs_per_session
        self.max_sessions = max_sessions
        self.session_count = 0
        self.last_training_time = None
        self.training_active = False
        
    def run_training_session(self):
        """Run a single training session"""
        if self.training_active:
            logger.warning("Training already active, skipping session")
            return
            
        if self.session_count >= self.max_sessions:
            logger.info(f"Reached maximum sessions ({self.max_sessions}), stopping")
            return
            
        try:
            self.training_active = True
            self.session_count += 1
            
            logger.info(f"Starting training session {self.session_count}/{self.max_sessions}")
            logger.info(f"Training for {self.epochs_per_session} epochs")
            
            # Activate virtual environment and run training
            cmd = [
                "bash", "-c", 
                f"source .venv/bin/activate && python3 main.py --mode train --epochs {self.epochs_per_session}"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Training session {self.session_count} completed successfully in {duration:.1f}s")
                self.last_training_time = datetime.now()
                
                # Log some output for monitoring
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:  # Last 5 lines
                        if 'Episode' in line or 'Reward' in line or 'Loss' in line:
                            logger.info(f"Training output: {line}")
                            
            else:
                logger.error(f"Training session {self.session_count} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr[-500:]}")  # Last 500 chars
                    
        except Exception as e:
            logger.error(f"Exception during training session: {e}")
        finally:
            self.training_active = False
    
    def run_test_session(self):
        """Run a quick test to evaluate current model"""
        try:
            logger.info("Running test session")
            
            cmd = [
                "bash", "-c", 
                "source .venv/bin/activate && python3 main.py --mode test"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info("Test session completed successfully")
                # Extract performance metrics from output
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-10:]:
                        if any(keyword in line.lower() for keyword in ['portfolio', 'profit', 'return', 'performance']):
                            logger.info(f"Test result: {line}")
            else:
                logger.error(f"Test session failed: {result.stderr[-200:]}")
                
        except Exception as e:
            logger.error(f"Exception during test session: {e}")
    
    def cleanup_old_checkpoints(self):
        """Clean up old checkpoint files to save space"""
        try:
            checkpoints_dir = Path("checkpoints")
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.h5"))
                if len(checkpoint_files) > 10:  # Keep only latest 10
                    # Sort by modification time
                    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                    # Remove oldest files
                    for old_file in checkpoint_files[:-10]:
                        old_file.unlink()
                        logger.info(f"Removed old checkpoint: {old_file.name}")
        except Exception as e:
            logger.error(f"Error cleaning checkpoints: {e}")
    
    def status_report(self):
        """Print current status"""
        logger.info(f"=== Auto-Trainer Status ===")
        logger.info(f"Sessions completed: {self.session_count}/{self.max_sessions}")
        logger.info(f"Training active: {self.training_active}")
        if self.last_training_time:
            logger.info(f"Last training: {self.last_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Next training: {schedule.next_run()}")

def main():
    parser = argparse.ArgumentParser(description="Automated StockBot2 Training")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per training session")
    parser.add_argument("--sessions", type=int, default=10, help="Maximum training sessions")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between training sessions")
    parser.add_argument("--test-interval", type=int, default=180, help="Minutes between test sessions")
    parser.add_argument("--run-once", action="store_true", help="Run one training session and exit")
    
    args = parser.parse_args()
    
    logger.info("Starting StockBot2 Auto-Trainer")
    logger.info(f"Configuration: {args.epochs} epochs/session, {args.sessions} max sessions")
    
    trainer = AutoTrainer(epochs_per_session=args.epochs, max_sessions=args.sessions)
    
    if args.run_once:
        logger.info("Running single training session")
        trainer.run_training_session()
        return
    
    # Schedule training sessions
    schedule.every(args.interval).minutes.do(trainer.run_training_session)
    
    # Schedule test sessions
    schedule.every(args.test_interval).minutes.do(trainer.run_test_session)
    
    # Schedule cleanup
    schedule.every().day.at("02:00").do(trainer.cleanup_old_checkpoints)
    
    # Schedule status reports
    schedule.every(30).minutes.do(trainer.status_report)
    
    logger.info(f"Scheduled training every {args.interval} minutes")
    logger.info(f"Scheduled testing every {args.test_interval} minutes")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while trainer.session_count < trainer.max_sessions:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        logger.info("All training sessions completed!")
        
    except KeyboardInterrupt:
        logger.info("Auto-trainer stopped by user")
    except Exception as e:
        logger.error(f"Auto-trainer error: {e}")

if __name__ == "__main__":
    main()
