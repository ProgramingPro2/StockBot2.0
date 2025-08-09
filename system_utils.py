"""
System utilities for StockBot2 - Better organized helper functions
"""
import os
import sys
import logging
import psutil
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SystemChecker:
    """Comprehensive system health and compatibility checker"""
    
    @staticmethod
    def check_python_version() -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python {version.major}.{version.minor} is not supported. Minimum required: 3.8")
            return False
        logger.info(f"Python version {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    @staticmethod
    def check_gpu_compatibility() -> Dict[str, any]:
        """Check GPU availability and compatibility"""
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'memory_info': []
        }
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info['available'] = len(gpus) > 0
            gpu_info['count'] = len(gpus)
            
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_info['devices'].append({
                        'name': gpu.name,
                        'device_type': gpu.device_type,
                        'details': details
                    })
                    
                    # Try to get memory info
                    try:
                        import GPUtil
                        gpu_list = GPUtil.getGPUs()
                        if i < len(gpu_list):
                            gpu_util = gpu_list[i]
                            gpu_info['memory_info'].append({
                                'total_memory': gpu_util.memoryTotal,
                                'free_memory': gpu_util.memoryFree,
                                'used_memory': gpu_util.memoryUsed
                            })
                    except ImportError:
                        logger.warning("GPUtil not available for detailed GPU memory info")
                        
                except Exception as e:
                    logger.warning(f"Could not get details for GPU {i}: {e}")
            
            if gpu_info['available']:
                logger.info(f"Found {gpu_info['count']} GPU(s)")
            else:
                logger.info("No GPUs detected - will use CPU")
                
        except Exception as e:
            logger.error(f"Error checking GPU compatibility: {e}")
        
        return gpu_info
    
    @staticmethod
    def check_system_resources() -> Dict[str, any]:
        """Check system memory, disk space, and CPU"""
        resources = {}
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            resources['memory'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
            
            # Disk space check
            disk = psutil.disk_usage(os.getcwd())
            resources['disk'] = {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': (disk.used / disk.total) * 100
            }
            
            # CPU check
            resources['cpu'] = {
                'count': psutil.cpu_count(),
                'percent_used': psutil.cpu_percent(interval=1)
            }
            
            logger.info(f"System resources - RAM: {resources['memory']['available_gb']:.1f}GB available, "
                       f"Disk: {resources['disk']['free_gb']:.1f}GB free, "
                       f"CPU: {resources['cpu']['count']} cores")
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
        
        return resources

class ConfigurationValidator:
    """Enhanced configuration validation with detailed checks"""
    
    @staticmethod
    def validate_state_action_dimensions(stock_symbols: List[str], lookback_window: int) -> Tuple[int, int]:
        """Calculate and validate state/action dimensions"""
        num_symbols = len(stock_symbols)
        
        # State calculation:
        # - Current prices: num_symbols
        # - Current shares: num_symbols  
        # - Current balance: 1
        # - Price history: num_symbols * lookback_window
        state_dim = (num_symbols * 2) + 1 + (num_symbols * lookback_window)
        action_dim = num_symbols
        
        logger.info(f"Calculated dimensions - State: {state_dim}, Action: {action_dim}")
        logger.info(f"  - {num_symbols} symbols")
        logger.info(f"  - {lookback_window} lookback window")
        logger.info(f"  - State breakdown: prices({num_symbols}) + shares({num_symbols}) + balance(1) + history({num_symbols * lookback_window})")
        
        return state_dim, action_dim
    
    @staticmethod
    def validate_hyperparameters(params: Dict[str, any]) -> List[str]:
        """Validate hyperparameter ranges and provide warnings"""
        warnings = []
        
        # Learning rate validation
        lr = params.get('learning_rate', 0.0001)
        if lr <= 0 or lr > 1.0:
            warnings.append(f"Learning rate {lr} is outside recommended range (1e-5 to 1e-2)")
        elif lr > 0.01:
            warnings.append(f"Learning rate {lr} may be too high, consider values < 0.01")
        
        # Batch size validation
        batch_size = params.get('batch_size', 512)
        if batch_size <= 0 or batch_size > 2048:
            warnings.append(f"Batch size {batch_size} is outside typical range (32-1024)")
        elif batch_size < 32:
            warnings.append(f"Batch size {batch_size} may be too small for stable training")
        
        # Gamma validation
        gamma = params.get('gamma', 0.99)
        if gamma <= 0 or gamma >= 1.0:
            warnings.append(f"Gamma {gamma} should be between 0 and 1")
        elif gamma < 0.9:
            warnings.append(f"Gamma {gamma} may be too low for long-term planning")
        
        # Reward scaling validation
        reward_scale = params.get('reward_scaling_factor', 0.01)
        if reward_scale <= 0:
            warnings.append(f"Reward scaling factor {reward_scale} must be positive")
        elif reward_scale > 1.0:
            warnings.append(f"Reward scaling factor {reward_scale} is very high and may cause instability")
        elif reward_scale < 0.001:
            warnings.append(f"Reward scaling factor {reward_scale} may be too small")
        
        return warnings

class FileSystemHelper:
    """Helper functions for file system operations"""
    
    @staticmethod
    def ensure_directories(directories: List[str]) -> None:
        """Ensure required directories exist"""
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            else:
                logger.debug(f"Directory exists: {directory}")
    
    @staticmethod
    def clean_old_files(directory: str, pattern: str, days_old: int = 7) -> int:
        """Clean old files matching a pattern"""
        import time
        from glob import glob
        
        cleaned_count = 0
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        try:
            pattern_path = os.path.join(directory, pattern)
            for file_path in glob(pattern_path):
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning old files: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} old files from {directory}")
        
        return cleaned_count
    
    @staticmethod
    def get_project_info() -> Dict[str, any]:
        """Get project directory and environment information"""
        return {
            'project_dir': Path.cwd(),
            'script_dir': Path(__file__).parent,
            'python_executable': sys.executable,
            'user': os.getenv('USER', 'unknown'),
            'home': os.getenv('HOME', 'unknown')
        }

def comprehensive_system_check() -> bool:
    """Run all system checks and return overall health status"""
    logger.info("=== Running Comprehensive System Check ===")
    
    all_good = True
    
    # Check Python version
    if not SystemChecker.check_python_version():
        all_good = False
    
    # Check system resources
    resources = SystemChecker.check_system_resources()
    if resources.get('memory', {}).get('available_gb', 0) < 2:
        logger.warning("Low available memory (< 2GB)")
        all_good = False
    
    if resources.get('disk', {}).get('free_gb', 0) < 1:
        logger.error("Insufficient disk space (< 1GB)")
        all_good = False
    
    # Check GPU
    gpu_info = SystemChecker.check_gpu_compatibility()
    if gpu_info['available']:
        logger.info("GPU acceleration available")
    else:
        logger.info("No GPU detected - training will be slower")
    
    # Ensure required directories
    FileSystemHelper.ensure_directories([
        './models', './data_cache', './checkpoints', './logs'
    ])
    
    if all_good:
        logger.info("✅ All system checks passed")
    else:
        logger.warning("⚠️ Some system checks failed - see warnings above")
    
    return all_good

if __name__ == "__main__":
    # Run system check if called directly
    comprehensive_system_check()
