import os
import shutil
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def invalidate_cache(days=1):
    """
    Invalidates the data cache by removing files older than specified days.
    
    Args:
        days: Number of days after which cache files are considered stale.
    """
    cache_dir = Path("./data_cache")
    if not cache_dir.exists():
        return
    
    import time
    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    
    removed_count = 0
    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file_path}: {e}")
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale cache files")

# Removed unused GPU utility functions
