import os
import shutil
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

def configure_ssl_settings():
    """
    Configure SSL settings globally to handle common SSL/TLS issues.
    This should be called early in the application startup.
    """
    try:
        # Configure SSL context with more compatible cipher suites
        ssl_context = ssl.create_default_context()
        ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Set environment variables for SSL configuration
        os.environ['SSL_CERT_FILE'] = ssl.get_default_verify_paths().cafile or '/etc/ssl/certs/ca-certificates.crt'
        os.environ['SSL_CERT_DIR'] = ssl.get_default_verify_paths().capath or '/etc/ssl/certs'
        
        # Configure requests session with retry logic
        session = requests.Session()
        
        # Retry strategy for handling temporary network issues
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set user agent to avoid being blocked
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info("SSL settings configured successfully")
        return session
        
    except Exception as e:
        logger.warning(f"Could not configure SSL settings: {e}")
        return None

def create_robust_session():
    """
    Create a requests session with robust error handling and retry logic.
    
    Returns:
        requests.Session: Configured session with retry logic
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    return session

# Removed unused GPU utility functions
