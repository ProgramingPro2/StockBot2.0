import logging
from datetime import datetime, timedelta
from config import STOCK_SYMBOLS, CACHE_DIR

logger = logging.getLogger(__name__)

def invalidate_cache(symbols=None, days=1):
    """
    Invalidates cache files for the specified symbols or all symbols
    that are older than the specified number of days.

    Args:
        symbols: List of symbols to invalidate cache for, or None for all.
        days: Number of days to consider for cache invalidation.
    """
    if symbols is None:
        symbols = STOCK_SYMBOLS

    # Get current date
    now = datetime.now()
    cutoff_time = now - timedelta(days=days)

    invalidated_count = 0

    for cache_file in CACHE_DIR.glob("*.pkl"):
        # Only process files for the specified symbols
        symbol_match = False
        for symbol in symbols:
            if symbol in cache_file.name:
                symbol_match = True
                break

        if not symbol_match:
            continue

        # Check file modification time
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if mod_time < cutoff_time:
            cache_file.unlink()  # Delete the file
            invalidated_count += 1
            logger.info(f"Invalidated cache file: {cache_file.name}")

    logger.info(f"Cache invalidation complete. Removed {invalidated_count} files.")
