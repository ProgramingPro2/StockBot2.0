import os
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "True").lower() == "true"

def validate_config():
    """
    Validates the configuration and raises errors for missing or invalid settings.
    """
    errors = []
    
    # Check API credentials
    if not ALPACA_API_KEY:
        errors.append("ALPACA_API_KEY not found in environment variables")
    if not ALPACA_SECRET_KEY:
        errors.append("ALPACA_SECRET_KEY not found in environment variables")
    
    # Check stock symbols
    if not STOCK_SYMBOLS:
        errors.append("No stock symbols configured")
    elif len(STOCK_SYMBOLS) == 0:
        errors.append("Stock symbols list is empty")
    
    # Check other critical parameters
    if INITIAL_BALANCE <= 0:
        errors.append("Initial balance must be positive")
    if LOOKBACK_WINDOW <= 0:
        errors.append("Lookback window must be positive")
    if MAX_SHARES_PER_TRADE <= 0:
        errors.append("Max shares per trade must be positive")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")
    logger.info(f"Trading {len(STOCK_SYMBOLS)} symbols with ${INITIAL_BALANCE} initial balance")
    logger.info(f"Paper trading enabled: {ALPACA_PAPER_TRADING}")

# Configuration parameters
STOCK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "SPY",
    "TSLA",
    "NVDA",
    "INTC",
    "AMD",
]
INITIAL_BALANCE = 1000.0
LOOKBACK_WINDOW = 168
TRADING_FREQUENCY = "hourly"
MAX_SHARES_PER_TRADE = 5
TRAINING_EPISODES = 1000
TEST_EPISODES = 100
REWARD_SCALING_FACTOR = 1.0  # Reduced from 100.0 to prevent unrealistic rewards

if os.path.exists("stocks.json"):
    with open("stocks.json", "r") as f:
        try:
            STOCK_SYMBOLS = json.load(f)
            if not isinstance(STOCK_SYMBOLS, list):
                raise ValueError(
                    "Invalid format in stocks.json. Expected a list of stock symbols."
                )
            # Ensure all symbols are uppercase
            STOCK_SYMBOLS = [symbol.upper() for symbol in STOCK_SYMBOLS]
            # Remove duplicates
            STOCK_SYMBOLS = list(set(STOCK_SYMBOLS))
            # Remove empty strings
            STOCK_SYMBOLS = [symbol for symbol in STOCK_SYMBOLS if symbol]
            logger.info(f"Loaded stock symbols from stocks.json: {STOCK_SYMBOLS}")
        except Exception as e:
            logger.error(f"Error loading stock symbols from stocks.json: {e}")

# RL model parameters
STATE_DIM = (len(STOCK_SYMBOLS) * 2) + 1 + (len(STOCK_SYMBOLS) * LOOKBACK_WINDOW)
ACTION_DIM = len(STOCK_SYMBOLS)
LEARNING_RATE = 0.0001
GAMMA = 0.95
BATCH_SIZE = 4096  # For max GPU utilization on RTX 3090 (with mixed precision)
MEMORY_SIZE = 100000  # Increased for better training stability
NOISE_SIGMA = 0.15
NOISE_THETA = 0.2

# Load hyperparameters if available
hyperparams_path = Path("./hyperparameters.json")
if hyperparams_path.exists():
    with open(hyperparams_path, "r") as f:
        try:
            optimized_params = json.load(f)
            LEARNING_RATE = optimized_params.get("learning_rate", LEARNING_RATE)
            BATCH_SIZE = optimized_params.get("batch_size", BATCH_SIZE)
            GAMMA = optimized_params.get("gamma", GAMMA)
            NOISE_SIGMA = optimized_params.get("noise_sigma", NOISE_SIGMA)
            NOISE_THETA = optimized_params.get("noise_theta", NOISE_THETA)
            REWARD_SCALING_FACTOR = optimized_params.get(
                "reward_scaling_factor", REWARD_SCALING_FACTOR
            )
            logger.info(f"Loaded optimized hyperparameters from {hyperparams_path}")
            logger.info(f"  LEARNING_RATE: {LEARNING_RATE}")
            logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
            logger.info(f"  GAMMA: {GAMMA}")
            logger.info(f"  NOISE_SIGMA: {NOISE_SIGMA}")
            logger.info(f"  NOISE_THETA: {NOISE_THETA}")
            logger.info(f"  REWARD_SCALING_FACTOR: {REWARD_SCALING_FACTOR}")
        except Exception as e:
            logger.error(f"Error loading optimized hyperparameters: {e}")

# Cache directory for stock data
CACHE_DIR = Path("./data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Data source configuration
USE_YFINANCE = True

# Validate configuration after all parameters are loaded
validate_config()
