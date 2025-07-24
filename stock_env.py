import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
import requests
import json
# Removed these as we are now using direct requests for historical bars
# from alpaca.data.historical import StockHistoricalDataClient
# from alpaca.data.requests import StockBarsRequest
# from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import yfinance as yf

# Ensure config.py is correctly set up with your keys
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, USE_YFINANCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
)
logger = logging.getLogger(__name__)

class StockTradingEnvironment:
    """A simulated stock trading environment for training and testing reinforcement learning agents."""

    # Class-level cache for stock data across all instances
    _global_data_cache = {}

    def __init__(self, symbols, initial_balance, lookback_window, stock_data_client, cache_dir, max_shares_per_trade, reward_scaling_factor):
        """
        Initializes the StockTradingEnvironment.

        Args:
            symbols: A list of stock symbols to trade.
            initial_balance: The initial cash balance for the trading account.
            lookback_window: The number of historical data points to consider for the state.
            stock_data_client: An instance of StockHistoricalDataClient for fetching data.
                               NOTE: This client is no longer directly used for bar data fetching
                                     within _get_data due to the new direct request method.
            cache_dir: The directory for caching stock data.
            max_shares_per_trade: The maximum number of shares to trade in a single transaction.
            reward_scaling_factor: The factor to scale the reward by.
        """
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        # self.stock_data_client = stock_data_client # Keep if you use it for other data types, but not for bars here
        self.cache_dir = cache_dir
        self.max_shares_per_trade = max_shares_per_trade
        self.reward_scaling_factor = reward_scaling_factor

        # Instance-level cache reference
        self.data_cache = {}
        # Precompute fixed values
        self.shares_normalization = 100.0
        self.price_normalization = 1000.0
        # Preallocate state array size for reuse
        self.state_size = (
            (len(self.symbols) * 2) + 1 + (len(self.symbols) * self.lookback_window)
        )
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.reset()

    def reset(self):
        """Resets the environment to its initial state for a new episode.

        Returns:
            The initial state of the environment.
        """
        self.balance = self.initial_balance
        self.data = self._get_data() # This calls the modified data fetching method
        self.prefetched_data = self.prefetch_data()

        # Ensure we have enough data for lookback window - use numpy for efficiency
        data_lengths = np.array(
            [len(self.prefetched_data[symbol]) for symbol in self.symbols]
        )
        min_data_length = np.min(data_lengths) if data_lengths.size > 0 else 0

        # Set current step to lookback_window or to data length if not enough data
        # IMPORTANT: Since data is now fetched in ASCENDING order (oldest first),
        # self.current_step will represent the index from the OLDEST data point.
        # The environment will effectively run forwards from the oldest data.
        self.current_step = self.lookback_window # Start after the initial lookback window

        if min_data_length < self.lookback_window + 1: # Need at least lookback_window + 1 data points to start
            logger.warning(
                f"Not enough data for full lookback window. Min data length: {min_data_length} vs required: {self.lookback_window + 1}."
            )
            # If not enough data, start at the earliest possible step
            self.current_step = min_data_length - 1 if min_data_length > 0 else 0
            if self.current_step < 0:
                self.current_step = 0 # Ensure it's not negative
            logger.warning(f"Adjusted starting step to {self.current_step} due to insufficient data.")

        self.shares = {symbol: 0 for symbol in self.symbols}
        self.done = False
        # Pre-allocate memory for history
        self.portfolio_value_history = [self.initial_balance]

        # Pre-cache current prices for all symbols
        self._update_current_prices()

        return self._get_state()


    def _update_current_prices(self):
        """Caches the current prices for all symbols to avoid repeated lookups."""
        self.current_prices = np.array(
            [
                self.prefetched_data[symbol][self.current_step]
                for symbol in self.symbols
            ],
            dtype=np.float32,
        )

    def _get_state(self):
        """Creates the state representation with vectorized operations for better performance.

        Returns:
            A numpy array representing the current state of the environment.
        """
        # Reuse pre-allocated array for better performance
        state = np.zeros(self.state_size, dtype=np.float32)

        # Current price and holdings with vectorized operations
        prices = self.current_prices / self.price_normalization
        shares = (
            np.array([self.shares[s] for s in self.symbols], dtype=np.float32)
            / self.shares_normalization
        )
        state[: len(self.symbols) * 2] = np.column_stack((prices, shares)).flatten()

        # Add normalized balance
        state[len(self.symbols) * 2] = self.balance / self.initial_balance

        # Add historical price data in a vectorized way
        lookback_start_idx = (len(self.symbols) * 2) + 1
        for i, symbol in enumerate(self.symbols):
            start_idx = lookback_start_idx + (i * self.lookback_window)
            end_idx = start_idx + self.lookback_window

            # Adjusting for ascending data:
            # We need prices from current_step - lookback_window to current_step
            # Since data is oldest (index 0) to newest
            current_data_for_symbol = self.prefetched_data[symbol]
            
            # Calculate the actual start and end indices for the lookback slice
            # The slice should be [current_step - lookback_window : current_step]
            # Ensure we don't go out of bounds at the beginning of the data
            actual_lookback_start_index = max(0, self.current_step - self.lookback_window)
            actual_lookback_end_index = self.current_step
            
            # Slice the relevant historical prices
            prices_slice = current_data_for_symbol[actual_lookback_start_index : actual_lookback_end_index]
            
            # If the slice is shorter than the lookback_window, pad it with the oldest available price
            # or a sensible default to maintain state size.
            if len(prices_slice) < self.lookback_window:
                # Pad with the oldest available price or a default if no prices at all
                padding_value = prices_slice[0] if prices_slice.size > 0 else 100.0
                prices = np.full(self.lookback_window, padding_value, dtype=np.float32)
                # Place the actual data at the end of the padded array
                prices[self.lookback_window - len(prices_slice):] = prices_slice.flatten() # Added .flatten()
            else:
                prices = prices_slice.flatten() # Added .flatten()


            # Normalize the historical prices
            price_max = np.max(prices) if prices.size > 0 else 1.0
            state[start_idx:end_idx] = prices / max(price_max, 1e-8)

        return state


    def _calculate_portfolio_value(self):
        """Calculates the current portfolio value.

        Returns:
            The current total value of the portfolio (cash + assets).
        """
        # Use the cached current prices for faster calculation
        portfolio_value = self.balance

        # Vectorized calculation using numpy for efficiency
        shares = np.array([self.shares[s] for s in self.symbols], dtype=np.float32)
        portfolio_value += np.sum(shares * self.current_prices)

        return portfolio_value

    def step(self, action):
        """
        Executes a trading action in the environment.

        Args:
            action: An array of continuous actions for each stock symbol.

        Returns:
            A tuple containing:
                - next_state: The new state of the environment after the action.
                - reward: The reward received for the action.
                - done: A boolean indicating if the episode is finished.
                - info: A dictionary containing additional information (e.g., portfolio value).
        """
        # Map continuous actions to discrete buy/sell decisions
        transaction_reward = 0
        profit_reward = 0
        for i, symbol in enumerate(self.symbols):
            signal = action[i]
            current_price = self.current_prices[i]

            if signal > 0:  # Buy signal
                shares_to_buy = int(signal * self.max_shares_per_trade)
                cost = shares_to_buy * current_price
                if cost <= self.balance and shares_to_buy > 0:
                    self.balance -= cost
                    self.shares[symbol] += shares_to_buy
                    transaction_reward += 0.01
            elif signal < 0:  # Sell signal
                shares_to_sell = int(
                    -signal * min(self.max_shares_per_trade, self.shares[symbol])
                )
                if shares_to_sell > 0:
                    self.balance += shares_to_sell * current_price
                    self.shares[symbol] -= shares_to_sell
                    transaction_reward += 0.01
                    profit_reward += (shares_to_sell * current_price) - (shares_to_sell * self.current_prices[i])

        # Calculate reward as the change in portfolio value
        old_portfolio_value = self.portfolio_value_history[-1]
        new_portfolio_value = self._calculate_portfolio_value()

        # Primary reward is the profit or loss
        profit_loss_reward = new_portfolio_value - old_portfolio_value

        # Combine reward components
        reward = profit_loss_reward + transaction_reward * 100 + profit_reward # Increased transaction reward

        self.portfolio_value_history.append(new_portfolio_value)

        # Move to next time step (i.e., move to the next newer bar)
        self.current_step += 1

        # Check if we've reached the end of our data (i.e., exhausted all available historical bars)
        min_data_length = min(
            len(self.prefetched_data[symbol]) for symbol in self.symbols
        )
        if self.current_step >= min_data_length: # No -1 here, as current_step is the index of the *next* bar
            self.done = True

        # Update current prices for the new time step (which is now a newer bar)
        if not self.done:
            self._update_current_prices() # Only update if not done to avoid index out of bounds

        return (
            self._get_state(),
            reward,
            self.done,
            {"portfolio_value": new_portfolio_value},
        )

    def _get_data(self):
        """Gets historical stock data for all symbols with caching using either Alpaca or yfinance.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        if USE_YFINANCE:
            return self._get_yfinance_data()
        else:
            return self._get_alpaca_data()
        
    def _get_alpaca_data(self):
        """Fetches historical stock data from Alpaca Markets API with caching.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        # BASE_URL for Alpaca Data API stock bars
        BARS_BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

        if all(symbol in self._global_data_cache for symbol in self.symbols):
            logger.info("Using in-memory cached data for all symbols")
            return {symbol: self._global_data_cache[symbol] for symbol in self.symbols}

        # Calculate start and end dates relative to current time for consistency
        # End date is now, start date is (1 year + 2 * lookback) days ago
        end_date = datetime.now(timezone.utc)
        # Adjusting start_date to be 365 days + (lookback_window * 2) days
        # The *2 is just to ensure enough historical data is available, as Alpaca might
        # limit bars per request or you might need more for training.
        start_date = end_date - timedelta(days=365 + self.lookback_window * 2)

        data = {}
        symbols_to_download = []

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        for symbol in self.symbols:
            # Generate a cache file name that is robust to date range changes
            cache_file = (
                self.cache_dir
                / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_desc.pkl" # Added _desc
            )
            if cache_file.exists():
                try:
                    stock_data = pd.read_pickle(cache_file)
                    data[symbol] = stock_data
                    self._global_data_cache[symbol] = stock_data
                    logger.info(f"Loaded cached data for {symbol} from {cache_file}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached data for {symbol} from {cache_file}: {e}. Will re-download."
                    )
                    symbols_to_download.append(symbol)
            else:
                symbols_to_download.append(symbol)

        if not symbols_to_download:
            logger.info("All data loaded from cache.")
            return data

        logger.info(
            f"Downloading data for {len(symbols_to_download)} symbols: {symbols_to_download}..."
        )

        def download_symbol_data(symbol):
            try:
                # Format dates to RFC-3339 with 'Z' for UTC and URL-encode them.
                start_date_str = quote(start_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
                end_date_str = quote(end_date.strftime('%Y-%m-%dT%H:%M:%SZ'))

                # Construct the URL with parameters, including sort=desc
                url = (
                    f"{BARS_BASE_URL}?"
                    f"symbols={symbol}&"
                    f"timeframe=1H&"
                    f"start={start_date_str}&"
                    f"end={end_date_str}&"
                    f"feed=iex&"
                    f"sort=desc"
                )
                
                headers = {
                    "accept": "application/json",
                    "APCA-API-KEY-ID": ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
                }
                
                logger.debug(f"Requesting URL for {symbol}: {url}")
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                
                api_data = response.json()
                
                if api_data and 'bars' in api_data and symbol in api_data['bars'] and api_data['bars'][symbol]:
                    stock_data = pd.DataFrame(api_data['bars'][symbol])
                    stock_data['t'] = pd.to_datetime(stock_data['t'])
                    stock_data = stock_data.set_index('t')
                    stock_data = stock_data.rename(
                        columns={
                            "o": "Open",
                            "h": "High",
                            "l": "Low",
                            "c": "Close",
                            "v": "Volume",
                        }
                    )
                    # Convert index to date only if needed, for hourly data, keeping datetime is often better
                    # If you convert to date only, multiple bars for the same day will overwrite previous ones
                    # stock_data.index = stock_data.index.map(lambda x: x.date())
                    stock_data.index.name = "Timestamp" # Changed from "Date" for hourly data
                    
                    logger.info(f"Downloaded {len(stock_data)} bars for {symbol} from Alpaca.")
                    
                    # Cache the downloaded data
                    cache_file = (
                        Path(self.cache_dir)
                        / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_desc.pkl"
                    )
                    stock_data.to_pickle(cache_file)
                    logger.info(f"Cached data for {symbol} to {cache_file}")

                    return symbol, stock_data
                else:
                    logger.warning(f"No bars found for {symbol} from Alpaca in response from {url}. Response: {api_data}")
                    return symbol, None
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error downloading {symbol}: {http_err} - Response: {http_err.response.text}")
                return symbol, None
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Request error downloading {symbol}: {req_err}")
                return symbol, None
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error for {symbol}: {json_err} - Response: {response.text}")
                return symbol, None
            except Exception as e:
                logger.error(f"Unexpected error processing data for {symbol}: {e}", exc_info=True)
                return symbol, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(download_symbol_data, symbols_to_download))

        # Collect valid data
        for symbol, stock_data in results:
            if stock_data is not None:
                data[symbol] = stock_data
                self._global_data_cache[symbol] = stock_data
            else:
                logger.warning(f"Skipping {symbol} due to failed data download or empty response.")

        if not data: # If no data was fetched for any symbol
            raise ValueError("Could not fetch valid data for any of the specified symbols.")

        return data

    def _get_yfinance_data(self):
        """Gets historical stock data for all symbols with caching using yfinance.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=729)
        for symbol in self.symbols:
            cache_file = self.cache_dir / f"{symbol}_yfinance.pkl"
            if cache_file.exists():
                try:
                    stock_data = pd.read_pickle(cache_file)
                    if not stock_data.empty:
                        data[symbol] = stock_data
                        self._global_data_cache[symbol] = stock_data
                        logger.info(f"Loaded cached yfinance data for {symbol} from {cache_file}")
                    else:
                        raise ValueError("Cached data is empty.")
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached yfinance data for {symbol} from {cache_file}: {e}. Will re-download."
                    )
                    stock_data = yf.download(symbol, start=start_date, end=end_date, interval="1h")
                    if not stock_data.empty:
                        data[symbol] = stock_data
                        stock_data.to_pickle(cache_file)
            else:
                stock_data = yf.download(symbol, start=start_date, end=end_date, interval="1h")
                if not stock_data.empty:
                    data[symbol] = stock_data
                    stock_data.to_pickle(cache_file)
        return data

    def prefetch_data(self):
        """Prefetches data to memory for faster training (CPU-optimized version).
           Assumes data is sorted descending (most recent first).

        Returns:
            A dictionary where keys are stock symbols and values are numpy arrays of close prices.
        """
        # Convert and cache frequently accessed data for faster access
        prefetched_data = {}
        for symbol in self.symbols:
            try:
                # Only prefetch what we need (close prices)
                # Store as numpy array for CPU optimization
                # Ensure the 'Close' column exists
                if symbol in self.data and "Close" in self.data[symbol].columns:
                    close_prices = np.asarray(
                        self.data[symbol]["Close"].values, dtype=np.float32
                    )
                    prefetched_data[symbol] = close_prices
                else:
                    logger.warning(f"No 'Close' data found for {symbol} after fetching.")
                    # Provide dummy data or raise an error, depending on desired behavior
                    prefetched_data[symbol] = np.full(
                        self.lookback_window + 1, 100.0, dtype=np.float32 # Ensure enough for lookback
                    )
            except Exception as e:
                logger.error(f"Error prefetching data for {symbol}: {e}", exc_info=True)
                # Create dummy data if prefetching fails to prevent crashes
                prefetched_data[symbol] = np.full(
                    self.lookback_window + 1, 100.0, dtype=np.float32
                ) # Ensure enough for lookback

        return prefetched_data