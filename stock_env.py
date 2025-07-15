import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
            cache_dir: The directory for caching stock data.
            max_shares_per_trade: The maximum number of shares to trade in a single transaction.
            reward_scaling_factor: The factor to scale the reward by.
        """
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.stock_data_client = stock_data_client
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
        # Use numpy arrays for shares for vectorized operations
        self.data = self._get_data()
        self.prefetched_data = self.prefetch_data()

        # Ensure we have enough data for lookback window - use numpy for efficiency
        data_lengths = np.array(
            [len(self.prefetched_data[symbol]) for symbol in self.symbols]
        )
        min_data_length = np.min(data_lengths) if data_lengths.size > 0 else 0

        # Set current step to lookback_window or to data length if not enough data
        self.current_step = min(self.lookback_window, min_data_length - 1)
        if self.current_step < self.lookback_window:
            logger.warning(
                f"Not enough data for full lookback window. Using {self.current_step} days instead."
            )

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

            if self.current_step < self.lookback_window:
                available_prices = self.prefetched_data[symbol][: self.current_step + 1]
                if available_prices.size > 0:
                    first_price = available_prices[0]
                    prices = np.full(
                        self.lookback_window, first_price, dtype=np.float32
                    )
                    prices[self.lookback_window - available_prices.size :] = (
                        available_prices
                    )
                else:
                    prices = np.full(self.lookback_window, 100.0, dtype=np.float32)
            else:
                prices = self.prefetched_data[symbol][
                    self.current_step - self.lookback_window : self.current_step
                ]

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
        for i, symbol in enumerate(self.symbols):
            signal = action[i]
            current_price = self.current_prices[i]

            if signal > 0:  # Buy signal
                shares_to_buy = int(signal * self.max_shares_per_trade)
                cost = shares_to_buy * current_price
                if cost <= self.balance and shares_to_buy > 0:
                    self.balance -= cost
                    self.shares[symbol] += shares_to_buy
            elif signal < 0:  # Sell signal
                shares_to_sell = int(
                    -signal * min(self.max_shares_per_trade, self.shares[symbol])
                )
                if shares_to_sell > 0:
                    self.balance += shares_to_sell * current_price
                    self.shares[symbol] -= shares_to_sell

        # Move to next time step
        self.current_step += 1

        # Check if we've reached the end of our data
        min_data_length = min(
            len(self.prefetched_data[symbol]) for symbol in self.symbols
        )
        if self.current_step >= min_data_length - 1:
            self.done = True

        # Update current prices for the new time step
        self._update_current_prices()

        # Calculate reward as portfolio value change
        old_portfolio_value = self.portfolio_value_history[-1]
        new_portfolio_value = self._calculate_portfolio_value()

        # Avoid division by zero and use faster calculation
        if old_portfolio_value > 0:
            reward = (
                (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            ) * self.reward_scaling_factor
        else:
            reward = 0.0

        self.portfolio_value_history.append(new_portfolio_value)

        return (
            self._get_state(),
            reward,
            self.done,
            {"portfolio_value": new_portfolio_value},
        )

    def _get_data(self):
        """Gets historical stock data for all symbols with caching using alpaca-py and parallel fetching.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        if all(symbol in self._global_data_cache for symbol in self.symbols):
            logger.info("Using in-memory cached data for all symbols")
            return {symbol: self._global_data_cache[symbol] for symbol in self.symbols}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 + self.lookback_window * 2)
        data = {}
        symbols_to_download = []

        for symbol in self.symbols:
            cache_file = (
                self.cache_dir
                / f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.pkl"
            )
            if cache_file.exists():
                try:
                    stock_data = pd.read_pickle(cache_file)
                    data[symbol] = stock_data
                    self._global_data_cache[symbol] = stock_data
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached data for {symbol}: {e}. Will re-download."
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
                request_params = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Hour,
                    start=start_date,
                    end=end_date,
                )
                stock_data_obj = self.stock_data_client.get_stock_bars(request_params)
                if symbol in stock_data_obj:
                    stock_data = stock_data_obj[symbol].df
                    stock_data.index = stock_data.index - pd.Timedelta(minutes=15)
                    stock_data = stock_data.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )
                    stock_data.index = stock_data.index.map(lambda x: x.date())
                    stock_data.index.name = "Date"
                    return symbol, stock_data
                else:
                    logger.warning(f"No data for {symbol} from Alpaca.")
                    return symbol, None
            except Exception as e:
                logger.error(
                    f"Error downloading {symbol} from Alpaca: {e}. No fallback available."
                )
                return symbol, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(download_symbol_data, symbols_to_download))

        if not self.symbols:
            raise ValueError("Could not fetch data for any of the symbols.")

        return data

    def prefetch_data(self):
        """Prefetches data to memory for faster training (CPU-optimized version).

        Returns:
            A dictionary where keys are stock symbols and values are numpy arrays of close prices.
        """
        # Convert and cache frequently accessed data for faster access
        prefetched_data = {}
        for symbol in self.symbols:
            try:
                # Only prefetch what we need (close prices)
                # Store as numpy array for CPU optimization
                close_prices = np.asarray(
                    self.data[symbol]["Close"].values, dtype=np.float32
                )
                prefetched_data[symbol] = close_prices
            except Exception as e:
                logger.error(f"Error prefetching data for {symbol}: {e}")
                # Create dummy data if prefetching fails
                prefetched_data[symbol] = np.full(
                    1000, 100.0, dtype=np.float32
                )  # Assuming a default length

        return prefetched_data