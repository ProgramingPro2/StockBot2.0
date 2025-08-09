import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
import requests
import json
import tensorflow as tf
# Removed these as we are now using direct requests for historical bars
# from alpaca.data.historical import StockHistoricalDataClient
# from alpaca.data.requests import StockBarsRequest
# from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import yfinance as yf
import os
import time

# Ensure config.py is correctly set up with your keys
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, USE_YFINANCE

logger = logging.getLogger(__name__)

class StockTradingEnvironment:
    """A simulated stock trading environment for training and testing reinforcement learning agents."""

    # Class-level cache for stock data across all instances
    _global_data_cache = {}

    @staticmethod
    def clear_in_memory_cache():
        StockTradingEnvironment._global_data_cache.clear()
        logger.info("In-memory data cache cleared.")

    def __init__(self, symbols, initial_balance, lookback_window, stock_data_client, cache_dir, max_shares_per_trade, reward_scaling_factor, debug=False):
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
        self.debug = debug

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
        
        # Initialize current_prices
        self.current_prices = {symbol: 100.0 for symbol in self.symbols}
        
        # GPU acceleration: Pre-allocate tensors on GPU
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        with tf.device(self.device):
            self.shares_tensor = tf.zeros(len(self.symbols), dtype=tf.float32)
            self.balance_tensor = tf.constant(initial_balance, dtype=tf.float32)
            self.price_history_tensor = None  # Will be initialized in reset()
        
        self.reset()

    def reset(self):
        """Resets the environment to its initial state for a new episode.

        Returns:
            The initial state of the environment.
        """
        self.balance = self.initial_balance
        
        # Reset shares to zero for each episode
        self.shares = {symbol: 0.0 for symbol in self.symbols}
        if self.debug:
            logger.info(f"RESET: Shares reset to zero: {self.shares}")
        
        # Reset GPU tensors to match CPU values
        if self.device == '/GPU:0':
            with tf.device(self.device):
                self.balance_tensor = tf.constant(self.initial_balance, dtype=tf.float32)
                self.shares_tensor = tf.zeros(len(self.symbols), dtype=tf.float32)
                if self.debug:
                    logger.info(f"RESET: GPU tensors reset - Balance: {self.balance_tensor.numpy()}, Shares: {self.shares_tensor.numpy()}")
        
        # Ensure balance is properly reset and reasonable
        if not np.isfinite(self.balance) or self.balance <= 0:
            logger.warning(f"Invalid initial balance detected: ${float(self.balance):.2f}, resetting to ${self.initial_balance:.2f}")
            self.balance = self.initial_balance
        
        # Use cached data if available to prevent repeated downloads
        if not StockTradingEnvironment._global_data_cache:
            self.data, failed_symbols = self._get_data()
            # Cache the data globally
            StockTradingEnvironment._global_data_cache = self.data.copy()
        else:
            self.data = StockTradingEnvironment._global_data_cache.copy()
            failed_symbols = []
            
        self.symbols = [s for s in self.symbols if s not in failed_symbols]
        
        logger.info(f"Loaded data for {len(self.symbols)} symbols. Failed symbols: {failed_symbols}")
        
        # Debug: Check data structure for first symbol
        if self.symbols and self.symbols[0] in self.data:
            first_symbol = self.symbols[0]
            first_data = self.data[first_symbol]
            logger.info(f"Data structure for {first_symbol}: shape={first_data.shape}, columns={list(first_data.columns)}")
        
        # Prefetch data for faster access
        self.prefetched_data = self.prefetch_data()
        
        # Initialize step counter
        self.current_step = self.lookback_window
        
        # Initialize current prices
        self._update_current_prices()
        
        # Initialize portfolio tracking
        self.portfolio_value_history = [self.initial_balance]
        self.done = False
        
        if self.symbols and self.symbols[0] in self.data:
            logger.info(f"Environment reset complete. Current step: {self.current_step}, Data length: {len(self.data[self.symbols[0]])}")
        else:
            logger.info(f"Environment reset complete. Current step: {self.current_step}, No valid data available")
        
        return self._get_state()

    def _calculate_portfolio_value_gpu(self, current_prices):
        """
        GPU-accelerated portfolio value calculation.
        
        Args:
            current_prices: Tensor of current prices for all symbols
            
        Returns:
            Portfolio value as a tensor
        """
        with tf.device(self.device):
            # Calculate stock values
            stock_values = self.shares_tensor * current_prices
            
            # Total portfolio value = cash + stock values
            portfolio_value = self.balance_tensor + tf.reduce_sum(stock_values)
            
            return portfolio_value

    def _get_state_gpu(self, current_prices, price_history):
        """
        GPU-accelerated state calculation.
        
        Args:
            current_prices: Tensor of current prices for all symbols
            price_history: Tensor of price history for all symbols
            
        Returns:
            State tensor
        """
        with tf.device(self.device):
            # Normalize prices and shares
            normalized_prices = current_prices / self.price_normalization
            normalized_shares = self.shares_tensor / self.shares_normalization
            normalized_balance = self.balance_tensor / self.price_normalization
            
            # Create state components
            price_component = tf.reshape(normalized_prices, [-1])
            shares_component = tf.reshape(normalized_shares, [-1])
            balance_component = tf.expand_dims(normalized_balance, 0)
        
        # Flatten price history
        history_component = tf.reshape(price_history, [-1])
        
        # Concatenate all components
        state = tf.concat([price_component, shares_component, balance_component, history_component], axis=0)
        
        return state

    def _update_current_prices(self):
        """Updates the current prices for all stocks."""
        try:
            for symbol in self.symbols:
                if symbol in self.data and self.current_step < len(self.data[symbol]):
                    # Get the data for this symbol
                    data = self.data[symbol]
                    
                    # Handle multi-level columns from yfinance
                    close_price = None
                    
                    # Try to find the close price column
                    if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                        # Multi-level columns - look for Close in the first level
                        for col in data.columns:
                            if isinstance(col, tuple) and 'Close' in col[0]:
                                close_price = data.iloc[self.current_step][col]
                                break
                        # If not found, try flattened columns
                        if close_price is None:
                            # Flatten multi-level columns
                            data.columns = [col[1] if isinstance(col, tuple) and len(col) > 1 else col for col in data.columns]
                            if 'Close' in data.columns:
                                close_price = data.iloc[self.current_step]['Close']
                    else:
                        # Single-level columns
                        if 'Close' in data.columns:
                            close_price = data.iloc[self.current_step]['Close']
                        else:
                            # Try alternative column names
                            for col_name in ["close", "c", "C"]:
                                if col_name in data.columns:
                                    close_price = data.iloc[self.current_step][col_name]
                                    break
                    
                    # Ensure we get a single float value
                    if close_price is not None:
                        if isinstance(close_price, (list, tuple, np.ndarray)):
                            # Take the first value if it's a sequence
                            close_price = float(close_price[0]) if len(close_price) > 0 else 100.0
                        else:
                            close_price = float(close_price)
                        
                        # Handle NaN or infinite values
                        if np.isnan(close_price) or np.isinf(close_price):
                            close_price = 100.0
                        
                        self.current_prices[symbol] = close_price
                    else:
                        # Fallback to first column value
                        try:
                            first_col = data.columns[0]
                            fallback_price = data.iloc[self.current_step][first_col]
                            if isinstance(fallback_price, (list, tuple, np.ndarray)):
                                fallback_price = float(fallback_price[0]) if len(fallback_price) > 0 else 100.0
                            else:
                                fallback_price = float(fallback_price)
                            
                            if np.isnan(fallback_price) or np.isinf(fallback_price):
                                fallback_price = 100.0
                            
                            self.current_prices[symbol] = fallback_price
                        except:
                            self.current_prices[symbol] = 100.0
                else:
                    logger.warning(f"No data available for {symbol} at step {self.current_step}")
                    # Set a default price if data is missing
                    self.current_prices[symbol] = 100.0
        except Exception as e:
            logger.error(f"Error updating current prices: {e}")
            # Set default prices for all symbols if there's an error
            for symbol in self.symbols:
                self.current_prices[symbol] = 100.0

    def _get_state(self):
        """
        Returns the current state of the environment optimized for GPU processing.
        
        Returns:
            A numpy array representing the current state.
        """
        # Ensure current prices are updated
        self._update_current_prices()
        
        # Convert current prices to tensor, ensuring all values are floats
        current_prices_list = []
        for symbol in self.symbols:
            price = self.current_prices[symbol]
            # Ensure price is a single float value
            if isinstance(price, (list, tuple, np.ndarray)):
                price = float(price[0]) if len(price) > 0 else 100.0
            else:
                price = float(price)
            
            # Handle NaN or infinite values
            if np.isnan(price) or np.isinf(price):
                price = 100.0
            
            current_prices_list.append(price)
        
        current_prices_array = np.array(current_prices_list, dtype=np.float32)
        
        # Get price history for lookback window
        price_history = []
        for symbol in self.symbols:
            symbol_data = self.prefetched_data[symbol]  # This is now a numpy array
            start_idx = max(0, self.current_step - self.lookback_window + 1)
            end_idx = self.current_step + 1
            # Use numpy array indexing instead of pandas iloc
            symbol_history = symbol_data[start_idx:end_idx]
            # Pad with the last known price if not enough history
            if len(symbol_history) < self.lookback_window:
                if len(symbol_history) > 0:
                    padding = [symbol_history[-1]] * (self.lookback_window - len(symbol_history))
                    symbol_history = np.concatenate([padding, symbol_history])
                else:
                    # If no data available, use a default price
                    symbol_history = np.full(self.lookback_window, 100.0, dtype=np.float32)
            price_history.append(symbol_history)
        
        price_history_array = np.array(price_history, dtype=np.float32)
        
        # Use GPU acceleration if available
        if self.device == '/GPU:0':
            with tf.device(self.device):
                current_prices_tensor = tf.convert_to_tensor(current_prices_array, dtype=tf.float32)
                price_history_tensor = tf.convert_to_tensor(price_history_array, dtype=tf.float32)
                
                # Update tensors
                self.shares_tensor = tf.convert_to_tensor([self.shares[symbol] for symbol in self.symbols], dtype=tf.float32)
                
                # Calculate state on GPU
                state_tensor = self._get_state_gpu(current_prices_tensor, price_history_tensor)
                return state_tensor.numpy()
        else:
            # Fallback to CPU calculation
            return self._get_state_cpu(current_prices_array, price_history_array)

    def _get_state_cpu(self, current_prices, price_history):
        """
        CPU fallback for state calculation.
        
        Args:
            current_prices: Array of current prices
            price_history: Array of price history
            
        Returns:
            State array
        """
        state = []
        
        # Add current prices
        for price in current_prices:
            state.append(price / self.price_normalization)
        
        # Add current shares
        for symbol in self.symbols:
            state.append(self.shares[symbol] / self.shares_normalization)
        
        # Add current balance
        state.append(self.balance / self.price_normalization)
        
        # Add price history
        for symbol_history in price_history:
            for price in symbol_history:
                state.append(price / self.price_normalization)
        
        return np.array(state, dtype=np.float32)


    def step(self, action):
        """
        Executes one step in the environment.

        Args:
            action: A numpy array of actions for each stock.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Ensure action is the right type and shape
        action = np.array(action, dtype=np.float32)
        if len(action) != len(self.symbols):
            raise ValueError(f"Action length {len(action)} doesn't match number of symbols {len(self.symbols)}")

        # Initialize current_prices if not already done
        if not hasattr(self, 'current_prices') or not self.current_prices:
            self.current_prices = {}
            for symbol in self.symbols:
                self.current_prices[symbol] = 100.0  # Default price

        # Update current prices
        self._update_current_prices()

        # Execute trades
        if self.device == '/GPU:0':
            try:
                current_prices_array = np.array([self.current_prices[symbol] for symbol in self.symbols], dtype=np.float32)
                balance_tensor, shares_tensor = self._execute_trades_gpu(action, current_prices_array)
                # Update both CPU and GPU values
                self.balance = float(balance_tensor.numpy())
                self.shares = {symbol: float(shares_tensor[i].numpy()) for i, symbol in enumerate(self.symbols)}
                self.balance_tensor = balance_tensor
                self.shares_tensor = shares_tensor
                if self.debug:
                    logger.info(f"GPU TRADING: Updated balance to ${self.balance:.2f}, shares: {self.shares}")
                # Ensure GPU tensors are in sync
                with tf.device(self.device):
                    self.balance_tensor = tf.constant(self.balance, dtype=tf.float32)
                    self.shares_tensor = tf.constant([self.shares[symbol] for symbol in self.symbols], dtype=tf.float32)
            except Exception as e:
                logger.warning(f"GPU trading failed, falling back to CPU: {e}")
                self._execute_trades_cpu(action)
        else:
            self._execute_trades_cpu(action)
            
        # Debug: Log trading details
        if self.debug:
            logger.info(f"=== STEP {self.current_step} DEBUG ===")
            logger.info(f"Actions: {action}")
            logger.info(f"Current Prices: {self.current_prices}")
            logger.info(f"Balance: ${self.balance:.2f}")
            logger.info(f"Shares: {self.shares}")
            logger.info(f"Portfolio Value: ${self._calculate_portfolio_value():.2f}")
            logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
            logger.info(f"Return: {((self._calculate_portfolio_value() - self.initial_balance) / self.initial_balance * 100):.2f}%")

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        if self.symbols and self.symbols[0] in self.data:
            done = self.current_step >= len(self.data[self.symbols[0]]) - 1
        else:
            done = True  # End episode if no valid data

        # Calculate reward - only give reward at the end of episode
        if done:
            final_portfolio_value = self._calculate_portfolio_value()
            # Calculate percentage return with bounds to prevent unrealistic values
            if self.initial_balance > 0:
                percentage_return = (final_portfolio_value - self.initial_balance) / self.initial_balance
                # Clamp percentage return to reasonable bounds (-0.5 to 2.0 = -50% to +200%)
                percentage_return = np.clip(percentage_return, -0.5, 2.0)
                reward = percentage_return * self.reward_scaling_factor
            else:
                reward = 0.0
        else:
            # Small intermediate reward based on current portfolio value
            current_portfolio_value = self._calculate_portfolio_value()
            if self.initial_balance > 0:
                current_return = (current_portfolio_value - self.initial_balance) / self.initial_balance
                # Clamp intermediate returns too
                current_return = np.clip(current_return, -0.5, 2.0)
                reward = current_return * 0.001  # Much smaller intermediate reward
            else:
                reward = 0.0
                
        # Debug: Log reward calculation
        if self.debug:
            if done:
                logger.info(f"FINAL REWARD: {reward:.6f} (Portfolio: ${final_portfolio_value:.2f}, Return: {percentage_return*100:.2f}%)")
            else:
                logger.info(f"INTERMEDIATE REWARD: {reward:.6f} (Portfolio: ${current_portfolio_value:.2f}, Return: {current_return*100:.2f}%)")

        # Get next state
        next_state = self._get_state()

        # Calculate current portfolio value for info
        current_portfolio_value = self._calculate_portfolio_value()

        # Prepare info
        info = {
            "portfolio_value": current_portfolio_value,
            "balance": self.balance,
            "shares": self.shares.copy(),
            "current_step": self.current_step
        }

        return next_state, reward, done, info

    def _execute_trades_gpu(self, action, current_prices):
        """
        GPU-accelerated trade execution.
        
        Args:
            action: Action array
            current_prices: Current prices tensor
        """
        with tf.device(self.device):
            # Convert action to tensor with proper data type handling
            if isinstance(action, tf.Tensor):
                action_tensor = tf.cast(action, tf.float32)
            else:
                action_tensor = tf.convert_to_tensor(action, dtype=tf.float32)
            
            # Convert current_prices to tensor if needed
            if not isinstance(current_prices, tf.Tensor):
                current_prices = tf.convert_to_tensor(current_prices, dtype=tf.float32)
            
            # Calculate trade amounts with limits to prevent excessive trading
            trade_amounts = action_tensor * self.max_shares_per_trade
            # Limit trade amounts to prevent excessive portfolio growth
            max_trade_amount = 10.0  # Maximum shares per trade
            trade_amounts = tf.clip_by_value(trade_amounts, -max_trade_amount, max_trade_amount)
            
            # FIRST: Prevent selling more shares than we own (apply share protection BEFORE any other calculations)
            max_sellable = self.shares_tensor  # Can't sell more than we own
            trade_amounts = tf.where(
                trade_amounts < 0,  # For sells
                tf.maximum(trade_amounts, -max_sellable),  # Limit sells to available shares
                trade_amounts  # Keep buys unchanged
            )
            
            # Don't allow buying if balance is zero (tensor-safe)
            balance_nonpos = tf.less_equal(self.balance_tensor, 0.0)
            trade_amounts = tf.where(balance_nonpos, tf.minimum(trade_amounts, 0.0), trade_amounts)
            
            # Calculate trade costs (now with share-protected amounts)
            trade_costs = trade_amounts * current_prices
            
            # DEBUG: Log detailed GPU trading calculations
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"GPU TRADE DEBUG:")
                logger.info(f"  Initial Balance: ${float(self.balance_tensor.numpy()):.2f}")
                logger.info(f"  Action: {action}")
                logger.info(f"  Trade Amounts: {trade_amounts.numpy()}")
                logger.info(f"  Current Prices: {current_prices.numpy()}")
                logger.info(f"  Trade Costs: {trade_costs.numpy()}")
                logger.info(f"  Total Cost (buys only): ${float(tf.reduce_sum(tf.maximum(trade_costs, 0)).numpy()):.2f}")
            
            # Check if we have enough balance for purchases
            buy_costs = tf.maximum(trade_costs, 0)  # Only positive costs (buys)
            total_buy_cost = tf.reduce_sum(buy_costs)
            
            # If we don't have enough balance, scale down ONLY the purchases
            # Use only 90% of balance to leave some cash for future trades
            available_for_trades = self.balance_tensor * 0.9
            need_scale = tf.greater(total_buy_cost, available_for_trades)
            safe_scale = tf.where(need_scale, available_for_trades / (total_buy_cost + 1e-8), 1.0)
            
            # Scale down only the buy amounts, keep sell amounts unchanged (sells are already protected)
            final_trade_amounts = tf.where(trade_amounts > 0, trade_amounts * safe_scale, trade_amounts)
            final_trade_costs = final_trade_amounts * current_prices
            
            # Calculate new balance - now net cost should NEVER be negative since we can't sell what we don't have
            new_balance = tf.maximum(self.balance_tensor - tf.reduce_sum(final_trade_costs), 0.0)
            
            # Update shares
            new_shares = tf.maximum(self.shares_tensor + final_trade_amounts, 0.0)
            
            # DEBUG: Log final calculations
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"  Final Calculations:")
                logger.info(f"    Need Scale: {need_scale.numpy()}")
                logger.info(f"    Scale Factor: {float(safe_scale.numpy()):.4f}")
                logger.info(f"    Available Shares: {self.shares_tensor.numpy()}")
                logger.info(f"    Final Trade Amounts: {final_trade_amounts.numpy()}")
                logger.info(f"    Final Trade Costs: {final_trade_costs.numpy()}")
                logger.info(f"    Net Cost: ${float(tf.reduce_sum(final_trade_costs).numpy()):.2f}")
                logger.info(f"    Old Balance: ${float(self.balance_tensor.numpy()):.2f}")
                logger.info(f"    New Balance: ${float(new_balance.numpy()):.2f}")
                
                # Log if we have net gains from selling (negative cost = money gained)
                net_cost = tf.reduce_sum(final_trade_costs)
                if net_cost < 0:
                    logger.info(f"    PROFIT: Net gain from selling: ${float(-net_cost):.2f}")
                    sells = tf.reduce_sum(tf.minimum(final_trade_costs, 0))  # Sum of sell proceeds
                    buys = tf.reduce_sum(tf.maximum(final_trade_costs, 0))   # Sum of buy costs
                    logger.info(f"    Breakdown: Sells=${float(-sells):.2f}, Buys=${float(buys):.2f}")
            
            # Return updated values
            return new_balance, new_shares

    def _execute_trades_cpu(self, action):
        """
        CPU fallback for trade execution.
        
        Args:
            action: Action array
        """
        if self.debug:
            logger.info(f"CPU TRADING - Initial Balance: ${self.balance:.2f}, Shares: {self.shares}")
            
        for i, symbol in enumerate(self.symbols):
            if symbol not in self.current_prices:
                continue
                
            current_price = self.current_prices[symbol]
            trade_amount = action[i] * self.max_shares_per_trade
            # Limit trade amounts to prevent excessive portfolio growth
            max_trade_amount = 10.0  # Maximum shares per trade
            trade_amount = np.clip(trade_amount, -max_trade_amount, max_trade_amount)
            
            # Don't allow buying if balance is zero
            if self.balance <= 0 and trade_amount > 0:
                trade_amount = 0  # Can't buy without money
            
            # Calculate trade cost with transaction fees
            trade_cost = trade_amount * current_price
            # Add transaction fees (0.1% of trade value)
            transaction_fee = abs(trade_cost) * 0.001
            total_cost = trade_cost + transaction_fee
            
            # Check if we have enough balance for purchases
            if total_cost > 0 and total_cost > self.balance:
                # Scale down the purchase
                scale_factor = self.balance / total_cost
                trade_amount *= scale_factor
                trade_cost = trade_amount * current_price
                transaction_fee = abs(trade_cost) * 0.001
                total_cost = trade_cost + transaction_fee
            
            # Execute the trade
            old_balance = self.balance
            old_shares = self.shares[symbol]
            self.balance -= total_cost
            self.shares[symbol] += trade_amount
            if self.debug:
                logger.info(f"TRADE EXECUTED: {symbol} {trade_amount:+.2f} shares, Balance: ${old_balance:.2f} -> ${self.balance:.2f}, Shares: {old_shares:.2f} -> {self.shares[symbol]:.2f}")
            
            # Prevent negative balance and shares
            self.balance = max(self.balance, 0.0)
            self.shares[symbol] = max(self.shares[symbol], 0.0)
            
            # Debug: Log individual trades
            if self.debug and trade_amount != 0:
                logger.info(f"TRADE {symbol}: {trade_amount:+.2f} shares @ ${current_price:.2f} = ${trade_cost:+.2f}")
                logger.info(f"  Balance: ${old_balance:.2f} -> ${self.balance:.2f}")
                logger.info(f"  Shares: {old_shares:.2f} -> {self.shares[symbol]:.2f}")
                
        if self.debug:
            logger.info(f"CPU TRADING - Final Balance: ${self.balance:.2f}, Shares: {self.shares}")

    def _calculate_portfolio_value(self):
        """
        Calculates the current portfolio value with GPU acceleration.
        
        Returns:
            The current portfolio value.
        """
        if not self.current_prices:
            return self.balance
        
        # Use GPU acceleration if available
        if self.device == '/GPU:0':
            try:
                with tf.device(self.device):
                    current_prices_array = np.array([self.current_prices[symbol] for symbol in self.symbols], dtype=np.float32)
                    current_prices_tensor = tf.convert_to_tensor(current_prices_array, dtype=tf.float32)
                    # Ensure shares tensor is up to date
                    shares_array = np.array([self.shares[symbol] for symbol in self.symbols], dtype=np.float32)
                    self.shares_tensor = tf.convert_to_tensor(shares_array, dtype=tf.float32)
                    portfolio_value = self._calculate_portfolio_value_gpu(current_prices_tensor)
                    portfolio_value = float(portfolio_value.numpy())
            except Exception as e:
                logger.warning(f"GPU portfolio calculation failed, falling back to CPU: {e}")
                # Fall back to CPU calculation
                portfolio_value = self.balance
                for symbol in self.symbols:
                    if symbol in self.current_prices:
                        portfolio_value += self.shares[symbol] * self.current_prices[symbol]
        else:
            # CPU fallback
            portfolio_value = self.balance
            for symbol in self.symbols:
                if symbol in self.current_prices:
                    portfolio_value += self.shares[symbol] * self.current_prices[symbol]
        
        # Debug: Log portfolio calculation details
        if self.debug:
            logger.info(f"PORTFOLIO CALCULATION:")
            logger.info(f"  Cash Balance: ${self.balance:.2f}")
            for symbol in self.symbols:
                if symbol in self.current_prices and self.shares[symbol] > 0:
                    stock_value = self.shares[symbol] * self.current_prices[symbol]
                    logger.info(f"  {symbol}: {self.shares[symbol]:.2f} shares @ ${self.current_prices[symbol]:.2f} = ${stock_value:.2f}")
            logger.info(f"  TOTAL PORTFOLIO: ${portfolio_value:.2f}")
            
        return portfolio_value

    def _get_data(self):
        """
        Fetches historical stock data with optimized caching and parallel processing.
        
        Returns:
            tuple: (data_dict, failed_symbols)
        """
        logger.info("Fetching historical stock data...")
        start_time = time.time()
        
        # Use yfinance for faster data fetching
        if USE_YFINANCE:
            data, failed_symbols = self._get_yfinance_data()
        else:
            data, failed_symbols = self._get_alpaca_data()
        
        fetch_time = time.time() - start_time
        logger.info(f"Data fetching completed in {fetch_time:.2f} seconds")
        
        return data, failed_symbols
    
    def _get_alpaca_data(self):
        """Fetches historical stock data from Alpaca Markets API with caching.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        # BASE_URL for Alpaca Data API stock bars
        BARS_BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

        if all(symbol in StockTradingEnvironment._global_data_cache for symbol in self.symbols):
            logger.info("Using in-memory cached data for all symbols")
            return {symbol: StockTradingEnvironment._global_data_cache[symbol] for symbol in self.symbols}

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
                    StockTradingEnvironment._global_data_cache[symbol] = stock_data
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
            return data, []

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
        failed_symbols = []
        for symbol, stock_data in results:
            if stock_data is not None:
                data[symbol] = stock_data
                StockTradingEnvironment._global_data_cache[symbol] = stock_data
            else:
                logger.warning(f"Skipping {symbol} due to failed data download or empty response.")
                failed_symbols.append(symbol)

        if not data: # If no data was fetched for any symbol
            raise ValueError("Could not fetch valid data for any of the specified symbols.")

        return data, failed_symbols
    
# Removed merge function - not used in current implementation

    def _get_yfinance_data(self):
        """Gets historical stock data for all symbols with caching using yfinance.

        Returns:
            A dictionary where keys are stock symbols and values are pandas DataFrames of historical data.
        """
        data = {}
        failed_symbols = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=729)
        for symbol in self.symbols:
            cache_file = self.cache_dir / f"{symbol}_yfinance.pkl"
            try:
                if cache_file.exists():
                    stock_data = pd.read_pickle(cache_file)
                    if not stock_data.empty:
                        data[symbol] = stock_data
                        StockTradingEnvironment._global_data_cache[symbol] = stock_data
                        logger.info(f"Loaded cached yfinance data for {symbol} from {cache_file}")
                    else:
                        raise ValueError("Cached data is empty.")
                else:
                    stock_data = yf.download(symbol, start=start_date, end=end_date, interval="1h", progress=False, auto_adjust=True)
                    if not stock_data.empty:
                        data[symbol] = stock_data
                        stock_data.to_pickle(cache_file)
            except Exception as e:
                logger.error(f"Failed to download or load data for {symbol}: {e}")
                failed_symbols.append(symbol)
        return data, failed_symbols

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
                # Check if symbol data exists
                if symbol not in self.data:
                    logger.warning(f"No data available for {symbol}, using dummy data")
                    prefetched_data[symbol] = np.full(
                        self.lookback_window + 1, 100.0, dtype=np.float32
                    )
                    continue
                    
                # Check for different possible column names
                symbol_data = self.data[symbol]
                close_column = None
                
                # Handle multi-level column names (like yfinance data)
                if hasattr(symbol_data.columns, 'levels') and len(symbol_data.columns.levels) > 1:
                    # Multi-level columns - look for Close in the first level
                    for col in symbol_data.columns:
                        if isinstance(col, tuple) and 'Close' in col[0]:
                            close_column = col
                            break
                    # If not found, try to flatten the columns
                    if close_column is None:
                        # Flatten multi-level columns
                        symbol_data.columns = [col[1] if isinstance(col, tuple) and len(col) > 1 else col for col in symbol_data.columns]
                        # Now try to find Close column
                        for col_name in ["Close", "close", "c", "C"]:
                            if col_name in symbol_data.columns:
                                close_column = col_name
                                break
                else:
                    # Single-level columns - try different possible column names
                    for col_name in ["Close", "close", "c", "C"]:
                        if col_name in symbol_data.columns:
                            close_column = col_name
                            break
                
                if close_column is not None:
                    # Ensure we get a clean numpy array
                    close_prices = symbol_data[close_column].values
                    if isinstance(close_prices, (list, tuple)):
                        close_prices = np.array(close_prices, dtype=np.float32)
                    else:
                        close_prices = np.asarray(close_prices, dtype=np.float32)
                    
                    # Handle any NaN values
                    if np.any(np.isnan(close_prices)):
                        # Replace NaN with forward fill, then backward fill (using newer pandas syntax)
                        close_prices = pd.Series(close_prices).ffill().bfill().values
                        close_prices = np.asarray(close_prices, dtype=np.float32)
                    
                    # Ensure we have enough data
                    if len(close_prices) < self.lookback_window:
                        # Pad with the last known price
                        padding = np.full(self.lookback_window - len(close_prices), close_prices[-1] if len(close_prices) > 0 else 100.0, dtype=np.float32)
                        close_prices = np.concatenate([padding, close_prices])
                    
                    prefetched_data[symbol] = close_prices
                    logger.debug(f"Prefetched {len(close_prices)} close prices for {symbol}")
                else:
                    logger.warning(f"No close price column found for {symbol}. Available columns: {list(symbol_data.columns)}")
                    # Provide dummy data
                    prefetched_data[symbol] = np.full(
                        self.lookback_window + 1, 100.0, dtype=np.float32
                    )
            except Exception as e:
                logger.error(f"Error prefetching data for {symbol}: {e}", exc_info=True)
                # Create dummy data if prefetching fails to prevent crashes
                prefetched_data[symbol] = np.full(
                    self.lookback_window + 1, 100.0, dtype=np.float32
                )

        return prefetched_data

# Removed unused fetch function