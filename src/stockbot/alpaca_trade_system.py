import logging
from datetime import datetime, timedelta
import numpy as np
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import time # Added for retry logic
import pandas as pd # Added for MultiIndex handling

# Add yfinance as fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    INITIAL_BALANCE,
    LOOKBACK_WINDOW,
    MAX_SHARES_PER_TRADE,
    ALLOW_SHORT_SELLING,
    PDT_PROTECTION_ENABLED,
)

logger = logging.getLogger(__name__)


class AlpacaTradingSystem:
    """A class for executing live trades using the Alpaca API based on agent's decisions."""

    def __init__(self, agent, symbols, trading_client, stock_data_client):
        """
        Initializes the AlpacaTradingSystem.

        Args:
            agent: The DRL agent responsible for making trading decisions.
            symbols: A list of stock symbols to trade.
            trading_client: An instance of TradingClient for executing trades.
            stock_data_client: An instance of StockHistoricalDataClient for fetching data.
        """
        self.agent = agent
        self.symbols = symbols
        self.trading_client = trading_client
        self.stock_data_client = stock_data_client
        
        # PDT protection tracking
        self.daily_trade_direction = {}  # Track buy/sell direction per symbol per day
        self.last_trading_day = None  # Track when we last reset the daily tracking

    def _get_current_state(self):
        """Creates the current state based on market data and account information using Alpaca API.

        Returns:
            A numpy array representing the current state for the agent.
        """
        state = []

        # Get account information from Alpaca with retry logic
        balance = INITIAL_BALANCE  # Default fallback
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                account = self.trading_client.get_account()
                balance = float(account.cash)
                logger.info(f"Successfully retrieved account balance: ${balance:.2f}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to get account info (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to get account info after {max_retries} attempts, using default balance")
                    balance = INITIAL_BALANCE

        # Get current positions from Alpaca with retry logic
        positions = {sym: 0 for sym in self.symbols}
        for attempt in range(max_retries):
            try:
                current_positions = self.trading_client.get_all_positions()
                for position in current_positions:
                    if position.symbol in self.symbols:
                        positions[position.symbol] = int(float(position.qty))
                logger.info(f"Successfully retrieved positions for {len(positions)} symbols")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to get positions (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get positions after {max_retries} attempts, using zero positions")

        # Get current price data - try yfinance first, fallback to Alpaca
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_WINDOW + 5)  # Add buffer days

        current_prices = {}
        historical_data = {}
        yfinance_success = False
        
        # Try yfinance first
        try:
            current_prices, historical_data = self._get_price_data_yfinance(self.symbols, end_date, start_date)
            if current_prices and len(current_prices) > 0:
                yfinance_success = True
                logger.info("Successfully retrieved data from yfinance")
        except Exception as e:
            logger.warning(f"yfinance failed, trying Alpaca fallback: {e}")
        
        # If yfinance failed, try Alpaca fallback
        if not yfinance_success:
            logger.info("Using Alpaca fallback for price data...")
            alpaca_success = False
            
            for attempt in range(max_retries):
                try:
                    # Get current price data from Alpaca for all symbols at once
                    request_params = StockBarsRequest(
                        symbol_or_symbols=self.symbols,
                        timeframe=TimeFrame.Hour,  # Use hour data for current price
                        start=end_date - timedelta(hours=1),  # Get last hour to ensure we get data
                        end=end_date,
                        feed='iex',
                    )
                    latest_bars_obj = self.stock_data_client.get_stock_bars(request_params)
                    latest_bars = latest_bars_obj.df

                    # Handle different data formats from Alpaca
                    if not latest_bars.empty:
                        # Check if we have a multi-level index (symbol, timestamp)
                        if isinstance(latest_bars.index, pd.MultiIndex):
                            symbols_in_data = latest_bars.index.get_level_values("symbol").unique()
                        else:
                            # Single symbol request
                            symbols_in_data = [self.symbols[0]] if len(self.symbols) == 1 else []
                        
                        for symbol in self.symbols:
                            if symbol in symbols_in_data:
                                try:
                                    if isinstance(latest_bars.index, pd.MultiIndex):
                                        symbol_data = latest_bars.loc[symbol]
                                    else:
                                        symbol_data = latest_bars
                                    
                                    if not symbol_data.empty:
                                        current_price = symbol_data["close"].iloc[-1]
                                        current_prices[symbol] = current_price
                                        logger.info(f"Alpaca current price for {symbol}: ${current_price:.2f}")
                                    else:
                                        raise ValueError(f"No data for {symbol}")
                                except Exception as e:
                                    logger.warning(f"Error processing {symbol}: {e}")
                                    current_prices[symbol] = 100.0
                            else:
                                logger.warning(f"No real-time price data for {symbol} from Alpaca. Using default price.")
                                current_prices[symbol] = 100.0
                    else:
                        raise ValueError("No data received from Alpaca")
                    
                    logger.info(f"Successfully retrieved prices for {len(current_prices)} symbols from Alpaca")
                    alpaca_success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to get price data from Alpaca (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to get price data from Alpaca after {max_retries} attempts")
            
            # If Alpaca also failed, get historical data from Alpaca as fallback
            if alpaca_success:
                for symbol in self.symbols:
                    try:
                        request_params = StockBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=TimeFrame.Hour,
                            start=start_date,
                            end=end_date,
                        )
                        hist_data = self.stock_data_client.get_stock_bars(request_params).df

                        if not hist_data.empty and len(hist_data) >= LOOKBACK_WINDOW:
                            historical_data[symbol] = hist_data["close"].values[-LOOKBACK_WINDOW:]
                        else:
                            logger.warning(f"Not enough history for {symbol}. Using available data.")
                            if not hist_data.empty:
                                available_prices = hist_data["close"].values
                                first_value = available_prices[0] if len(available_prices) > 0 else 100.0
                                padding = np.full(LOOKBACK_WINDOW - len(available_prices), first_value)
                                historical_data[symbol] = np.concatenate([padding, available_prices])
                            else:
                                historical_data[symbol] = np.full(LOOKBACK_WINDOW, 100.0)  # Default price

                    except Exception as e:
                        logger.error(f"Error processing price history for {symbol}: {e}")
                        historical_data[symbol] = np.full(LOOKBACK_WINDOW, 100.0)
        
        # Use retrieved data for state
        for symbol in self.symbols:
            price = current_prices.get(symbol, 100.0)
            state.append(float(price) / 1000.0)  # Normalize price
            state.append(positions[symbol] / 100.0)  # Normalize position

        # Add historical data
        for symbol in self.symbols:
            if symbol in historical_data:
                prices = historical_data[symbol]
                # Normalize prices
                price_max = np.max(prices) if prices.size > 0 else 1.0
                normalized_prices = prices / max(price_max, 1e-8)  # Avoid div by zero
                state.extend(normalized_prices)
            else:
                # Add placeholder data
                state.extend([0.0] * LOOKBACK_WINDOW)

        state.append(balance / INITIAL_BALANCE)  # Normalized balance

        return np.array(state, dtype=np.float32), current_prices

    def _check_and_reset_daily_tracking(self):
        """
        Check if we need to reset daily trade direction tracking.
        Resets tracking at the start of each trading day (9:30 AM ET).
        """
        if not PDT_PROTECTION_ENABLED:
            return
            
        current_time = datetime.now()
        current_trading_day = current_time.strftime('%Y-%m-%d')
        
        # Reset tracking if it's a new trading day
        if self.last_trading_day != current_trading_day:
            if self.daily_trade_direction:
                logger.info(f"New trading day detected. Resetting daily trade direction tracking.")
            self.daily_trade_direction = {}
            self.last_trading_day = current_trading_day

    def _check_pdt_restriction(self, symbol, trade_direction, portfolio_value):
        """
        Check if a trade should be blocked due to PDT restrictions.
        
        Args:
            symbol: Stock symbol
            trade_direction: 'BUY' or 'SELL'
            portfolio_value: Current portfolio value
            
        Returns:
            bool: True if trade should be blocked, False if allowed
        """
        if not PDT_PROTECTION_ENABLED or portfolio_value >= 25000:
            return False
            
        # Check if we've already traded this symbol today
        if symbol in self.daily_trade_direction:
            if self.daily_trade_direction[symbol] != trade_direction:
                logger.info(f"[{symbol}] PDT Protection: Skipping {trade_direction} - already traded {self.daily_trade_direction[symbol]} today (Portfolio: ${portfolio_value:.2f})")
                return True
        else:
            # First trade of the day for this symbol - log that we're attempting it
            logger.info(f"[{symbol}] PDT Protection: First trade of day - attempting {trade_direction} (Portfolio: ${portfolio_value:.2f})")
            
        return False

    def _record_successful_trade(self, symbol, trade_direction):
        """
        Record a successful trade for PDT protection tracking.
        
        Args:
            symbol: Stock symbol
            trade_direction: 'BUY' or 'SELL'
        """
        if PDT_PROTECTION_ENABLED:
            self.daily_trade_direction[symbol] = trade_direction
            logger.info(f"[{symbol}] PDT Protection: Successfully recorded {trade_direction} trade for today")

    def _get_price_data_yfinance(self, symbols, end_date, start_date):
        """
        Get price data from yfinance as primary data source.
        
        Args:
            symbols: List of stock symbols
            end_date: End date for data
            start_date: Start date for data
            
        Returns:
            Dictionary of current prices and historical data
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return {}, {}
        
        current_prices = {}
        historical_data = {}
        
        try:
            logger.info("Getting live price data from yfinance...")
            
            # Get current prices
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    # Get current price
                    current_price = ticker.info.get('regularMarketPrice', 100.0)
                    if current_price is None:
                        current_price = 100.0
                    current_prices[symbol] = current_price
                    logger.info(f"yfinance current price for {symbol}: ${current_price:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to get yfinance price for {symbol}: {e}")
                    current_prices[symbol] = 100.0
            
            # Get historical data
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, interval='1h')
                    
                    if not hist.empty and len(hist) >= LOOKBACK_WINDOW:
                        historical_data[symbol] = hist['Close'].values[-LOOKBACK_WINDOW:]
                    else:
                        logger.warning(f"Insufficient yfinance history for {symbol}")
                        historical_data[symbol] = np.full(LOOKBACK_WINDOW, current_prices.get(symbol, 100.0))
                        
                except Exception as e:
                    logger.warning(f"Failed to get yfinance history for {symbol}: {e}")
                    historical_data[symbol] = np.full(LOOKBACK_WINDOW, current_prices.get(symbol, 100.0))
            
            logger.info(f"Successfully retrieved yfinance data for {len(current_prices)} symbols")
            return current_prices, historical_data
            
        except Exception as e:
            logger.error(f"yfinance data retrieval failed: {e}")
            return {}, {}

    def execute_trades(self):
        """Executes trades based on the agent's decisions while avoiding wash trades.

        Returns:
            A list of action values (floats) for each symbol.
        """
        # Check and reset daily tracking for PDT protection
        self._check_and_reset_daily_tracking()
        
        # Get current state
        state, current_prices = self._get_current_state()

        # Get actions from agent (no exploration in live trading)
        actions = self.agent.act(state, explore=False)

        if actions is None:
            return []

        # Get current positions to avoid wash trades
        positions = {}
        try:
            current_positions = self.trading_client.get_all_positions()
            for position in current_positions:
                positions[position.symbol] = {"qty": int(float(position.qty))}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            positions = {sym: {"qty": 0} for sym in self.symbols}
            
        # Calculate current portfolio value for PDT protection
        portfolio_value = 0.0
        try:
            account = self.trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
        except Exception as e:
            logger.warning(f"Could not get portfolio value for PDT check: {e}")
            portfolio_value = 1000.0  # Default fallback

        # Execute actions
        trades_attempted = 0
        trades_executed = 0
        
        for i, symbol in enumerate(self.symbols):
            signal = actions[i]

            # Get current position
            current_shares = positions.get(symbol, {}).get("qty", 0)

            if signal > 0:  # Buy signal
                # Check PDT protection
                if self._check_pdt_restriction(symbol, "BUY", portfolio_value):
                    continue  # Skip this trade due to PDT restriction
                    
                shares_to_buy = max(1, int(signal * MAX_SHARES_PER_TRADE))
                logger.debug(f"BUY SIGNAL {symbol}: signal={signal:.3f}, shares_to_buy={shares_to_buy}")
                trades_attempted += 1
                if shares_to_buy > 0:
                    current_price = current_prices.get(symbol)
                    if current_price is not None and current_price > 0:
                        try:
                            # Submit order
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                qty=shares_to_buy,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.DAY,
                            )
                            self.trading_client.submit_order(
                                order_data=market_order_data
                            )
                            logger.info(
                                f"[{symbol}] BUY ORDER: {shares_to_buy} shares"
                            )
                            trades_executed += 1
                            # Record successful trade for PDT protection
                            self._record_successful_trade(symbol, "BUY")

                        except Exception as e:
                            logger.error(f"[{symbol}] Buy order failed: {e}")
            elif signal < 0:  # Sell signal
                # Check if we can sell (either have shares or short selling is allowed)
                if current_shares > 0 or ALLOW_SHORT_SELLING:
                    # Check PDT protection
                    if self._check_pdt_restriction(symbol, "SELL", portfolio_value):
                        continue  # Skip this trade due to PDT restriction
                        
                    shares_to_sell = max(1, int(
                        -signal * min(MAX_SHARES_PER_TRADE, current_shares if not ALLOW_SHORT_SELLING else MAX_SHARES_PER_TRADE)
                    ))
                    logger.debug(f"SELL SIGNAL {symbol}: signal={signal:.3f}, shares_to_sell={shares_to_sell}, current_shares={current_shares}")
                    trades_attempted += 1
                    if shares_to_sell > 0:
                        current_price = current_prices.get(symbol)
                    if current_price is not None and current_price > 0:
                        try:
                            # Submit order
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                qty=shares_to_sell,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                            )
                            self.trading_client.submit_order(
                                order_data=market_order_data
                            )
                            logger.info(
                                f"[{symbol}] SELL ORDER: {shares_to_sell} shares"
                            )
                            trades_executed += 1
                            # Record successful trade for PDT protection
                            self._record_successful_trade(symbol, "SELL")

                        except Exception as e:
                            logger.error(f"[{symbol}] Sell order failed: {e}")
        
        # Return the original action values for logging purposes
        logger.info(f"Trading Summary: {trades_executed} trades executed out of {len(self.symbols)} symbols analyzed")
        return actions