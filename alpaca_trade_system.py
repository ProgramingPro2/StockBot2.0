import logging
from datetime import datetime, timedelta
import numpy as np
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import (
    INITIAL_BALANCE,
    LOOKBACK_WINDOW,
    MAX_SHARES_PER_TRADE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
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

    def _get_current_state(self):
        """Creates the current state based on market data and account information using Alpaca API.

        Returns:
            A numpy array representing the current state for the agent.
        """
        state = []

        # Get account information from Alpaca
        try:
            account = self.trading_client.get_account()
            balance = float(account.cash)
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            balance = INITIAL_BALANCE  # Use default if API fails

        # Get current positions from Alpaca
        positions = {sym: 0 for sym in self.symbols}
        try:
            current_positions = self.trading_client.get_all_positions()
            for position in current_positions:
                if position.symbol in self.symbols:
                    positions[position.symbol] = int(float(position.qty))
        except Exception as e:
            logger.error(f"Error getting positions: {e}")

        # Get current price data from Alpaca
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_WINDOW + 5)  # Add buffer days

        # Get current price data from Alpaca for all symbols at once
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbols,
                timeframe=TimeFrame.Hour,  # Use hour data for current price
                start=end_date
                - timedelta(hours=1),  # Get last hour to ensure we get data
                end=end_date,
                feed='iex',
            )
            latest_bars_obj = self.stock_data_client.get_stock_bars(request_params)
            latest_bars = latest_bars_obj.df

            for symbol in self.symbols:
                if symbol in latest_bars.index.get_level_values("symbol"):
                    current_price = latest_bars.loc[symbol]["close"].iloc[-1]
                    state.append(float(current_price) / 1000.0)  # Normalize price
                    state.append(positions[symbol] / 100.0)  # Normalize position
                    logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                else:
                    logger.warning(
                        f"No real-time price data for {symbol} from Alpaca. Using default price."
                    )
                    state.append(100.0 / 1000.0)  # Default normalized price
                    state.append(
                        positions[symbol] / 100.0
                    )  # Normalize position

        except Exception as e:
            logger.error(
                f"Error getting current price data for all symbols from Alpaca: {e}. Using default prices."
            )
            for symbol in self.symbols:
                state.append(100.0 / 1000.0)  # Default normalized price
                state.append(positions[symbol] / 100.0)  # Normalize position

        state.append(balance / INITIAL_BALANCE)  # Normalized balance

        # Add historical price data
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
                    prices = hist_data["close"].values[-LOOKBACK_WINDOW:]
                else:
                    logger.warning(
                        f"Not enough history for {symbol}. Using available data."
                    )
                    if not hist_data.empty:
                        available_prices = hist_data["close"].values
                        first_value = (
                            available_prices[0] if len(available_prices) > 0 else 100.0
                        )
                        padding = np.full(
                            LOOKBACK_WINDOW - len(available_prices), first_value
                        )
                        prices = np.concatenate([padding, available_prices])
                    else:
                        prices = np.full(LOOKBACK_WINDOW, 100.0)  # Default price

                # Normalize prices
                price_max = np.max(prices) if prices.size > 0 else 1.0
                normalized_prices = prices / max(price_max, 1e-8)  # Avoid div by zero

                state.extend(normalized_prices)

            except Exception as e:
                logger.error(f"Error processing price history for {symbol}: {e}")
                # Add placeholder data to maintain state vector size
                state.extend([0.0] * LOOKBACK_WINDOW)

        return np.array(state, dtype=np.float32)

    def execute_trades(self):
        """Executes trades based on the agent's decisions while avoiding wash trades.

        Returns:
            A list of actions taken (buy/sell signals) for each symbol.
        """
        # Get current state
        state = self._get_current_state()

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

        # Create trading log for this cycle
        trade_actions = []

        # Execute actions
        for i, symbol in enumerate(self.symbols):
            signal = actions[i]

            # Get current position
            current_shares = positions.get(symbol, {}).get("qty", 0)

            if signal > 0:  # Buy signal
                shares_to_buy = int(signal * MAX_SHARES_PER_TRADE)
                if shares_to_buy > 0:
                    current_price = None
                    try:
                        # Attempt to get latest price with Alpaca
                        request_params = StockBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=TimeFrame.Hour,
                            start=datetime.now() - timedelta(hours=1),
                            end=datetime.now(),
                            feed='iex',
                        )
                        bars = self.stock_data_client.get_stock_bars(request_params).df
                        if not bars.empty:
                            current_price = bars["close"].iloc[-1]
                            logger.info(
                                f"Latest price for {symbol} from Alpaca: ${current_price:.2f}"
                            )
                        else:
                            logger.warning(
                                f"No real-time price data for {symbol} from Alpaca. Skipping buy order."
                            )
                            continue  # Skip to next symbol if no price data
                    except Exception as e:
                        logger.warning(
                            f"Error getting price data for {symbol} from Alpaca: {e}. Attempting yfinance fallback."
                        )

                    if current_price is None:
                        logger.warning(
                            f"No real-time price data for {symbol} from Alpaca. Skipping buy order."
                        )
                        continue  # Skip to next symbol if no price data

                    if current_price is not None and current_price > 0:
                        try:
                            # Submit order
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                qty=shares_to_buy,
                                side=OrderSide.Buy,
                                time_in_force=TimeInForce.DAY,
                            )
                            self.trading_client.submit_order(
                                order_data=market_order_data
                            )
                            logger.info(
                                f"BUY ORDER: {shares_to_buy} shares of {symbol}"
                            )
                            trade_actions.append(
                                f"BUY {symbol}: {shares_to_buy} shares"
                            )

                        except Exception as e:
                            logger.error(f"Buy order failed for {symbol}: {e}")
            elif signal < 0:  # Sell signal
                shares_to_sell = int(
                    -signal * min(MAX_SHARES_PER_TRADE, current_shares)
                )
                if shares_to_sell > 0:
                    current_price = None
                    try:
                        # Attempt to get latest price with Alpaca
                        request_params = StockBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=TimeFrame.Hour,
                            start=datetime.now() - timedelta(hours=1),
                            end=datetime.now(),
                            feed='iex',
                        )
                        bars = self.stock_data_client.get_stock_bars(request_params).df
                        if not bars.empty:
                            current_price = bars["close"].iloc[-1]
                            logger.info(
                                f"Latest price for {symbol} from Alpaca: ${current_price:.2f}"
                            )
                        else:
                            logger.warning(
                                f"No real-time price data for {symbol} from Alpaca. Skipping buy order."
                            )
                            continue  # Skip to next symbol if no price data
                    except Exception as e:
                        logger.warning(
                            f"Error getting price data for {symbol} from Alpaca: {e}. Attempting yfinance fallback."
                        )

                    if current_price is None:
                        logger.warning(
                            f"No real-time price data for {symbol} from Alpaca. Skipping sell order."
                        )
                        continue  # Skip to next symbol if no price data

                    if current_price is not None and current_price > 0:
                        try:
                            # Submit order
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                qty=shares_to_sell,
                                side=OrderSide.Sell,
                                time_in_force=TimeInForce.DAY,
                            )
                            self.trading_client.submit_order(
                                order_data=market_order_data
                            )
                            logger.info(
                                f"SELL ORDER: {shares_to_sell} shares of {symbol}"
                            )
                            trade_actions.append(
                                f"SELL {symbol}: {shares_to_sell} shares"
                            )

                        except Exception as e:
                            logger.error(f"Sell order failed for {symbol}: {e}")
        return trade_actions