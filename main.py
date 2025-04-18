import os
import pandas as pd
import numpy as np
import time
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, concatenate
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import yfinance as yf
import dotenv
import joblib
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', filename='app.log')
logger = logging.getLogger(__name__)

# Configure GPU memory growth to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU is available. Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.warning(f"Memory growth setting failed: {e}")
else:
    logger.info("No GPU available, using CPU for training")

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Load environment variables
dotenv.load_dotenv()

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Configuration parameters
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "TSLA", "NVDA", "INTC", "AMD"]
INITIAL_BALANCE = 100000.0
LOOKBACK_WINDOW = 10
TRADING_FREQUENCY = "minute"
MAX_SHARES_PER_TRADE = 100
TRAINING_EPISODES = 1000
TEST_EPISODES = 1000

# Cache directory for stock data
CACHE_DIR = Path("./data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# RL model parameters
STATE_DIM = (len(STOCK_SYMBOLS) * 2) + 1 + (len(STOCK_SYMBOLS) * LOOKBACK_WINDOW)
ACTION_DIM = len(STOCK_SYMBOLS) * 3
LEARNING_RATE = 0.001
GAMMA = 0.95
BATCH_SIZE = 128  # Increased batch size for better GPU utilization
MEMORY_SIZE = 100000  # Increased memory size for better experience diversity

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Save replay buffer to disk"""
        joblib.dump(self.buffer, path)
    
    def load(self, path):
        """Load replay buffer from disk"""
        if os.path.exists(path):
            self.buffer = joblib.load(path)
            self.position = len(self.buffer) % self.capacity
            return True
        return False

class DRLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Create separate optimizers for actor and critic
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Build models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # TensorFlow checkpoint for saving models
        self.checkpoint_dir = "./checkpoints"
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.actor_checkpoint = tf.train.Checkpoint(model=self.actor, optimizer=self.actor_optimizer)
        self.critic_checkpoint = tf.train.Checkpoint(model=self.critic, optimizer=self.critic_optimizer)
        self.actor_manager = tf.train.CheckpointManager(
            self.actor_checkpoint, directory=f"{self.checkpoint_dir}/actor", max_to_keep=3
        )
        self.critic_manager = tf.train.CheckpointManager(
            self.critic_checkpoint, directory=f"{self.checkpoint_dir}/critic", max_to_keep=3
        )
        
    def _build_actor(self):
        """Build policy network (actor) that selects actions"""
        model = Sequential([
            Dense(128, input_shape=(self.state_dim,), activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='tanh')
        ])
        model.compile(loss='mse', optimizer=self.actor_optimizer)
        return model

    def _build_critic(self):
        """Build value network (critic) that estimates Q-values"""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        state_h1 = Dense(128, activation='relu')(state_input)
        state_h2 = Dense(64, activation='relu')(state_h1)
        
        action_h1 = Dense(64, activation='relu')(action_input)
        
        merged = concatenate([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        merged_h2 = Dense(32, activation='relu')(merged_h1)
        output = Dense(1, activation='linear')(merged_h2)
        
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model
    
    @tf.function
    def _actor_train_step(self, states):
        """TensorFlow function for actor training step"""
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred], training=True))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return actor_loss

    @tf.function
    def _critic_train_step(self, states, actions, q_targets):
        """TensorFlow function for critic training step"""
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(q_targets - q_values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return critic_loss
    
    def act(self, state, explore=True):
        """Choose an action based on the current state"""
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random actions
            return np.random.uniform(-1, 1, self.action_dim)
        
        # Exploitation: use the actor to predict actions
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.actor(state_tensor).numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experiences from memory"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Calculate target Q-values in a single batch
        next_actions = self.actor(next_states_tensor)
        q_next = self.critic([next_states_tensor, next_actions])
        q_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * q_next[:, 0]
        
        # Train critic using optimized function
        critic_loss = self._critic_train_step(states_tensor, actions_tensor, tf.expand_dims(q_targets, axis=1))
        
        # Train actor using optimized function
        actor_loss = self._actor_train_step(states_tensor)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {"critic_loss": critic_loss.numpy(), "actor_loss": actor_loss.numpy()}
    
    def save_models(self, path=None):
        """Save actor and critic models using TensorFlow checkpoints"""
        if path:
            # Save models in Keras format
            self.actor.save(f"{path}/actor_model.h5")
            self.critic.save(f"{path}/critic_model.h5")
        
        # Save in checkpoint format
        actor_path = self.actor_manager.save()
        critic_path = self.critic_manager.save()
        
        # Save replay buffer
        self.memory.save(f"{self.checkpoint_dir}/replay_buffer.pkl")
        
        logger.info(f"Models saved to {actor_path} and {critic_path}")
        return actor_path, critic_path
    
    def load_models(self, path=None):
        if path and os.path.exists(f"{path}/actor_model.h5") and os.path.exists(f"{path}/critic_model.h5"):
            import tensorflow as tf
            # Try with metrics instead of losses
            self.actor = load_model(f"{path}/actor_model.h5", custom_objects={'mse': tf.keras.metrics.MeanSquaredError()})
            self.critic = load_model(f"{path}/critic_model.h5", custom_objects={'mse': tf.keras.metrics.MeanSquaredError()})
            logger.info(f"Models loaded from {path}")
            
            # Important: Recreate optimizers with the new model variables
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Needed for optimizer to know about the variables
            dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
            dummy_action = np.zeros((1, self.action_dim), dtype=np.float32)
            
            # Initialize the optimizer with model variables
            with tf.GradientTape() as tape:
                actions_pred = self.actor(dummy_state)
                _ = self.critic([dummy_state, dummy_action])
            
            # Try to load replay buffer
            buffer_path = f"{self.checkpoint_dir}/replay_buffer.pkl"
            if self.memory.load(buffer_path):
                logger.info(f"Replay buffer loaded from {buffer_path}")
            
            return True
        else:
            # Try loading from checkpoint
            actor_status = self.actor_checkpoint.restore(self.actor_manager.latest_checkpoint)
            critic_status = self.critic_checkpoint.restore(self.critic_manager.latest_checkpoint)
            
            if self.actor_manager.latest_checkpoint and self.critic_manager.latest_checkpoint:
                logger.info(f"Models restored from checkpoints")
                return True
            else:
                logger.warning("No models found to load")
                return False

class StockTradingEnvironment:
    def __init__(self, symbols, initial_balance, lookback_window):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.data_cache = {}
        self.reset()
        
    def reset(self):
        """Reset the environment for new episode"""
        self.balance = self.initial_balance
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.data = self._get_data()
        
        # Ensure we have enough data for lookback window
        min_data_length = min([len(self.data[symbol]) for symbol in self.symbols])
        
        # Set current step to lookback_window or to data length if not enough data
        self.current_step = min(self.lookback_window, min_data_length - 1)
        if self.current_step < self.lookback_window:
            logger.warning(f"Not enough data for full lookback window. Using {self.current_step} days instead.")
        
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        
        return self._get_state()
        
    def _get_data(self):
        """Get historical stock data for all symbols with caching using yfinance"""
        end_date = datetime.now()
        # Get more data to ensure we have enough for lookback
        start_date = end_date - timedelta(days=365 + self.lookback_window * 2)
        
        # Format dates as strings for consistent caching
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        data = {}
        for symbol in self.symbols:
            # Check if data is already cached
            cache_file = CACHE_DIR / f"{symbol}_{start_date_str}_{end_date_str}.pkl"
            
            if cache_file.exists():
                try:
                    stock_data = pd.read_pickle(cache_file)
                    logger.info(f"Loaded cached data for {symbol} from {cache_file}")
                    data[symbol] = stock_data
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached data for {symbol}: {e}")
            
            try:
                # Use yfinance to get historical data with error handling
                logger.info(f"Downloading data for {symbol}...")
                
                # Try multiple approaches to get data
                attempts = 0
                max_attempts = 3
                while attempts < max_attempts:
                    try:
                        stock_data = yf.download(
                            symbol, 
                            start=start_date_str, 
                            end=end_date_str, 
                            progress=False,
                            threads=False  # Disable multi-threading to avoid conflicts
                        )
                        
                        if len(stock_data) > 0:
                            break  # Successful download
                        
                        # Try with a different range if first attempt failed
                        stock_data = yf.download(
                            symbol, 
                            period="1y",  # Use period instead of date range
                            progress=False,
                            threads=False
                        )
                        
                        if len(stock_data) > 0:
                            break  # Successful download
                            
                    except Exception as e:
                        logger.warning(f"Attempt {attempts+1} failed for {symbol}: {e}")
                        attempts += 1
                        time.sleep(2)  # Add delay between attempts
                
                if len(stock_data) == 0:
                    logger.warning(f"No data available for {symbol}")
                    # Create dummy data as fallback
                    dummy_index = pd.date_range(start=start_date, end=end_date, freq='B')
                    stock_data = pd.DataFrame(index=dummy_index, 
                                            columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                    stock_data['Close'] = 100.0  # Default price
                    stock_data.fillna(method='ffill', inplace=True)
                
                # Cache the data
                stock_data.to_pickle(cache_file)
                logger.info(f"Downloaded and cached {len(stock_data)} days of data for {symbol}")
                
                data[symbol] = stock_data
                
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
                # Create dummy data
                dummy_index = pd.date_range(start=start_date, end=end_date, freq='B')
                stock_data = pd.DataFrame(index=dummy_index, 
                                        columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                stock_data['Close'] = 100.0  # Default price
                data[symbol] = stock_data

        return data
    
    def _get_state(self):
        """Create state representation with price, holdings, balance, and price history"""
        state = []
        
        # Current price, holdings and balance (normalized)
        for symbol in self.symbols:
            try:
                current_price = self.data[symbol]['Close'].iloc[self.current_step]
                # Extract float value from pandas Series if needed
                if hasattr(current_price, 'iloc'):
                    current_price = current_price.iloc[0]
                state.append(current_price / 1000.0)  # Normalize price
                state.append(self.shares[symbol] / 100.0)  # Normalize shares
            except IndexError as e:
                logger.error(f"Error accessing price data for {symbol} at step {self.current_step}")
                logger.error(f"Data length: {len(self.data[symbol])}")
                raise e
        
        state.append(self.balance / self.initial_balance)  # Normalized balance
        
        # Historical price data
        for symbol in self.symbols:
            try:
                # Make sure we have enough data
                if self.current_step < self.lookback_window:
                    available_data = self.data[symbol]['Close'].iloc[:self.current_step].values
                    # Pad with the earliest available price if needed
                    if len(available_data) > 0:
                        padding = np.full(self.lookback_window - len(available_data), available_data[0])
                        prices = np.concatenate([padding, available_data])
                    else:
                        # If no data at all, use a default value
                        prices = np.full(self.lookback_window, 100.0)  # Default price
                else:
                    prices = self.data[symbol]['Close'].iloc[self.current_step-self.lookback_window:self.current_step].values
                
                # Avoid division by zero
                price_max = prices.max()
                if price_max == 0:
                    normalized_prices = prices
                else:
                    normalized_prices = prices / price_max
                    
                # Flatten any nested structures and ensure they're all scalars
                for price in normalized_prices.tolist():
                    if isinstance(price, (list, tuple, np.ndarray)):
                        # If the price is still a container, take the first element
                        state.append(float(price[0]))
                    else:
                        state.append(float(price))
            except Exception as e:
                logger.error(f"Error processing price history for {symbol}: {e}")
                # Add placeholder data to maintain state vector size
                state.extend([0.0] * self.lookback_window)
        
        # Final check to ensure all state elements are scalar
        for i in range(len(state)):
            if hasattr(state[i], "__len__") and not isinstance(state[i], (str, bytes)):
                if hasattr(state[i], 'iloc'):  # Handle pandas Series
                    state[i] = float(state[i].iloc[0])
                elif len(state[i]) > 0:  # Handle list, array, etc.
                    state[i] = float(state[i][0])
                else:
                    state[i] = 0.0  # Default for empty containers
        
        return np.array(state, dtype=np.float32)  # Force float32 for TensorFlow efficiency
    
    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        portfolio_value = self.balance
        
        for symbol in self.symbols:
            current_price = self.data[symbol]['Close'].iloc[self.current_step]
            # Convert to scalar if it's a Series
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            elif isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            
            portfolio_value += self.shares[symbol] * current_price
        
        return portfolio_value
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Map continuous actions to discrete buy/sell decisions
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 3
            # Each stock has 3 action dimensions: buy, hold, sell
            buy_signal = max(0, action[action_idx])
            sell_signal = max(0, -action[action_idx + 1])
            
            current_price = self.data[symbol]['Close'].iloc[self.current_step]
            # Convert to scalar if it's a Series
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            elif isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            
            # Calculate shares to buy/sell based on the signal strength
            shares_to_buy = int(buy_signal * MAX_SHARES_PER_TRADE)
            shares_to_sell = int(sell_signal * min(MAX_SHARES_PER_TRADE, self.shares[symbol]))
            
            # Execute buy order if possible
            cost = shares_to_buy * current_price
            if cost <= self.balance and shares_to_buy > 0:
                self.balance -= cost
                self.shares[symbol] += shares_to_buy
            
            # Execute sell order
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares[symbol] -= shares_to_sell
        
        # Move to next time step
        self.current_step += 1
        
        # Check if we've reached the end of our data
        if self.current_step >= min(len(self.data[symbol]) for symbol in self.symbols) - 1:
            self.done = True
        
        # Calculate reward as portfolio value change
        old_portfolio_value = self.portfolio_value_history[-1]
        new_portfolio_value = self._calculate_portfolio_value()
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        self.portfolio_value_history.append(new_portfolio_value)
        
        return self._get_state(), reward, self.done, {"portfolio_value": new_portfolio_value}

class AlpacaTradingSystem:
    def __init__(self, agent, symbols):
        self.agent = agent
        self.symbols = symbols
        self.api = api
        
    def _get_current_state(self):
        """Create the current state based on market data and account info using yfinance instead of Alpaca API"""
        state = []
        
        # Get account information from Alpaca (still needed for balance/positions)
        try:
            account = self.api.get_account()
            balance = float(account.cash)
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            balance = INITIAL_BALANCE  # Use default if API fails
        
        # Get current positions from Alpaca
        positions = {sym: 0 for sym in self.symbols}
        try:
            current_positions = self.api.list_positions()
            for position in current_positions:
                if position.symbol in self.symbols:
                    positions[position.symbol] = int(position.qty)
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
        
        # Use yfinance to get price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_WINDOW + 5)  # Add buffer days
        
        # Format dates for yfinance
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # First add current price data
        for symbol in self.symbols:
            try:
                # Get current price using yfinance
                ticker_data = yf.Ticker(symbol)
                current_data = ticker_data.history(period="1d")
                
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                else:
                    # Try downloading with a different method
                    df = yf.download(symbol, period="1d", progress=False)
                    if not df.empty:
                        current_price = df['Close'].iloc[-1]
                    else:
                        logger.warning(f"No price data for {symbol}, using default price")
                        current_price = 100.0
                
                state.append(current_price / 1000.0)  # Normalize price
                state.append(positions[symbol] / 100.0)  # Normalize position
                
                logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error getting price data for {symbol}: {e}")
                # Use default values
                state.append(100.0 / 1000.0)  # Default normalized price
                state.append(positions[symbol] / 100.0)  # Normalize position
        
        state.append(balance / INITIAL_BALANCE)  # Normalized balance
        
        # Add historical price data
        for symbol in self.symbols:
            try:
                # Get historical data using yfinance
                hist_data = yf.download(
                    symbol, 
                    start=start_date_str,
                    end=end_date_str,
                    progress=False
                )
                
                if not hist_data.empty and len(hist_data) >= LOOKBACK_WINDOW:
                    prices = hist_data['Close'].values[-LOOKBACK_WINDOW:]
                else:
                    logger.warning(f"Not enough history for {symbol}. Using available data.")
                    if not hist_data.empty:
                        available_prices = hist_data['Close'].values
                        padding = np.full(LOOKBACK_WINDOW - len(available_prices), available_prices[0] if len(available_prices) > 0 else 100.0)
                        prices = np.concatenate([padding, available_prices])
                    else:
                        prices = np.full(LOOKBACK_WINDOW, 100.0)  # Default price
                
                # Normalize prices
                price_max = max(prices) if max(prices) > 0 else 1.0
                normalized_prices = []
                for p in prices:
                    # Ensure p is a scalar
                    if hasattr(p, 'shape') and len(p.shape) > 0:
                        p_scalar = p.item() if p.size == 1 else p[0]
                    elif isinstance(p, (list, tuple)) and len(p) > 0:
                        p_scalar = p[0]
                    else:
                        p_scalar = p
                    normalized_prices.append(float(p_scalar) / price_max)
                
                if hasattr(prices, 'shape') and len(prices.shape) > 1:
                    prices = np.ravel(prices)
                    
                # Ensure price_max is a scalar
                price_max = float(np.max(prices)) if prices.size > 0 else 1.0
                
                # Process each price safely
                normalized_prices = []
                for p in prices:
                    if hasattr(p, 'item'):
                        p = p.item()
                    normalized_prices.append(float(p) / price_max)
                    
                state.extend(normalized_prices)
                
            except Exception as e:
                logger.error(f"Error processing price history for {symbol}: {e}")
                # Add placeholder data to maintain state vector size
                state.extend([0.0] * LOOKBACK_WINDOW)
        
        return np.array(state, dtype=np.float32)
    
    def execute_trades(self):
        """Execute trades based on agent's decisions while avoiding wash trades"""
        # Get current state
        state = self._get_current_state()
        
        # Get actions from agent (no exploration in live trading)
        actions = self.agent.act(state, explore=False)
        
        # Get current positions to avoid wash trades
        positions = {}
        try:
            current_positions = self.api.list_positions()
            for position in current_positions:
                positions[position.symbol] = {
                    'qty': int(position.qty),
                    'last_trade': datetime.now() - timedelta(hours=24)  # Default to 24h ago
                }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            positions = {sym: {'qty': 0, 'last_trade': datetime.now() - timedelta(hours=24)} for sym in self.symbols}
        
        # Create trading log for this cycle
        trade_actions = []
        
        # Execute actions
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 3
            buy_signal = max(0, actions[action_idx])
            sell_signal = max(0, -actions[action_idx + 1])
            
            # Avoid wash trades: only do buy or sell in a single cycle, not both
            if buy_signal > 0.2 and sell_signal > 0.2:
                if buy_signal > sell_signal:
                    sell_signal = 0
                    logger.info(f"Prevented wash trade for {symbol} - choosing buy over sell")
                else:
                    buy_signal = 0
                    logger.info(f"Prevented wash trade for {symbol} - choosing sell over buy")
            
            # Get current position
            try:
                position = positions.get(symbol, {'qty': 0})
                current_shares = position.get('qty', 0)
            except:
                current_shares = 0
            
            # Calculate shares to buy/sell
            shares_to_buy = int(buy_signal * MAX_SHARES_PER_TRADE)
            shares_to_sell = int(sell_signal * min(MAX_SHARES_PER_TRADE, current_shares))
            
            # Execute buy order
            if shares_to_buy > 0:
                try:
                    # Get latest price with yfinance before placing order
                    ticker_data = yf.Ticker(symbol)
                    latest_data = ticker_data.history(period="1d")
                    if not latest_data.empty:
                        current_price = latest_data['Close'].iloc[-1]
                        logger.info(f"Latest price for {symbol}: ${current_price:.2f}")
                    
                    self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_buy,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"BUY ORDER: {shares_to_buy} shares of {symbol}")
                    trade_actions.append(f"BUY {symbol}: {shares_to_buy} shares")
                    
                    # Update position tracking
                    if symbol not in positions:
                        positions[symbol] = {'qty': shares_to_buy, 'last_trade': datetime.now()}
                    else:
                        positions[symbol]['qty'] = positions[symbol].get('qty', 0) + shares_to_buy
                        positions[symbol]['last_trade'] = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Buy order failed for {symbol}: {e}")
            
            # Wait a few seconds between buy and sell to avoid wash trade detection
            if shares_to_buy > 0 and shares_to_sell > 0:
                time.sleep(5)
            
            # Execute sell order
            if shares_to_sell > 0:
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"SELL ORDER: {shares_to_sell} shares of {symbol}")
                    trade_actions.append(f"SELL {symbol}: {shares_to_sell} shares")
                    
                    # Update position tracking
                    if symbol in positions:
                        positions[symbol]['qty'] = max(0, positions[symbol].get('qty', 0) - shares_to_sell)
                        positions[symbol]['last_trade'] = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Sell order failed for {symbol}: {e}")
        
        if trade_actions:
            logger.info(f"Actions taken: {', '.join(trade_actions)}")
        else:
            logger.info("No significant trading actions taken")
        
        # Get updated portfolio value
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            logger.info(f"Current Portfolio Value: ${equity:.2f}")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
        
        return actions

def train_agent():
    """Train the DRL agent using historical data with GPU acceleration"""
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
    
    best_reward = -np.inf
    checkpoint_interval = 50  # Save every 50 episodes
    
    # Create a progress bar for training
    progress_bar = tqdm(range(TRAINING_EPISODES), desc="Training Progress")
    
    for episode in progress_bar:
        state = env.reset()
        total_reward = 0
        done = False
        episode_losses = []
        
        # Use mini-batch updates for faster training
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            losses = agent.train()
            
            if losses:
                episode_losses.append(losses)
            
            state = next_state
            total_reward += reward
        
        # Calculate mean losses for this episode
        if episode_losses:
            mean_actor_loss = np.mean([loss['actor_loss'] for loss in episode_losses if 'actor_loss' in loss])
            mean_critic_loss = np.mean([loss['critic_loss'] for loss in episode_losses if 'critic_loss' in loss])
            progress_bar.set_postfix({
                'reward': total_reward, 
                'value': info['portfolio_value'],
                'actor_loss': mean_actor_loss,
                'critic_loss': mean_critic_loss
            })
        
        # Save checkpoint periodically
        if episode % checkpoint_interval == 0:
            agent.save_models()
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_models("./models")
    
    # Test the trained agent
    test_agent(agent)
    
    return agent

def test_agent(agent=None):
    """Test the trained agent"""
    if agent is None:
        agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
        agent.load_models("./models")
    
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
    
    total_rewards = []
    portfolio_values = []
    
    for episode in tqdm(range(TEST_EPISODES), desc="Testing Progress"):
        state = env.reset()
        total_reward = 0
        done = False
        episode_portfolio = [INITIAL_BALANCE]
        
        while not done:
            action = agent.act(state, explore=False)  # No exploration during testing
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            episode_portfolio.append(info['portfolio_value'])
        
        total_rewards.append(total_reward)
        portfolio_values.append(episode_portfolio)
        
        if episode % 10 == 0:  # Print only every 10 episodes to reduce output
            logger.info(f"Test Episode {episode+1}/{TEST_EPISODES}, Final Reward: {total_reward:.4f}, Final Portfolio Value: {info['portfolio_value']:.2f}")
    
# Calculate statistics
    final_values = [portfolio[-1] for portfolio in portfolio_values]
    mean_final_value = np.mean(final_values)
    std_final_value = np.std(final_values)
    
    logger.info(f"Testing completed. Mean final portfolio value: ${mean_final_value:.2f} ± ${std_final_value:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot average portfolio value
    avg_portfolio = np.mean([np.array(p[:min(len(p) for p in portfolio_values)]) for p in portfolio_values], axis=0)
    plt.plot(avg_portfolio, 'b-', linewidth=2, label='Average Portfolio Value')
    
    # Plot confidence interval
    if len(portfolio_values) > 1:
        std_portfolio = np.std([np.array(p[:min(len(p) for p in portfolio_values)]) for p in portfolio_values], axis=0)
        plt.fill_between(range(len(avg_portfolio)), 
                         avg_portfolio - std_portfolio, 
                         avg_portfolio + std_portfolio, 
                         color='blue', alpha=0.2, label='Standard Deviation')
    
    # Plot some individual episodes
    num_episodes_to_plot = min(5, len(portfolio_values))
    for i in range(num_episodes_to_plot):
        plt.plot(portfolio_values[i], 'r-', alpha=0.3)
    
    plt.title("Portfolio Value During Testing Episodes")
    plt.xlabel("Trading Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_results.png", dpi=300)
    plt.close()
    
    # Plot the distribution of final values
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=20, alpha=0.7, color='green')
    plt.axvline(mean_final_value, color='r', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_final_value:.2f}')
    plt.title("Distribution of Final Portfolio Values")
    plt.xlabel("Final Portfolio Value ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_values_distribution.png", dpi=300)
    plt.close()
    
    return agent

def invalidate_cache(symbols=None, days=1):
    """
    Invalidates cache files for the specified symbols or all symbols
    that are older than the specified number of days
    
    Args:
        symbols: List of symbols to invalidate cache for, or None for all
        days: Number of days to consider for cache invalidation
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
    
def run_live_trading():
    """Run the agent in live trading mode with Alpaca"""
    # Load trained agent
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
    if not agent.load_models("./models"):
        logger.error("Failed to load model weights. Please train the model first.")
        return
    
    # Create trading system
    trading_system = AlpacaTradingSystem(agent, STOCK_SYMBOLS)
    
    # Invalidate cache at the start to ensure fresh data
    invalidate_cache(days=1)
    
    # Variables for tracking time since last cache invalidation
    last_cache_invalidation = datetime.now()
    cache_invalidation_interval = timedelta(hours=4)  # Invalidate every 4 hours
    
    # Main trading loop
    try:
        max_retries = 5
        retry_delay = 30  # seconds
        
        while True:
            # Check if it's time to invalidate cache
            now = datetime.now()
            if now - last_cache_invalidation > cache_invalidation_interval:
                invalidate_cache(days=1)
                last_cache_invalidation = now
                logger.info("Performed regular cache invalidation")
            
            # Get market status with retry logic
            for attempt in range(max_retries):
                try:
                    clock = api.get_clock()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"API connection failed (attempt {attempt+1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect to API after {max_retries} attempts: {e}")
                        # Wait longer before trying again to avoid hammering the API
                        time.sleep(300)  # 5 minutes
                        retry_delay = 30  # Reset delay
                        continue
            
            try:
                if clock.is_open:
                    logger.info("\n--- Executing Trades ---")
                    actions = trading_system.execute_trades()
                    
                    # Log the actions in a more readable format
                    action_summary = []
                    for i, symbol in enumerate(STOCK_SYMBOLS):
                        action_idx = i * 3
                        buy_signal = max(0, actions[action_idx])
                        sell_signal = max(0, -actions[action_idx + 1])
                        if buy_signal > 0.1:
                            action_summary.append(f"BUY {symbol}: {buy_signal:.2f}")
                        elif sell_signal > 0.1:
                            action_summary.append(f"SELL {symbol}: {sell_signal:.2f}")
                    
                    if action_summary:
                        logger.info(f"Actions taken: {', '.join(action_summary)}")
                    else:
                        logger.info("No significant trading actions taken")
                    
                    # Get current portfolio value with retry logic
                    for attempt in range(max_retries):
                        try:
                            account = api.get_account()
                            equity = float(account.equity)
                            logger.info(f"Current Portfolio Value: ${equity:.2f}")
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Failed to get account info (attempt {attempt+1}/{max_retries}): {e}")
                                time.sleep(retry_delay)
                            else:
                                logger.error(f"Failed to get account info after {max_retries} attempts")
                    
                    # Sleep until next trading decision
                    if TRADING_FREQUENCY == "daily":
                        # Check time to market close
                        time_to_close = (clock.next_close - clock.timestamp).total_seconds()
                        if time_to_close < 60*60:  # If less than 1 hour to close
                            logger.info(f"Market closing soon. Sleeping until next market open.")
                            time.sleep(time_to_close + 60*60)  # Sleep until next day
                        else:
                            logger.info("Waiting for next trading window...")
                            time.sleep(60*60)  # Check every hour
                    elif TRADING_FREQUENCY == "hourly":
                        logger.info("Waiting for next hourly window...")
                        time.sleep(60*60)  # Sleep for 1 hour
                    else:  # "minute"
                        logger.info("Waiting for next minute window...")
                        time.sleep(60)  # Sleep for 1 minute
                else:
                    logger.info("Market is closed. Waiting for market to open...")
                    time.sleep(60*60)  # Check every hour if market opened
            except Exception as e:
                logger.error(f"Error during trading operations: {e}")
                # Add a delay to prevent rapid retries if there's a persistent error
                time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Trading session terminated by user")

def run_parallel_training(num_processes=None):
    """
    Run multiple parallel training sessions and combine the results
    This can help utilize multi-core CPUs better when GPU is not available
    """
    if num_processes is None:
        import multiprocessing
        num_processes = max(2, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    from multiprocessing import Pool
    import random
    
    def train_with_seed(seed):
        """Train with a specific random seed"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        # Use a different output path for each process
        model_path = f"./models/model_{seed}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
        agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
        
        best_reward = -np.inf
        
        # Train for fewer episodes in each process
        episodes_per_process = TRAINING_EPISODES // num_processes
        
        for episode in range(episodes_per_process):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
                total_reward += reward
            
            if episode % 10 == 0:
                logger.info(f"Process {seed}: Episode {episode+1}/{episodes_per_process}, Reward: {total_reward:.4f}, Value: {info['portfolio_value']:.2f}")
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_models(model_path)
        
        # Test the trained agent
        test_rewards = []
        for _ in range(50):  # Run a small number of test episodes
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state, explore=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        avg_reward = np.mean(test_rewards)
        logger.info(f"Process {seed} completed. Average test reward: {avg_reward:.4f}")
        
        return {
            'seed': seed,
            'avg_reward': avg_reward,
            'model_path': model_path,
            'best_reward': best_reward
        }
    
    # Generate random seeds for different processes
    seeds = [random.randint(1, 10000) for _ in range(num_processes)]
    
    # Run training in parallel
    logger.info(f"Starting parallel training with {num_processes} processes...")
    with Pool(num_processes) as pool:
        results = pool.map(train_with_seed, seeds)
    
    # Find the best model based on test performance
    best_result = max(results, key=lambda x: x['avg_reward'])
    logger.info(f"Best model found from process with seed {best_result['seed']}")
    logger.info(f"Best model average test reward: {best_result['avg_reward']:.4f}")
    
    # Copy the best model to the main models directory
    best_model_path = best_result['model_path']
    import shutil
    for file in Path(best_model_path).glob('*'):
        shutil.copy(file, "./models/")
    
    logger.info(f"Best model copied to main models directory")
    
    # Load and return the best agent
    best_agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
    best_agent.load_models("./models")
    
    return best_agent

def hyperparameter_optimization(trials=10):
    """
    Run hyperparameter optimization to find better model parameters
    Uses a simple random search approach
    """
    import random
    from joblib import Parallel, delayed
    
    # Define hyperparameter search space
    param_space = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'gamma': [0.9, 0.95, 0.98, 0.99],
        'batch_size': [32, 64, 128, 256],
        'epsilon_decay': [0.99, 0.995, 0.998, 0.999]
    }
    
    def evaluate_hyperparams(trial_idx, lr, gamma, batch_size, epsilon_decay):
        """Train and evaluate with specific hyperparameters"""
        global BATCH_SIZE, GAMMA
        
        # Store original values
        orig_batch_size = BATCH_SIZE
        orig_gamma = GAMMA
        
        # Set new values for this trial
        BATCH_SIZE = batch_size
        GAMMA = gamma
        
        # Set random seed for reproducibility
        random_seed = trial_idx * 1000
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        random.seed(random_seed)
        
        # Create model path for this trial
        model_path = f"./models/trial_{trial_idx}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize environment and agent
        env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
        agent = DRLAgent(STATE_DIM, ACTION_DIM, learning_rate=lr)
        agent.epsilon_decay = epsilon_decay
        
        # Train for fewer episodes to speed up optimization
        train_episodes = 200
        
        # Train the agent
        best_reward = -np.inf
        for episode in range(train_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
                total_reward += reward
            
            if episode % 50 == 0:
                logger.info(f"Trial {trial_idx}, Episode {episode+1}/{train_episodes}, Reward: {total_reward:.4f}")
            
            # Save best model for this trial
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_models(model_path)
        
        # Evaluate the trained agent
        test_episodes = 50
        test_rewards = []
        
        for _ in range(test_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state, explore=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        std_test_reward = np.std(test_rewards)
        
        # Restore original values
        BATCH_SIZE = orig_batch_size
        GAMMA = orig_gamma
        
        logger.info(f"Trial {trial_idx} completed. Hyperparameters: lr={lr}, gamma={gamma}, "
                   f"batch_size={batch_size}, epsilon_decay={epsilon_decay}")
        logger.info(f"Average test reward: {avg_test_reward:.4f} ± {std_test_reward:.4f}")
        
        return {
            'trial_idx': trial_idx,
            'learning_rate': lr,
            'gamma': gamma,
            'batch_size': batch_size,
            'epsilon_decay': epsilon_decay,
            'avg_test_reward': avg_test_reward,
            'std_test_reward': std_test_reward,
            'model_path': model_path
        }
    
    # Generate random hyperparameter combinations
    param_combinations = []
    for i in range(trials):
        params = {
            'trial_idx': i,
            'lr': random.choice(param_space['learning_rate']),
            'gamma': random.choice(param_space['gamma']),
            'batch_size': random.choice(param_space['batch_size']),
            'epsilon_decay': random.choice(param_space['epsilon_decay'])
        }
        param_combinations.append(params)
    
    # Run trials in parallel (adjust n_jobs based on available cores)
    logger.info(f"Starting hyperparameter optimization with {trials} trials...")
    results = Parallel(n_jobs=3)(
        delayed(evaluate_hyperparams)(
            p['trial_idx'], p['lr'], p['gamma'], p['batch_size'], p['epsilon_decay']
        ) for p in param_combinations
    )
    
    # Find best hyperparameters
    best_trial = max(results, key=lambda x: x['avg_test_reward'])
    
    logger.info("\n--- Hyperparameter Optimization Results ---")
    logger.info(f"Best parameters: learning_rate={best_trial['learning_rate']}, "
               f"gamma={best_trial['gamma']}, batch_size={best_trial['batch_size']}, "
               f"epsilon_decay={best_trial['epsilon_decay']}")
    logger.info(f"Best average test reward: {best_trial['avg_test_reward']:.4f} ± {best_trial['std_test_reward']:.4f}")
    
    # Copy the best model to the main models directory
    best_model_path = best_trial['model_path']
    import shutil
    for file in Path(best_model_path).glob('*'):
        shutil.copy(file, "./models/")
    
    logger.info(f"Best model copied to main models directory")
    
    return best_trial

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Trading AI with Alpaca API - Optimized Version')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'live', 'parallel', 'optimize'], default='train',
                        help='Operation mode: train, test, live trading, parallel training, or hyperparameter optimization')
    parser.add_argument('--api_key', type=str, help='Alpaca API Key')
    parser.add_argument('--api_secret', type=str, help='Alpaca API Secret Key')
    parser.add_argument('--processes', type=int, default=None, help='Number of parallel processes for training')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--epochs', type=int, default=TRAINING_EPISODES, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage (will raise error if not available)')
    
    args = parser.parse_args()
    
    # Set API keys if provided
    if args.api_key and args.api_secret:
        ALPACA_API_KEY = args.api_key
        ALPACA_SECRET_KEY = args.api_secret
    
    # Check GPU availability if requested
    if args.gpu:
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError("GPU was explicitly requested but none is available. Please check your installation.")
        else:
            logger.info("GPU usage explicitly requested and GPU is available.")
    
    # Create models directory if it doesn't exist
    Path("./models").mkdir(exist_ok=True)
    
    # Run in the specified mode
    try:
        if args.mode == 'train':
            logger.info("Training agent...")
            logger.info(f"Number of epochs: {args.epochs}")
            TRAINING_EPISODES = args.epochs
            agent = train_agent()
        elif args.mode == 'test':
            logger.info("Testing agent...")
            test_agent()
        elif args.mode == 'live':
            logger.info("Starting live trading...")
            run_live_trading()
        elif args.mode == 'parallel':
            logger.info(f"Starting parallel training with {args.processes} processes...")
            run_parallel_training(args.processes)
        elif args.mode == 'optimize':
            logger.info(f"Starting hyperparameter optimization with {args.trials} trials...")
            hyperparameter_optimization(args.trials)
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())