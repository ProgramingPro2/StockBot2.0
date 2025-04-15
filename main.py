# Stock Trading AI with Alpaca API and Deep RL
# This script integrates Alpaca API with a Deep Reinforcement Learning model
# based on https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading

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
dotenv.load_dotenv()

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Configuration parameters
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "TSLA", "NVDA", "INTC", "AMD"]  # Stocks to trade
INITIAL_BALANCE = 100000.0  # Initial portfolio value
LOOKBACK_WINDOW = 10  # Number of past days to consider for state
TRADING_FREQUENCY = "daily"  # or "hourly", "minute" 
MAX_SHARES_PER_TRADE = 100  # Maximum number of shares per trade
TRAINING_EPISODES = 100  # Number of episodes for training
TEST_EPISODES = 10  # Number of episodes for testing

# RL model parameters
STATE_DIM = len(STOCK_SYMBOLS) * (3 + LOOKBACK_WINDOW)  # Current price, shares, cash + price history
ACTION_DIM = len(STOCK_SYMBOLS) * 3  # Buy, hold, sell for each stock
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
BATCH_SIZE = 32
MEMORY_SIZE = 10000

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
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_actor(self):
        """Build policy network (actor) that selects actions"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='tanh'))  # Action values between -1 and 1
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def _build_critic(self):
        """Build value network (critic) that estimates Q-values"""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(32, activation='relu')(state_h1)
        
        action_h1 = Dense(32, activation='relu')(action_input)
        
        merged = concatenate([state_h2, action_h1])
        merged_h1 = Dense(32, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def act(self, state, explore=True):
        """Choose an action based on the current state"""
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random actions
            return np.random.uniform(-1, 1, self.action_dim)
        
        # Exploitation: use the actor to predict actions
        return self.actor.predict(state.reshape(1, -1))[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experiences from memory"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Train critic
        next_actions = self.actor.predict(next_states)
        q_values = self.critic.predict([next_states, next_actions])
        targets = rewards + self.gamma * q_values * (1 - dones)
        self.critic.fit([states, actions], targets, epochs=1, verbose=0)
        
        # Train actor
        a_for_grad = self.actor.predict(states)
        critic_grads = np.zeros([BATCH_SIZE, self.action_dim])
        
        for i in range(BATCH_SIZE):
            with tf.GradientTape() as tape:
                tape.watch(a_for_grad[i:i+1])
                q_values = self.critic([states[i:i+1], a_for_grad[i:i+1]])
            critic_grads[i] = tape.gradient(q_values, a_for_grad[i:i+1])[0]
        
        # Apply actor gradients
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_models(self, path):
        """Save actor and critic models"""
        self.actor.save(f"{path}/actor_model.h5")
        self.critic.save(f"{path}/critic_model.h5")
    
    def load_models(self, path):
        """Load actor and critic models"""
        self.actor = load_model(f"{path}/actor_model.h5")
        self.critic = load_model(f"{path}/critic_model.h5")

class StockTradingEnvironment:
    def __init__(self, symbols, initial_balance, lookback_window):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.reset()
        
    def reset(self):
        """Reset the environment for new episode"""
        self.balance = self.initial_balance
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.current_step = self.lookback_window
        self.data = self._download_data()
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        
        return self._get_state()
    
    def _download_data(self):
        """Download historical stock data for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data
        
        data = {}
        for symbol in self.symbols:
            # Use yfinance to get historical data
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            data[symbol] = stock_data
        
        return data
    
    def _get_state(self):
        """Create state representation with price, holdings, balance, and price history"""
        state = []
        
        # Current price, holdings and balance (normalized)
        for symbol in self.symbols:
            current_price = self.data[symbol]['Close'].iloc[self.current_step]
            state.append(current_price / 1000.0)  # Normalize price
            state.append(self.shares[symbol] / 100.0)  # Normalize shares
        
        state.append(self.balance / self.initial_balance)  # Normalized balance
        
        # Historical price data
        for symbol in self.symbols:
            prices = self.data[symbol]['Close'].iloc[self.current_step-self.lookback_window:self.current_step].values
            # Normalize prices
            normalized_prices = prices / prices.max()
            state.extend(normalized_prices.tolist())
        
        return np.array(state)
    
    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        portfolio_value = self.balance
        
        for symbol in self.symbols:
            current_price = self.data[symbol]['Close'].iloc[self.current_step]
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
            # hold_signal = action[action_idx + 2]  # Unused - for future extensions
            
            current_price = self.data[symbol]['Close'].iloc[self.current_step]
            
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
        if self.current_step >= len(self.data[self.symbols[0]]) - 1:
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
        """Create the current state based on market data and account info"""
        state = []
        
        # Get account information
        account = self.api.get_account()
        balance = float(account.cash)
        
        # Get current positions
        positions = {sym: 0 for sym in self.symbols}
        current_positions = self.api.list_positions()
        for position in current_positions:
            if position.symbol in self.symbols:
                positions[position.symbol] = int(position.qty)
        
        # Get current prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_WINDOW + 5)  # Add buffer days
        
        # Add current price and position data
        for symbol in self.symbols:
            # Get current market data
            barset = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date.isoformat(), end=end_date.isoformat())
            current_price = barset[-1].c
            
            state.append(current_price / 1000.0)  # Normalize price
            state.append(positions[symbol] / 100.0)  # Normalize position
        
        state.append(balance / INITIAL_BALANCE)  # Normalized balance
        
        # Add price history
        for symbol in self.symbols:
            bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date.isoformat(), end=end_date.isoformat())
            prices = [bar.c for bar in bars[-LOOKBACK_WINDOW-1:-1]]
            normalized_prices = np.array(prices) / max(prices)
            state.extend(normalized_prices.tolist())
        
        return np.array(state)
    
    def execute_trades(self):
        """Execute trades based on agent's decisions"""
        # Get current state
        state = self._get_current_state()
        
        # Get actions from agent (no exploration in live trading)
        actions = self.agent.act(state, explore=False)
        
        # Execute actions
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 3
            buy_signal = max(0, actions[action_idx])
            sell_signal = max(0, -actions[action_idx + 1])
            
            # Get current position
            try:
                position = self.api.get_position(symbol)
                current_shares = int(position.qty)
            except:
                current_shares = 0
            
            # Calculate shares to buy/sell
            shares_to_buy = int(buy_signal * MAX_SHARES_PER_TRADE)
            shares_to_sell = int(sell_signal * min(MAX_SHARES_PER_TRADE, current_shares))
            
            # Execute buy order
            if shares_to_buy > 0:
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_buy,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"BUY ORDER: {shares_to_buy} shares of {symbol}")
                except Exception as e:
                    print(f"Buy order failed for {symbol}: {e}")
            
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
                    print(f"SELL ORDER: {shares_to_sell} shares of {symbol}")
                except Exception as e:
                    print(f"Sell order failed for {symbol}: {e}")
        
        return actions

def train_agent():
    """Train the DRL agent using historical data"""
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
    
    best_reward = -np.inf
    
    for episode in range(TRAINING_EPISODES):
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
        
        print(f"Episode: {episode+1}/{TRAINING_EPISODES}, Total Reward: {total_reward}, Final Portfolio Value: {info['portfolio_value']}")
        
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
    
    for episode in range(TEST_EPISODES):
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
        
        print(f"Test Episode {episode+1}/{TEST_EPISODES}, Final Reward: {total_reward}, Final Portfolio Value: {info['portfolio_value']}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for i, portfolio in enumerate(portfolio_values):
        plt.plot(portfolio, label=f"Episode {i+1}")
    
    plt.title("Portfolio Value During Testing Episodes")
    plt.xlabel("Trading Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig("test_results.png")
    plt.show()
    
    return agent

def run_live_trading():
    """Run the agent in live trading mode with Alpaca"""
    # Load trained agent
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE)
    agent.load_models("./models")
    
    # Create trading system
    trading_system = AlpacaTradingSystem(agent, STOCK_SYMBOLS)
    
    # Check market status
    clock = api.get_clock()
    
    print("Starting live trading system...")
    print(f"Current Time: {clock.timestamp}")
    print(f"Market Open: {clock.is_open}")
    
    if not clock.is_open:
        next_open = api.get_calendar(start=datetime.now().date().isoformat())[0].open
        print(f"Market is closed. Next opening at {next_open}")
    
    # Main trading loop
    while True:
        clock = api.get_clock()
        if clock.is_open:
            print("\n--- Executing Trades ---")
            actions = trading_system.execute_trades()
            print(f"Actions taken: {actions}")
            
            # Get current portfolio value
            account = api.get_account()
            equity = float(account.equity)
            print(f"Current Portfolio Value: ${equity}")
            
            # Sleep until next trading decision
            if TRADING_FREQUENCY == "daily":
                # Check time to market close
                time_to_close = (clock.next_close - clock.timestamp).total_seconds()
                if time_to_close < 60*60:  # If less than 1 hour to close
                    print(f"Market closing soon. Sleeping until next market open.")
                    time.sleep(time_to_close + 60*60)  # Sleep until next day
                else:
                    print("Waiting for next trading window...")
                    time.sleep(60*60)  # Check every hour
            elif TRADING_FREQUENCY == "hourly":
                print("Waiting for next hourly window...")
                time.sleep(60*60)  # Sleep for 1 hour
            else:  # "minute"
                print("Waiting for next minute window...")
                time.sleep(60)  # Sleep for 1 minute
        else:
            print("Market is closed. Waiting for market to open...")
            time.sleep(60*60)  # Check every hour if market opened

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Trading AI with Alpaca API')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'live'], default='train',
                        help='Operation mode: train, test, or live trading')
    parser.add_argument('--api_key', type=str, help='Alpaca API Key')
    parser.add_argument('--api_secret', type=str, help='Alpaca API Secret Key')
    
    args = parser.parse_args()
    
    # Set API keys if provided
    if args.api_key and args.api_secret:
        ALPACA_API_KEY = args.api_key
        ALPACA_SECRET_KEY = args.api_secret
    
    # Create models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Run in the specified mode
    if args.mode == 'train':
        print("Training agent...")
        agent = train_agent()
    elif args.mode == 'test':
        print("Testing agent...")
        test_agent()
    elif args.mode == 'live':
        print("Starting live trading...")
        run_live_trading()