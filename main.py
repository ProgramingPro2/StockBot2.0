import os
import pandas as pd
import numpy as np
import time
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    concatenate,
    BatchNormalization,
)
import matplotlib.pyplot as plt
import yfinance as yf
import dotenv
from pathlib import Path
import logging
from tqdm import tqdm
import json
import optuna
from concurrent.futures import ThreadPoolExecutor


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="app.log",
)
logger = logging.getLogger(__name__)

# Configure GPU memory growth to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth and optimize for RTX 3090
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Enable mixed precision training for faster performance
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

        logger.info(
            f"GPU is available. Found {len(gpus)} GPU(s). Mixed precision policy set to {policy.name}"
        )
    except RuntimeError as e:
        logger.warning(f"Memory growth setting failed: {e}")
else:
    logger.info("No GPU available, using CPU for training")

# Enable eager execution
tf.config.run_functions_eagerly(False)

tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
tf.config.threading.set_inter_op_parallelism_threads(8)  # Number of P-cores
tf.config.threading.set_intra_op_parallelism_threads(8)  # Number of P-cores

# Load environment variables
dotenv.load_dotenv()

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Connect to Alpaca API
api = tradeapi.REST(
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2"
)

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
INITIAL_BALANCE = 100000.0
LOOKBACK_WINDOW = 10
TRADING_FREQUENCY = "minute"
MAX_SHARES_PER_TRADE = 100
TRAINING_EPISODES = 1000
TEST_EPISODES = 1000
REWARD_SCALING_FACTOR = 1000.0

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
            REWARD_SCALING_FACTOR = optimized_params.get("reward_scaling_factor", REWARD_SCALING_FACTOR)
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

# RL model parameters
STATE_DIM = (len(STOCK_SYMBOLS) * 2) + 1 + (len(STOCK_SYMBOLS) * LOOKBACK_WINDOW)
ACTION_DIM = len(STOCK_SYMBOLS) * 3
LEARNING_RATE = 0.001
GAMMA = 0.95
BATCH_SIZE = 4096  # For max GPU utilization on RTX 3090 (with mixed precision)
MEMORY_SIZE = 100000  # Increased for better training stability
NOISE_SIGMA = 0.15
NOISE_THETA = 0.2


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate memory for better performance
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # Direct assignment to pre-allocated arrays
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Generate random indices - much faster than random.choice
        indices = np.random.randint(0, self.size, batch_size)

        # Return slices of the arrays for better performance
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size

    def save(self, path):
        """Save replay buffer to disk"""
        buffer_data = {
            "states": self.states[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "next_states": self.next_states[: self.size],
            "dones": self.dones[: self.size],
            "position": self.position,
            "size": self.size,
        }
        np.savez_compressed(path, **buffer_data)

    def load(self, path):
        """Load replay buffer from disk"""
        if not os.path.exists(path):
            logger.info(f"Replay buffer file not found: {path}")
            return False

        try:
            data = np.load(path, allow_pickle=True)

            # Check if all expected keys are present
            expected_keys = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
                "position",
                "size",
            ]
            if not all(key in data for key in expected_keys):
                logger.error(
                    f"Corrupted replay buffer file: {path}. Missing expected keys."
                )
                self.position = 0
                self.size = 0
                return False

            size = min(int(data["size"]), self.capacity)
            if size > 0:
                # Ensure loaded arrays have compatible shapes before assignment
                if (
                    data["states"].shape[1] != self.states.shape[1]
                    or data["actions"].shape[1] != self.actions.shape[1]
                ):
                    logger.error(
                        f"Shape mismatch when loading replay buffer from {path}. Buffer dimensions might have changed."
                    )
                    self.position = 0
                    self.size = 0
                    return False

                self.states[:size] = data["states"][:size]
                self.actions[:size] = data["actions"][:size]
                self.rewards[:size] = data["rewards"][:size]
                self.next_states[:size] = data["next_states"][:size]
                self.dones[:size] = data["dones"][:size]

                self.position = int(data["position"]) % self.capacity
                self.size = size
                logger.info(f"Replay buffer loaded {self.size} transitions from {path}")
                return True
            else:
                logger.warning(f"Replay buffer file {path} was empty or had size 0.")
                self.position = 0
                self.size = 0
                return False
        except Exception as e:
            logger.error(f"Error loading replay buffer from {path}: {e}")
            self.position = 0
            self.size = 0
            return False


class DRLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=LEARNING_RATE, noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Use the optimized replay buffer
        self.memory = ReplayBuffer(MEMORY_SIZE, state_dim, action_dim)

        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=noise_sigma, theta=noise_theta)

        # Set default TensorFlow dtype to float32 explicitly
        tf.keras.backend.set_floatx("float32")

        # Create separate optimizers for actor and critic
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Build models
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Create target networks for more stable learning
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()

        # Initialize target networks with same weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # TensorFlow checkpoint for saving models
        self.checkpoint_dir = "./checkpoints"
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        self.actor_checkpoint = tf.train.Checkpoint(
            model=self.actor, optimizer=self.actor_optimizer
        )
        self.critic_checkpoint = tf.train.Checkpoint(
            model=self.critic, optimizer=self.critic_optimizer
        )
        self.actor_manager = tf.train.CheckpointManager(
            self.actor_checkpoint,
            directory=f"{self.checkpoint_dir}/actor",
            max_to_keep=3,
        )
        self.critic_manager = tf.train.CheckpointManager(
            self.critic_checkpoint,
            directory=f"{self.checkpoint_dir}/critic",
            max_to_keep=3,
        )

    def _build_actor(self):
        """Build policy network (actor) that selects actions with GPU optimizations"""
        model = Sequential(
            [
                # Use CuDNN-optimized layers when running on GPU
                Dense(
                    512,
                    input_shape=(self.state_dim,),
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                ),
                BatchNormalization(dtype="float32"),
                Dense(
                    256,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                ),
                BatchNormalization(dtype="float32"),
                Dense(
                    128,
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                ),
                Dense(
                    self.action_dim,
                    activation="tanh",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                ),
            ]
        )
        model.compile(loss="mse", optimizer=self.actor_optimizer)
        return model

    def _build_critic(self):
        """Build value network (critic) that estimates Q-values with GPU optimizations"""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))

        # State branch with batch normalization
        state_h1 = Dense(
            512, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )(state_input)
        state_bn1 = BatchNormalization(dtype="float32")(state_h1)
        state_h2 = Dense(
            256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )(state_bn1)
        state_bn2 = BatchNormalization(dtype="float32")(state_h2)

        # Action branch
        action_h1 = Dense(
            256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )(action_input)
        action_bn1 = BatchNormalization(dtype="float32")(action_h1)

        # Merge
        merged = concatenate([state_bn2, action_bn1])
        merged_h1 = Dense(
            256, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )(merged)
        merged_bn1 = BatchNormalization(dtype="float32")(merged_h1)
        merged_h2 = Dense(
            128, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )(merged_bn1)
        output = Dense(
            1,
            activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )(merged_h2)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=self.critic_optimizer)
        return model

    @tf.function
    def _actor_train_step(self, states):
        """TensorFlow function for actor training step"""
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(
                self.critic([states, actions_pred], training=True)
            )

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Apply gradient clipping
        actor_grads = [(tf.clip_by_norm(grad, 1.0)) for grad in actor_grads]
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )
        return actor_loss

    @tf.function
    def _critic_train_step(self, states, actions, q_targets):
        """TensorFlow function for critic training step"""
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            # Ensure q_values is float32 for calculation with q_targets
            q_values_float32 = tf.cast(q_values, tf.float32)
            critic_loss = tf.reduce_mean(tf.square(q_targets - q_values_float32))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Apply gradient clipping
        critic_grads = [(tf.clip_by_norm(grad, 1.0)) for grad in critic_grads]
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )
        return critic_loss

    def act(self, state, explore=True):
        """Choose an action based on the current state"""
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random actions
            return np.random.uniform(-1, 1, self.action_dim)

        # Exploitation: use the actor to predict actions
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action_output = self.actor(state_tensor).numpy()[0]

        # Add exploration noise
        if explore:
            action_output += self.noise()

        return action_output

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """Train the agent using experiences from memory with GPU optimizations"""
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a larger batch for better GPU utilization
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to tensors with proper GPU-friendly types, explicitly specifying float32
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Calculate target Q-values using target models for stability
        # This calculation can be done in a single batched operation for speed
        next_actions = self.target_actor(next_states_tensor, training=False)
        q_next = self.target_critic([next_states_tensor, next_actions], training=False)
        # Ensure q_next is float32 for calculation with gamma and rewards_tensor
        q_next_float32 = tf.cast(q_next, tf.float32)
        q_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * q_next_float32[:, 0]

        # Run critic and actor training with optimized GPU functions
        critic_loss = self._critic_train_step(
            states_tensor, actions_tensor, tf.expand_dims(q_targets, axis=1)
        )
        actor_loss = self._actor_train_step(states_tensor)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Perform soft target updates
        self._update_target_networks()

        return {"critic_loss": critic_loss.numpy(), "actor_loss": actor_loss.numpy()}

    def _update_target_networks(self, tau=0.005):
        """Soft update target networks"""
        for target_param, param in zip(
            self.target_actor.variables, self.actor.variables
        ):
            target_param.assign(param * tau + target_param * (1 - tau))

        for target_param, param in zip(
            self.target_critic.variables, self.critic.variables
        ):
            target_param.assign(param * tau + target_param * (1 - tau))

    def save_models(self, path=None):
        """Save actor and critic models using TensorFlow checkpoints and .keras format"""
        # Use .keras format for path-based saving (recommended)
        if path:
            Path(path).mkdir(exist_ok=True)  # Ensure directory exists
            try:
                self.actor.save(f"{path}/actor_model.keras")
                self.critic.save(f"{path}/critic_model.keras")
                logger.info(f"Models saved in .keras format to {path}")
            except Exception as e:
                logger.error(f"Failed to save models in .keras format: {e}")

        # Save in checkpoint format (always do this)
        try:
            actor_path = self.actor_manager.save()
            critic_path = self.critic_manager.save()
            logger.info(
                f"Checkpoint saved: Actor to {actor_path}, Critic to {critic_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoints: {e}")
            actor_path, critic_path = None, None  # Indicate failure

        # Save replay buffer using the correct extension (.npz)
        buffer_path = f"{self.checkpoint_dir}/replay_buffer.npz"
        try:
            self.memory.save(buffer_path)
            logger.info(f"Replay buffer saved to {buffer_path}")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")

        return actor_path, critic_path

    def load_models(self, path=None):
        """Load models, prioritizing checkpoints, then .keras files."""
        checkpoint_loaded = False
        # --- Attempt 1: Load from Checkpoints ---
        if (
            self.actor_manager.latest_checkpoint
            and self.critic_manager.latest_checkpoint
        ):
            try:
                logger.info(
                    f"Attempting to restore models from checkpoints: {self.actor_manager.latest_checkpoint}"
                )
                # Restore model weights using the checkpoint objects
                actor_status = self.actor_checkpoint.restore(
                    self.actor_manager.latest_checkpoint
                )
                critic_status = self.critic_checkpoint.restore(
                    self.critic_manager.latest_checkpoint
                )

                # Verify restoration
                actor_status.assert_existing_objects_matched()
                critic_status.assert_existing_objects_matched()

                logger.info("Models successfully restored from checkpoints.")

                # CRITICAL: Re-initialize optimizers *after* restoring model weights
                # The optimizer state is also restored by the checkpoint if it was saved.
                # If optimizers were part of the checkpoint (they are), restoration handles them.
                # We just need to make sure the agent's optimizer attributes point correctly.
                self.actor_optimizer = self.actor_checkpoint.optimizer
                self.critic_optimizer = self.critic_checkpoint.optimizer

                checkpoint_loaded = True

            except Exception as e:
                logger.warning(
                    f"Checkpoint restoration failed: {e}. Trying other methods..."
                )
                # Fall through to try path-based loading

        # Load replay buffer - always try this after attempting checkpoint load
        buffer_path = f"{self.checkpoint_dir}/replay_buffer.npz"
        if self.memory.load(buffer_path):
            logger.info(f"Replay buffer loaded from {buffer_path}")
        else:
            logger.warning(
                f"Replay buffer not found or failed to load from {buffer_path}"
            )

        if checkpoint_loaded:
            return True  # Successfully loaded from checkpoint

        # --- Attempt 2: Load from .keras files (if path provided) ---
        if path:
            keras_actor_path = f"{path}/actor_model.keras"
            keras_critic_path = f"{path}/critic_model.keras"

            if os.path.exists(keras_actor_path) and os.path.exists(keras_critic_path):
                logger.info(f"Attempting to load models from .keras files in {path}")
                try:
                    # No custom_objects usually needed for .keras format with standard layers/ops
                    self.actor = load_model(
                        keras_actor_path, compile=False
                    )  # Re-compile after loading
                    self.critic = load_model(keras_critic_path, compile=False)

                    # Re-compile the models with their respective optimizers
                    self.actor.compile(
                        optimizer=self.actor_optimizer, loss="mse"
                    )  # Loss might not matter if only using for inference/gradients
                    self.critic.compile(optimizer=self.critic_optimizer, loss="mse")

                    logger.info(
                        f"Models successfully loaded from .keras files in {path}"
                    )
                    return True
                except Exception as e:
                    logger.error(
                        f"Failed to load models from .keras files in {path}: {e}. Trying H5 fallback..."
                    )
            else:
                logger.info(
                    f".keras models not found in {path}. Checking for H5 fallback..."
                )

            # --- Attempt 3: Load from H5 files (Fallback - Not Recommended) ---
            h5_actor_path = f"{path}/actor_model.h5"
            h5_critic_path = f"{path}/critic_model.h5"
            if os.path.exists(h5_actor_path) and os.path.exists(h5_critic_path):
                logger.warning(
                    f"Attempting to load models from legacy .h5 files in {path}. This is not recommended with mixed precision."
                )
                try:
                    # H5 loading might still fail with 'Cast' error even with scope
                    with tf.keras.utils.custom_object_scope(
                        {"mse": tf.keras.metrics.MeanSquaredError()}
                    ):
                        self.actor = load_model(h5_actor_path, compile=False)
                        self.critic = load_model(h5_critic_path, compile=False)

                    # Re-compile the models
                    self.actor.compile(optimizer=self.actor_optimizer, loss="mse")
                    self.critic.compile(optimizer=self.critic_optimizer, loss="mse")

                    logger.info(
                        f"Models successfully loaded from legacy .h5 files in {path}"
                    )
                    return True
                except ValueError as ve:
                    if "Unknown layer: 'Cast'" in str(ve):
                        logger.error(
                            "Loading from .h5 failed due to 'Cast' layer issue, as expected with mixed precision. Cannot load these files."
                        )
                    else:
                        logger.error(f"Error loading models from .h5 files: {ve}")
                except Exception as e:
                    logger.error(f"Unexpected error loading models from .h5 files: {e}")
            else:
                logger.info(f"Legacy .h5 models not found in {path}.")

        # --- Failed to load ---
        logger.error("Failed to load models from checkpoints or specified path.")
        return False


class StockTradingEnvironment:
    # Class-level cache for stock data across all instances
    _global_data_cache = {}

    def __init__(self, symbols, initial_balance, lookback_window):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        # Instance-level cache reference
        self.data_cache = {}
        # Precompute fixed values
        self.shares_normalization = 100.0
        self.price_normalization = 1000.0
        # Preallocate state array size for reuse
        self.state_size = (
            (len(self.symbols) * 2) + 1 + (len(self.symbols) * self.lookback_window)
        )
        self.reset()

    def reset(self):
        """Reset the environment for new episode"""
        self.balance = self.initial_balance
        # Use numpy arrays for shares for vectorized operations
        self.shares = {symbol: 0 for symbol in self.symbols}
        self.data = self._get_data()

        # Ensure we have enough data for lookback window - use numpy for efficiency
        data_lengths = np.array([len(self.data[symbol]) for symbol in self.symbols])
        min_data_length = np.min(data_lengths)

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
        """Cache current prices for all symbols to avoid repeated lookups"""
        self.current_prices = {}
        for symbol in self.symbols:
            try:
                current_price = self.data[symbol]["Close"].iloc[self.current_step]

                # Convert to scalar if necessary
                if isinstance(current_price, pd.Series):
                    current_price = (
                        current_price.iloc[0] if len(current_price) > 0 else 0.0
                    )
                elif (
                    hasattr(current_price, "item")
                    and hasattr(current_price, "size")
                    and current_price.size == 1
                ):
                    current_price = current_price.item()
                elif hasattr(current_price, "shape") and len(current_price.shape) > 0:
                    current_price = current_price[0] if current_price.size > 0 else 0.0

                self.current_prices[symbol] = float(current_price)
            except Exception as e:
                logger.error(f"Error updating current price for {symbol}: {e}")
                self.current_prices[symbol] = 100.0  # Default price

    def _get_state(self):
        """Create state representation with vectorized operations for better performance"""
        # Reuse pre-allocated array for better performance
        state = np.zeros(self.state_size, dtype=np.float32)

        # Current price and holdings with vectorized operations
        for i, symbol in enumerate(self.symbols):
            try:
                # Use cached price
                current_price = self.current_prices[symbol]

                # Directly assign to pre-allocated array (faster)
                state[i * 2] = (
                    current_price / self.price_normalization
                )  # Normalize price
                state[i * 2 + 1] = (
                    self.shares[symbol] / self.shares_normalization
                )  # Normalize shares

            except Exception as e:
                logger.error(
                    f"Error accessing price data for {symbol} at step {self.current_step}: {e}"
                )
                # Use default values
                state[i * 2] = 0.0
                state[i * 2 + 1] = 0.0

        # Add normalized balance
        state[len(self.symbols) * 2] = self.balance / self.initial_balance

        # Add historical price data in a vectorized way
        lookback_start_idx = (len(self.symbols) * 2) + 1
        for i, symbol in enumerate(self.symbols):
            try:
                # Get historical prices for the lookback window
                # Ensure we have enough data for the lookback window
                if self.current_step < self.lookback_window:
                    # If not enough data, pad with the first available price
                    available_prices = self.data[symbol]['Close'].iloc[:self.current_step + 1].values
                    if available_prices.ndim > 1:
                        available_prices = available_prices.flatten()
                    
                    if len(available_prices) > 0:
                        # Pad with the first available price
                        first_price = available_prices[0]
                        prices = np.full(self.lookback_window, first_price, dtype=np.float32)
                        prices[self.lookback_window - len(available_prices):] = available_prices
                    else:
                        # If no available data, fill with default price
                        prices = np.full(self.lookback_window, 100.0, dtype=np.float32)
                else:
                    prices = self.data[symbol]['Close'].iloc[self.current_step - self.lookback_window : self.current_step].values
                    if prices.ndim > 1:
                        prices = prices.flatten()
                
                # Normalize prices
                price_max = np.max(prices) if prices.size > 0 else 1.0
                normalized_prices = prices / max(price_max, 1e-8)
                
                # Assign to state array
                state_start = lookback_start_idx + (i * self.lookback_window)
                state_end = state_start + self.lookback_window
                state[state_start:state_end] = normalized_prices

            except Exception as e:
                logger.error(f"Error processing price history for {symbol}: {e}")
                # Add zeroes to maintain state vector size
                state_start = lookback_start_idx + (i * self.lookback_window)
                state_end = state_start + self.lookback_window
                state[state_start:state_end] = 0.0

        return state

    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        # Use the cached current prices for faster calculation
        portfolio_value = self.balance

        # Vectorized calculation using numpy for efficiency
        for symbol in self.symbols:
            portfolio_value += self.shares[symbol] * self.current_prices[symbol]

        return portfolio_value

    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Map continuous actions to discrete buy/sell decisions
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 3
            # Each stock has 3 action dimensions: buy, hold, sell
            buy_signal = max(0, action[action_idx])
            sell_signal = max(0, -action[action_idx + 1])

            # Use cached current price
            current_price = self.current_prices[symbol]

            # Calculate shares to buy/sell based on the signal strength
            shares_to_buy = int(buy_signal * MAX_SHARES_PER_TRADE)
            if buy_signal > 0 and shares_to_buy == 0: # Ensure at least 1 share if signal is positive
                shares_to_buy = 1
            shares_to_sell = int(
                sell_signal * min(MAX_SHARES_PER_TRADE, self.shares[symbol])
            )
            if sell_signal > 0 and shares_to_sell == 0: # Ensure at least 1 share if signal is positive
                shares_to_sell = 1

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
        min_data_length = min(len(self.data[symbol]) for symbol in self.symbols)
        if self.current_step >= min_data_length - 1:
            self.done = True

        # Update current prices for the new time step
        self._update_current_prices()

        # Calculate reward as portfolio value change
        old_portfolio_value = self.portfolio_value_history[-1]
        new_portfolio_value = self._calculate_portfolio_value()

        # Avoid division by zero and use faster calculation
        if old_portfolio_value > 0:
            reward = ((new_portfolio_value - old_portfolio_value) / old_portfolio_value) * REWARD_SCALING_FACTOR
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
        """Get historical stock data for all symbols with caching using yfinance"""
        # Check if we already have all the data loaded in class-level memory
        if all(symbol in self._global_data_cache for symbol in self.symbols):
            logger.info("Using in-memory cached data for all symbols")
            return {symbol: self._global_data_cache[symbol] for symbol in self.symbols}

        end_date = datetime.now()
        # Get more data to ensure we have enough for lookback
        start_date = end_date - timedelta(days=365 + self.lookback_window * 2)

        # Format dates as strings for consistent caching
        end_date_str = end_date.strftime("%Y-%m-%d")
        start_date_str = start_date.strftime("%Y-%m-%d")

        data = {}
        
        def download_symbol_data(symbol):
            cache_file = CACHE_DIR / f"{symbol}_{start_date_str}_{end_date_str}.pkl"
            
            if cache_file.exists():
                try:
                    stock_data = pd.read_pickle(cache_file)
                    logger.info(f"Loaded cached data for {symbol} from {cache_file}")
                    return symbol, stock_data
                except Exception as e:
                    logger.warning(f"Failed to load cached data for {symbol}: {e}. Attempting re-download.")
            
            try:
                logger.info(f"Downloading data for {symbol}...")
                attempts = 0
                max_attempts = 5 # Increased attempts for robustness
                while attempts < max_attempts:
                    try:
                        stock_data = yf.download(
                            symbol,
                            start=start_date_str,
                            end=end_date_str,
                            progress=False,
                            threads=False,  # Disable multi-threading to avoid conflicts
                        )
                        
                        # Check for data completeness (e.g., at least 80% of expected days)
                        expected_days = (end_date - start_date).days
                        if len(stock_data) > 0.8 * expected_days:
                            break  # Successful and reasonably complete download
                        else:
                            logger.warning(f"Incomplete data for {symbol} (only {len(stock_data)}/{expected_days} days). Retrying.")

                    except Exception as e:
                        logger.warning(
                            f"Attempt {attempts + 1} failed for {symbol}: {e}"
                        )
                    attempts += 1
                    time.sleep(2)  # Add delay between attempts
                
                if len(stock_data) == 0 or len(stock_data) < 0.8 * expected_days:
                    logger.warning(f"No sufficient data available for {symbol} after {max_attempts} attempts. Creating dummy data.")
                    # Create dummy data as fallback
                    dummy_index = pd.date_range(
                        start=start_date, end=end_date, freq="B"
                    )
                    stock_data = pd.DataFrame(
                        index=dummy_index,
                        columns=["Open", "High", "Low", "Close", "Volume"],
                    )
                    stock_data["Close"] = 100.0  # Default price
                    stock_data.fillna(method="ffill", inplace=True)

                # Cache the data
                stock_data.to_pickle(cache_file)
                logger.info(
                    f"Downloaded and cached {len(stock_data)} days of data for {symbol}"
                )
                return symbol, stock_data

            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
                # Create dummy data
                dummy_index = pd.date_range(start=start_date, end=end_date, freq="B")
                stock_data = pd.DataFrame(
                    index=dummy_index,
                    columns=["Open", "High", "Low", "Close", "Volume"],
                )
                stock_data["Close"] = 100.0  # Default price
                return symbol, stock_data

        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(download_symbol_data, self.symbols)
            for symbol, stock_data in results:
                data[symbol] = stock_data
                self._global_data_cache[symbol] = stock_data

        return data

    def prefetch_data(self):
        """Prefetch data to memory for faster training (CPU-optimized version)"""
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

        return prefetched_data


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
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # First add current price data
        for symbol in self.symbols:
            try:
                # Get current price using yfinance
                ticker_data = yf.Ticker(symbol)
                current_data = ticker_data.history(period="1d")

                if not current_data.empty:
                    current_price = current_data["Close"].iloc[-1]
                else:
                    # Try downloading with a different method
                    df = yf.download(symbol, period="1d", progress=False)
                    if not df.empty:
                        current_price = df["Close"].iloc[-1]
                    else:
                        logger.warning(
                            f"No real-time price data for {symbol} from yfinance, using default price. Consider Alpaca streaming API for better accuracy."
                        )
                        current_price = 100.0

                # Ensure current_price is a scalar
                if isinstance(current_price, pd.Series):
                    current_price = (
                        current_price.iloc[0] if not current_price.empty else 100.0
                    )
                elif (
                    hasattr(current_price, "item")
                    and hasattr(current_price, "size")
                    and current_price.size == 1
                ):
                    current_price = current_price.item()
                elif not isinstance(current_price, (int, float)):
                    current_price = float(current_price)

                state.append(float(current_price) / 1000.0)  # Normalize price
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
                    symbol, start=start_date_str, end=end_date_str, progress=False
                )

                if not hist_data.empty and len(hist_data) >= LOOKBACK_WINDOW:
                    prices = hist_data["Close"].values[-LOOKBACK_WINDOW:]

                    # Ensure prices is a 1D array
                    if prices.ndim > 1:
                        prices = prices.flatten()
                else:
                    logger.warning(
                        f"Not enough history for {symbol}. Using available data."
                    )
                    if not hist_data.empty:
                        available_prices = hist_data["Close"].values
                        # Ensure available_prices is a 1D array
                        if (
                            hasattr(available_prices, "shape")
                            and len(available_prices.shape) > 1
                        ):
                            available_prices = available_prices.flatten()

                        first_value = (
                            available_prices[0] if len(available_prices) > 0 else 100.0
                        )
                        # Convert first_value to scalar if necessary
                        if hasattr(first_value, "item"):
                            first_value = first_value.item()

                        padding = np.full(
                            LOOKBACK_WINDOW - len(available_prices), first_value
                        )
                        prices = np.concatenate([padding, available_prices])
                    else:
                        prices = np.full(LOOKBACK_WINDOW, 100.0)  # Default price

                # Ensure prices is 1D and contains only scalars
                prices_1d = np.zeros(len(prices), dtype=np.float32)
                for j, p in enumerate(prices):
                    if hasattr(p, "item") and hasattr(p, "size") and p.size == 1:
                        prices_1d[j] = p.item()
                    elif isinstance(p, pd.Series):
                        prices_1d[j] = p.iloc[0] if len(p) > 0 else 0.0
                    elif hasattr(p, "shape") and len(p.shape) > 0:
                        prices_1d[j] = p[0] if p.size > 0 else 0.0
                    else:
                        prices_1d[j] = float(p)

                # Normalize prices
                price_max = np.max(prices_1d) if prices_1d.size > 0 else 1.0
                normalized_prices = prices_1d / max(
                    price_max, 1e-8
                )  # Avoid div by zero

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
                    "qty": int(position.qty),
                    "last_trade": datetime.now()
                    - timedelta(hours=24),  # Default to 24h ago
                }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            positions = {
                sym: {"qty": 0, "last_trade": datetime.now() - timedelta(hours=24)}
                for sym in self.symbols
            }

        # Create trading log for this cycle
        trade_actions = []

        # Execute actions
        for i, symbol in enumerate(self.symbols):
            action_idx = i * 3
            buy_signal = max(0, actions[action_idx])
            sell_signal = max(0, -actions[action_idx + 1])

            # Avoid wash trades: only do buy or sell in a single cycle, not both
            # If both signals are strong, prioritize the stronger one
            if (
                buy_signal > 0.1 and sell_signal > 0.1
            ):  # Threshold to consider a signal "strong"
                if buy_signal >= sell_signal:
                    sell_signal = 0
                    logger.info(
                        f"Prevented wash trade for {symbol} - prioritizing buy over sell"
                    )
                else:
                    buy_signal = 0
                    logger.info(
                        f"Prevented wash trade for {symbol} - prioritizing sell over buy"
                    )

            # Get current position
            try:
                position = positions.get(symbol, {"qty": 0})
                current_shares = position.get("qty", 0)
            except Exception as e:
                logger.error(f"Error getting current shares for {symbol}: {e}")
                current_shares = 0

            # Calculate shares to buy/sell
            shares_to_buy = int(buy_signal * MAX_SHARES_PER_TRADE)
            shares_to_sell = int(
                sell_signal * min(MAX_SHARES_PER_TRADE, current_shares)
            )

            # Execute buy order
            if shares_to_buy > 0:
                try:
                    # Get latest price with yfinance before placing order
                    ticker_data = yf.Ticker(symbol)
                    latest_data = ticker_data.history(period="1d")
                    if not latest_data.empty:
                        current_price = latest_data["Close"].iloc[-1]
                        logger.info(f"Latest price for {symbol}: ${current_price:.2f}")

                    # Retry logic for submit_order
                    order_placed = False
                    for order_attempt in range(3): # Max 3 attempts
                        try:
                            self.api.submit_order(
                                symbol=symbol,
                                qty=shares_to_buy,
                                side="buy",
                                type="market",
                                time_in_force="day",
                            )
                            order_placed = True
                            break
                        except Exception as order_e:
                            logger.warning(f"Order submission failed for {symbol} (attempt {order_attempt + 1}/3): {order_e}")
                            time.sleep(5) # Wait before retrying
                    if not order_placed:
                        logger.error(f"Failed to place buy order for {symbol} after multiple attempts.")
                        continue # Skip to next symbol if order fails
                    logger.info(f"BUY ORDER: {shares_to_buy} shares of {symbol}")
                    trade_actions.append(f"BUY {symbol}: {shares_to_buy} shares")

                    # Update position tracking
                    if symbol not in positions:
                        positions[symbol] = {
                            "qty": shares_to_buy,
                            "last_trade": datetime.now(),
                        }
                    else:
                        positions[symbol]["qty"] = (
                            positions[symbol].get("qty", 0) + shares_to_buy
                        )
                        positions[symbol]["last_trade"] = datetime.now()

                except Exception as e:
                    logger.error(
                        f"Buy order failed for {symbol}: {e}, for shares: {shares_to_buy} at total price: {current_price * shares_to_buy}"
                    )

            # Execute sell order
            if shares_to_sell > 0:
                # Retry logic for submit_order
                order_placed = False
                for order_attempt in range(3): # Max 3 attempts
                    try:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=shares_to_sell,
                            side="sell",
                            type="market",
                            time_in_force="day",
                        )
                        order_placed = True
                        break
                    except Exception as order_e:
                        logger.warning(f"Order submission failed for {symbol} (attempt {order_attempt + 1}/3): {order_e}")
                        time.sleep(5) # Wait before retrying
                if not order_placed:
                    logger.error(f"Failed to place sell order for {symbol} after multiple attempts.")
                    continue # Skip to next symbol if order fails
                try:
                    logger.info(f"SELL ORDER: {shares_to_sell} shares of {symbol}")
                    trade_actions.append(f"SELL {symbol}: {shares_to_sell} shares")

                    # Update position tracking
                    if symbol in positions:
                        positions[symbol]["qty"] = max(
                            0, positions[symbol].get("qty", 0) - shares_to_sell
                        )
                        positions[symbol]["last_trade"] = datetime.now()

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


def optimize_agent():
    """Perform hyperparameter optimization using Optuna."""
    logger.info("Starting hyperparameter optimization...")

    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical(
            "batch_size", [256, 512, 1024, 2048, 4096]
        )
        gamma = trial.suggest_uniform("gamma", 0.9, 0.99)
        noise_sigma = trial.suggest_uniform("noise_sigma", 0.05, 0.3)
        noise_theta = trial.suggest_uniform("noise_theta", 0.1, 0.5)
        reward_scaling_factor = trial.suggest_loguniform("reward_scaling_factor", 10.0, 10000.0)

        # Update global parameters or pass them to the agent
        global LEARNING_RATE, BATCH_SIZE, GAMMA
        LEARNING_RATE = learning_rate
        BATCH_SIZE = batch_size
        GAMMA = gamma

        logger.info(
            f"Trial {trial.number}: LR={learning_rate:.6f}, BS={batch_size}, Gamma={gamma:.3f}"
        )

        # Train the agent with suggested hyperparameters
        env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
        agent = DRLAgent(STATE_DIM, ACTION_DIM, learning_rate, noise_sigma=noise_sigma, noise_theta=noise_theta)
        agent.gamma = gamma  # Ensure agent uses the trial's gamma

        # Train for a reduced number of episodes for optimization speed
        optimization_episodes = 200  # Reduced episodes for faster trials

        for episode in tqdm(range(optimization_episodes), desc=f"Trial {trial.number} Training"): # Add tqdm here
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if len(agent.memory) >= BATCH_SIZE:
                for _ in range(2):  # Train less frequently during optimization
                    agent.train()

        # Test the agent and return the mean final portfolio value
        test_env = StockTradingEnvironment(
            STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW
        )
        final_values = []
        for _ in tqdm(range(50), desc=f"Trial {trial.number} Testing"): # Add tqdm here
            state = test_env.reset()
            done = False
            while not done:
                action = agent.act(state, explore=False)
                next_state, _, done, info = test_env.step(action)
                state = next_state
            final_values.append(info["portfolio_value"])

        mean_final_value = np.mean(final_values)
        logger.info(
            f"Trial {trial.number} finished with mean portfolio value: ${mean_final_value:.2f}"
        )
        return mean_final_value

    study = optuna.create_study(direction="maximize")
    # Create a progress bar for optimization trials
    for _ in tqdm(range(args.trials), desc="Optimization Progress (Trials)"):
        study.optimize(objective, n_trials=1, show_progress_bar=False) # Run one trial at a time to update tqdm

    logger.info("Optimization finished.")
    logger.info(f"""Best trial:
  Value: {study.best_value:.2f}
  Params: {study.best_params}""")

    # Update global parameters with the best found
    global LEARNING_RATE, BATCH_SIZE, GAMMA, REWARD_SCALING_FACTOR
    LEARNING_RATE = study.best_params["learning_rate"]
    BATCH_SIZE = study.best_params["batch_size"]
    GAMMA = study.best_params["gamma"]
    REWARD_SCALING_FACTOR = study.best_params["reward_scaling_factor"]

    logger.info("Global hyperparameters updated to best found values.")

    # Save best hyperparameters to a file
    hyperparams_path = Path("./hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best hyperparameters saved to {hyperparams_path}")


def train_agent():
    """Train the DRL agent using historical data with optimized GPU acceleration"""
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW)
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE, noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA)

    best_reward = -np.inf
    checkpoint_interval = 50  # Save every 50 episodes

    # Pre-fetch commonly used data to GPU (if applicable, though not directly used in this training loop)
    # prefetched_data = env.prefetch_data() # Commented out as it's not directly used in the training loop

    # Create a progress bar for training
    progress_bar = tqdm(range(TRAINING_EPISODES), desc="Training Progress")

    for episode in progress_bar:
        state = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        # Collect experiences for this episode
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # Train on collected experiences in larger batches
        if len(agent.memory) >= BATCH_SIZE:
            # Multiple training iterations per episode for better learning
            for _ in range(4):  # Train 4x per episode
                losses = agent.train()
                if losses:
                    episode_losses.append(losses)

        # Calculate mean losses for this episode
        if episode_losses:
            mean_actor_loss = np.mean(
                [loss["actor_loss"] for loss in episode_losses if "actor_loss" in loss]
            )
            mean_critic_loss = np.mean(
                [
                    loss["critic_loss"]
                    for loss in episode_losses
                    if "critic_loss" in loss
                ]
            )
            progress_bar.set_postfix(
                {
                    "reward": f"{total_reward:.6f}", # Display more decimal places
                    "value": info["portfolio_value"],
                    "actor_loss": mean_actor_loss,
                    "critic_loss": mean_critic_loss,
                }
            )

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
        agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE, noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA)
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
            episode_portfolio.append(info["portfolio_value"])

        total_rewards.append(total_reward)
        portfolio_values.append(episode_portfolio)

        if episode % 10 == 0:  # Print only every 10 episodes to reduce output
            logger.info(
                f"Test Episode {episode + 1}/{TEST_EPISODES}, Final Reward: {total_reward:.6f}, Final Portfolio Value: {info['portfolio_value']:.2f}"
            )

    # Calculate statistics
    final_values = [portfolio[-1] for portfolio in portfolio_values]
    mean_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    std_final_value = np.std(final_values)

    logger.info(
        f"Testing completed. Mean final portfolio value: ${mean_final_value:.2f} ± ${std_final_value:.2f}"
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot results
    plt.figure(figsize=(14, 8))  # Slightly larger figure for better readability

    # Determine the minimum length of all portfolio value histories
    min_len = min(len(p) for p in portfolio_values)

    # Align all portfolio histories to the minimum length for accurate averaging
    aligned_portfolios = [np.array(p[:min_len]) for p in portfolio_values]

    # Plot average portfolio value
    avg_portfolio = np.mean(aligned_portfolios, axis=0)
    plt.plot(avg_portfolio, "b-", linewidth=2, label="Average Portfolio Value")

    # Plot confidence interval (if enough data)
    if len(aligned_portfolios) > 1:
        std_portfolio = np.std(aligned_portfolios, axis=0)
        plt.fill_between(
            range(len(avg_portfolio)),
            avg_portfolio - std_portfolio,
            avg_portfolio + std_portfolio,
            color="blue",
            alpha=0.2,
            label="Standard Deviation",
        )

    # Plot some individual episodes for visual variety
    num_episodes_to_plot = min(
        10, len(portfolio_values)
    )  # Plot more individual episodes
    for i in np.random.choice(
        len(portfolio_values), num_episodes_to_plot, replace=False
    ):
        plt.plot(portfolio_values[i], "r-", alpha=0.15)  # Make individual lines lighter

    plt.title("Portfolio Value During Testing Episodes", fontsize=16)
    plt.xlabel("Trading Steps", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(
        "test_results.png", dpi=600, bbox_inches="tight"
    )  # Save with tight bounding box
    plt.close()

    # Plot the distribution of final values
    plt.figure(figsize=(10, 7))  # Slightly larger figure
    plt.hist(
        final_values, bins=30, alpha=0.8, color="skyblue", edgecolor="black"
    )  # More bins, better color
    plt.axvline(
        mean_final_value,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: ${mean_final_value:.2f}",
    )
    plt.axvline(
        median_final_value,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: ${median_final_value:.2f}",
    )
    plt.title("Distribution of Final Portfolio Values", fontsize=16)
    plt.xlabel("Final Portfolio Value ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("final_values_distribution.png", dpi=600, bbox_inches="tight")
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
    logger.info("Starting live trading...")
    # Load trained agent
    agent = DRLAgent(STATE_DIM, ACTION_DIM, LEARNING_RATE, noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA)
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
                        logger.warning(
                            f"API connection failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to connect to API after {max_retries} attempts: {e}"
                        )
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

                    # Log all weights
                    logger.info("\n--- Model Weights ---")
                    for i, symbol in enumerate(STOCK_SYMBOLS):
                        action_idx = i * 3
                        buy_weight = actions[action_idx]
                        sell_weight = actions[action_idx + 1]
                        logger.info(
                            f"{symbol}: Buy Weight: {buy_weight:.2f}, Sell Weight: {sell_weight:.2f}"
                        )

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
                                logger.warning(
                                    f"Failed to get account info (attempt {attempt + 1}/{max_retries}): {e}"
                                )
                                time.sleep(retry_delay)
                            else:
                                logger.error(
                                    f"Failed to get account info after {max_retries} attempts"
                                )

                    # Sleep until next trading decision
                    if TRADING_FREQUENCY == "daily":
                        # Check time to market close
                        time_to_close = (
                            clock.next_close - clock.timestamp
                        ).total_seconds()
                        if time_to_close < 60 * 60:  # If less than 1 hour to close
                            logger.info(
                                "Market closing soon. Sleeping until next market open."
                            )
                            time.sleep(time_to_close + 60 * 60)  # Sleep until next day
                        else:
                            logger.info("Waiting for next trading window...")
                            time.sleep(60 * 60)  # Check every hour
                    elif TRADING_FREQUENCY == "hourly":
                        logger.info("Waiting for next hourly window...")
                        time.sleep(60 * 60)  # Sleep for 1 hour
                    else:  # "minute"
                        logger.info("Waiting for next minute window...")
                        time.sleep(60)  # Sleep for 1 minute
                else:
                    logger.info("Market is closed. Waiting for market to open...")
                    time.sleep(60)  # Check every hour if market opened
            except Exception as e:
                logger.error(f"Error during trading operations: {e}")
                # Add a delay to prevent rapid retries if there's a persistent error
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Trading session terminated by user")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock Trading AI with Alpaca API - Optimized Version"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "live", "parallel", "optimize"],
        default="train",
        help="Operation mode: train, test, live trading, parallel training, or hyperparameter optimization",
    )
    parser.add_argument("--api_key", type=str, help="Alpaca API Key")
    parser.add_argument("--api_secret", type=str, help="Alpaca API Secret Key")
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_EPISODES,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage (will raise error if not available)",
    )

    args = parser.parse_args()

    # Set API keys if provided
    if args.api_key and args.api_secret:
        ALPACA_API_KEY = args.api_key
        ALPACA_SECRET_KEY = args.api_secret

    # Check GPU availability if requested
    if args.gpu:
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError(
                "GPU was explicitly requested but none is available. Please check your installation."
            )
        else:
            logger.info("GPU usage explicitly requested and GPU is available.")

    # Create models directory if it doesn't exist
    Path("./models").mkdir(exist_ok=True)

    # Run in the specified mode
    try:
        if args.mode == "train":
            logger.info("Training agent...")
            logger.info(f"Number of epochs: {args.epochs}")
            TRAINING_EPISODES = args.epochs
            agent = train_agent()
        elif args.mode == "test":
            logger.info("Testing agent...")
            test_agent()
        elif args.mode == "live":
            logger.info("Starting live trading...")
            run_live_trading()
        elif args.mode == "optimize":
            logger.info("Starting hyperparameter optimization...")
            optimize_agent()
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())
