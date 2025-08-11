import numpy as np
import time
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import matplotlib
matplotlib.use('Agg')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import dotenv
from pathlib import Path
import logging
from tqdm import tqdm
import json
import optuna
from collections import deque

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADING, STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, CACHE_DIR, MAX_SHARES_PER_TRADE, REWARD_SCALING_FACTOR, STATE_DIM, ACTION_DIM, LEARNING_RATE, BATCH_SIZE, GAMMA, MEMORY_SIZE, NOISE_SIGMA, NOISE_THETA, TRAINING_EPISODES, TEST_EPISODES, TRADING_FREQUENCY
from src.stockbot.utils import invalidate_cache, configure_ssl_settings
from src.stockbot.system_utils import comprehensive_system_check, ConfigurationValidator
from src.stockbot.drl_agent import DRLAgent
from src.stockbot.stock_env import StockTradingEnvironment
from src.stockbot.alpaca_trade_system import AlpacaTradingSystem
from src.stockbot.performance_monitor import PerformanceMonitor

if Path("app.log").exists():
    Path("app.log").unlink()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

# Configure SSL settings early to prevent SSL errors
configure_ssl_settings()

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
})


policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Configure GPU memory growth instead of hard limits for better compatibility
        # This allows TensorFlow to allocate GPU memory as needed
        for gpu in gpus:
            try:
                # Try to get GPU memory info
                gpu_details = tf.config.experimental.get_device_details(gpu)
                memory_limit = gpu_details.get('memory_limit', 8192)  # Default to 8GB
                # Use 75% of available GPU memory
                memory_limit = int(memory_limit * 0.75)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"GPU memory limit set to {memory_limit}MB")
            except Exception as e:
                logger.warning(f"Could not set GPU memory limit: {e}. Using memory growth instead.")
                # Fallback: memory growth is already enabled above
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        logger.info(f"GPU is available. Found {len(gpus)} GPU(s).")
        
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logger.info("Default device set to GPU")
    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")
else:
    logger.info("No GPU found, using CPU.")

tf.config.run_functions_eagerly(False)
dotenv.load_dotenv()

def _verify_gpu_availability():
    """Check and log GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU detected: {len(gpus)} physical GPU(s)")
    else:
        logger.warning("No GPU detected. Training will be slower on CPU")

def _create_required_directories():
    """Create necessary directories if they don't exist."""
    required_directories = ["./models", "./data_cache", "./checkpoints"]
    for directory_path in required_directories:
        Path(directory_path).mkdir(exist_ok=True)

def _validate_configuration():
    """Validate that required configuration is present."""
    if not STOCK_SYMBOLS:
        logger.error("No stock symbols configured")
        return False
    return True

def check_system_health():
    """Perform comprehensive system health checks."""
    logger.info("Starting system health checks...")
    
    # Use the new comprehensive system check
    system_ok = comprehensive_system_check()
    
    # Validate hyperparameters
    hyperparams = {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'reward_scaling_factor': REWARD_SCALING_FACTOR
    }
    
    warnings = ConfigurationValidator.validate_hyperparameters(hyperparams)
    for warning in warnings:
        logger.warning(f"Hyperparameter warning: {warning}")
    
    # Validate state/action dimensions
    calculated_state_dim, calculated_action_dim = ConfigurationValidator.validate_state_action_dimensions(
        STOCK_SYMBOLS, LOOKBACK_WINDOW
    )
    
    if calculated_state_dim != STATE_DIM:
        logger.error(f"State dimension mismatch! Calculated: {calculated_state_dim}, Config: {STATE_DIM}")
        return False
    
    if calculated_action_dim != ACTION_DIM:
        logger.error(f"Action dimension mismatch! Calculated: {calculated_action_dim}, Config: {ACTION_DIM}")
        return False
    
    if system_ok:
        logger.info(f"✅ Health checks passed. Trading {len(STOCK_SYMBOLS)} symbols")
    else:
        logger.warning("⚠️ Some health checks failed but continuing...")
    
    return True


trading_client = None
stock_data_client = None





def _initialize_training_environment(debug=False):
    """Initialize and return training environment and agent."""
    invalidate_cache(days=0)
    StockTradingEnvironment.clear_in_memory_cache()
    
    environment = StockTradingEnvironment(
        STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, 
        stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, 
        REWARD_SCALING_FACTOR, debug=debug
    )
    
    agent = DRLAgent(
        STATE_DIM, ACTION_DIM, LEARNING_RATE, GAMMA, MEMORY_SIZE,
        noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA
    )
    
    return environment, agent

def _prefill_replay_buffer(agent, environment):
    """Pre-fill the replay buffer with initial experiences."""
    prefill_size = min(BATCH_SIZE, 2048)
    logger.info(f"Pre-filling replay buffer with {prefill_size} experiences...")
    
    progress_bar = tqdm(total=prefill_size, desc="Pre-filling Buffer")
    
    while len(agent.memory) < prefill_size:
        state = environment.reset()
        done = False
        
        while not done and len(agent.memory) < prefill_size:
            action = agent.act(state, explore=True)
            next_state, reward, done, _ = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            progress_bar.update(1)
    
    progress_bar.close()
    logger.info("Replay buffer pre-filled.")
    environment.reset()
    logger.info("Environment reset after prefill.")

def _create_dataset_iterator(agent):
    """Create and return dataset iterator for training."""
    dataset = agent.memory.as_dataset(BATCH_SIZE)
    if dataset:
        return iter(dataset)
    else:
        logger.error("Failed to create dataset iterator")
        return None

def _run_training_episode(agent, environment, dataset_iterator, training_params):
    """Execute a single training episode and return results."""
    state = environment.reset()
    total_reward = 0.0
    done = False
    episode_losses = []
    
    # Collect experiences for this episode
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = environment.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        training_params['steps_since_training'] += 1
    
    # Train agent if enough experiences collected
    if len(agent.memory) >= BATCH_SIZE and training_params['steps_since_training'] >= training_params['frequency']:
        for _ in range(4):  # Train multiple times per episode
            losses = agent.train(dataset_iterator)
            if losses:
                episode_losses.append(losses)
        training_params['steps_since_training'] = 0
    
    return total_reward, episode_losses, info

def _calculate_episode_metrics(episode_losses):
    """Calculate and return mean losses for the episode."""
    if not episode_losses:
        return None, None
    
    mean_actor_loss = np.mean([
        float(loss["actor_loss"]) for loss in episode_losses 
        if "actor_loss" in loss
    ])
    
    mean_critic_loss = np.mean([
        float(loss["critic_loss"]) for loss in episode_losses 
        if "critic_loss" in loss
    ])
    
    return mean_actor_loss, mean_critic_loss

def _should_stop_early(episode, reward_history, best_reward, early_stopping_counter, warmup_episodes, patience):
    """Check if early stopping criteria are met."""
    if episode <= warmup_episodes:
        return False, best_reward, early_stopping_counter
    
    current_avg_reward = np.mean(reward_history)
    if current_avg_reward > best_reward:
        return False, current_avg_reward, 0
    else:
        early_stopping_counter += 1
        return early_stopping_counter >= patience, best_reward, early_stopping_counter

def train_agent(debug=False, time_limit_hours=None, max_time_hours=None):
    """Main training pipeline with monitoring, checkpointing, and testing."""
    # Initialize monitoring and environment
    monitor = PerformanceMonitor(log_interval=10)
    monitor.start_monitoring()
    training_start_time = time.time()
    
    # Convert time limits to seconds
    time_limit_seconds = time_limit_hours * 3600 if time_limit_hours else None
    max_time_seconds = max_time_hours * 3600 if max_time_hours else None
    
    if time_limit_hours:
        logger.info(f"Starting training with {time_limit_hours:.1f} hour time limit")
    if max_time_hours:
        logger.info(f"Maximum training time: {max_time_hours:.1f} hours (hard limit)")
    
    logger.info("Starting optimized training with performance monitoring...")
    
    environment, agent = _initialize_training_environment(debug)
    _prefill_replay_buffer(agent, environment)
    dataset_iterator = _create_dataset_iterator(agent)
    
    if not dataset_iterator:
        return agent
    
    # Training configuration
    training_config = {
        'checkpoint_interval': 25,
        'early_stopping_patience': 8,
        'warmup_episodes': 50,
        'reward_history_size': 50
    }
    
    training_params = {
        'frequency': 1,
        'steps_since_training': 0
    }
    
    # Training state
    best_reward = -np.inf
    early_stopping_counter = 0
    reward_history = deque(maxlen=training_config['reward_history_size'])
    
    # Main training loop
    progress_bar = tqdm(range(TRAINING_EPISODES), desc="Training Progress")
    
    for episode in progress_bar:
        episode_start_time = time.time()
        current_time = time.time()
        elapsed_time = current_time - training_start_time
        
        # Check hard time limit (stops immediately)
        if max_time_seconds and elapsed_time >= max_time_seconds:
            logger.info(f"Hard time limit reached ({max_time_hours:.1f}h). Stopping training.")
            break
        
        # Check soft time limit (finish current episode then stop)
        if time_limit_seconds and elapsed_time >= time_limit_seconds:
            logger.info(f"Time limit reached ({time_limit_hours:.1f}h). Finishing current episode...")
            # Continue with current episode but mark to stop after
            time_limit_reached = True
        else:
            time_limit_reached = False
        
        total_reward, episode_losses, info = _run_training_episode(
            agent, environment, dataset_iterator, training_params
        )
        
        # Calculate and display metrics
        mean_actor_loss, mean_critic_loss = _calculate_episode_metrics(episode_losses)
        
        if mean_actor_loss is not None:
            current_metrics = monitor.get_current_metrics()
            gpu_usage = current_metrics['gpu_usage'] if current_metrics else 0
            episode_time = time.time() - episode_start_time
            
            progress_bar.set_postfix({
                "reward": f"{total_reward:.6f}",
                "value": info["portfolio_value"],
                "actor_loss": f"{mean_actor_loss:.4f}",
                "critic_loss": f"{mean_critic_loss:.4f}",
                "gpu": f"{gpu_usage:.1f}%",
                "ep_time": f"{episode_time:.1f}s"
            })
        
        reward_history.append(total_reward)
        
        # Checkpoint saving
        if episode % training_config['checkpoint_interval'] == 0:
            agent.save_models()
        
        # Early stopping check
        should_stop, best_reward, early_stopping_counter = _should_stop_early(
            episode, reward_history, best_reward, early_stopping_counter,
            training_config['warmup_episodes'], training_config['early_stopping_patience']
        )
        
        if should_stop:
            logger.info(f"Early stopping triggered after {episode} episodes.")
            break
        
        # Save best model
        if best_reward > -np.inf and np.mean(reward_history) == best_reward:
            agent.save_models("./models")
        
        # Check if time limit was reached
        if time_limit_reached:
            logger.info(f"Training stopped after {episode + 1} episodes due to time limit.")
            break
    
    # Training completion
    monitor.stop_monitoring()
    total_training_time = time.time() - training_start_time
    
    logger.info(f"Training completed in {total_training_time:.1f} seconds")
    logger.info(f"Average time per episode: {total_training_time/TRAINING_EPISODES:.2f} seconds")
    
    monitor.print_summary()
    monitor.plot_performance("training_performance.png")
    
    # Quick test after training
    try:
        logger.info("Running post-training evaluation...")
        test_agent(agent, episodes=5)
    except Exception as e:
        logger.error(f"Post-training evaluation failed: {e}")
        logger.error("Training completed successfully, but evaluation had issues")
    
    return agent





def optimize_agent(trials=20, time_limit_hours=None, max_time_hours=None):
    """Run hyperparameter optimization using Optuna"""
    try:
        import optuna
        import json
        
        # Convert time limits to seconds
        time_limit_seconds = time_limit_hours * 3600 if time_limit_hours else None
        max_time_seconds = max_time_hours * 3600 if max_time_hours else None
        
        if time_limit_hours:
            logger.info(f"Starting optimization with {trials} trials and {time_limit_hours:.1f}h time limit")
        else:
            logger.info(f"Starting hyperparameter optimization with {trials} trials")
        
        optimization_start_time = time.time()
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            gamma = trial.suggest_float("gamma", 0.9, 0.99)
            noise_sigma = trial.suggest_float("noise_sigma", 0.05, 0.3)
            noise_theta = trial.suggest_float("noise_theta", 0.1, 0.5)
            reward_scaling_factor = trial.suggest_float("reward_scaling_factor", 1.0, 50.0)
            
            # Create environment and agent with suggested params
            env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, 
                                        stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, reward_scaling_factor)
            
            agent = DRLAgent(env.state_size, len(env.symbols), learning_rate=learning_rate, 
                           gamma=gamma, memory_size=10000, noise_sigma=noise_sigma, noise_theta=noise_theta)
            
            # Quick training and evaluation
            total_portfolio_values = []
            for episode in range(10):  # Quick evaluation
                # Check time limits during optimization
                current_time = time.time()
                elapsed_time = current_time - optimization_start_time
                
                if max_time_seconds and elapsed_time >= max_time_seconds:
                    logger.info(f"Optimization hard time limit reached. Stopping trial early.")
                    break
                    
                if time_limit_seconds and elapsed_time >= time_limit_seconds:
                    logger.info(f"Optimization time limit reached. Finishing current trial.")
                    break
                
                state = env.reset()
                episode_reward = 0
                final_portfolio_value = INITIAL_BALANCE
                
                for step in range(100):  # Short episodes
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    
                    # Track final portfolio value
                    if 'portfolio_value' in info:
                        final_portfolio_value = info['portfolio_value']
                    
                    if done:
                        break
                
                total_portfolio_values.append(final_portfolio_value)
                
                if len(agent.memory) >= batch_size:
                    try:
                        dataset = agent.memory.as_dataset(batch_size)
                        if dataset:
                            dataset_iterator = iter(dataset)
                            agent.train(dataset_iterator)
                    except Exception as e:
                        logger.warning(f"Training error: {e}")
            
            # Return average portfolio value (more meaningful than reward)
            avg_portfolio_value = sum(total_portfolio_values) / len(total_portfolio_values) if total_portfolio_values else INITIAL_BALANCE
            return avg_portfolio_value
        
        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        
        # Use Optuna's built-in timeout if time limit is specified
        if time_limit_seconds:
            study.optimize(objective, n_trials=trials, timeout=time_limit_seconds)
        else:
            study.optimize(objective, n_trials=trials)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        # Save best parameters to hyperparameters.json (used by config)
        best_params = study.best_trial.params
        with open("hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        logger.info("Optimization completed successfully")
        logger.info(f"Best parameters saved to hyperparameters.json: {best_params}")
        return best_params
        
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return None
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None


def test_agent(agent=None, episodes=None, debug=False, time_limit_hours=None, max_time_hours=None):
    """Tests the trained agent.

    Args:
        agent: The DRLAgent instance to test. If None, a new agent is initialized and models are loaded.
        episodes: Number of episodes to test. If None, uses TEST_EPISODES from config.
        debug: Enable debug logging.
        time_limit_hours: Soft time limit in hours.
        max_time_hours: Hard time limit in hours.
    """
    if episodes is None:
        episodes = TEST_EPISODES
    
    # Convert time limits to seconds
    time_limit_seconds = time_limit_hours * 3600 if time_limit_hours else None
    max_time_seconds = max_time_hours * 3600 if max_time_hours else None
    
    test_start_time = time.time()
    
    if time_limit_hours:
        logger.info(f"Starting testing with {time_limit_hours:.1f} hour time limit")
    if max_time_hours:
        logger.info(f"Maximum testing time: {max_time_hours:.1f} hours (hard limit)")
    if agent is None:
        agent = DRLAgent(
            STATE_DIM,
            ACTION_DIM,
            LEARNING_RATE,
            GAMMA,
            MEMORY_SIZE,
            noise_sigma=NOISE_SIGMA,
            noise_theta=NOISE_THETA,
        )
        agent.load_models("./models")

    invalidate_cache(days=0)
    StockTradingEnvironment.clear_in_memory_cache()
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, REWARD_SCALING_FACTOR, debug=debug)

    total_rewards = []
    portfolio_values = []

    for episode in tqdm(range(episodes), desc="Testing Progress"):
        current_time = time.time()
        elapsed_time = current_time - test_start_time
        
        # Check hard time limit
        if max_time_seconds and elapsed_time >= max_time_seconds:
            logger.info(f"Hard time limit reached ({max_time_hours:.1f}h). Stopping testing.")
            break
            
        # Check soft time limit
        if time_limit_seconds and elapsed_time >= time_limit_seconds:
            logger.info(f"Time limit reached ({time_limit_hours:.1f}h). Finishing current episode...")
            time_limit_reached = True
        else:
            time_limit_reached = False
        
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_portfolio = [INITIAL_BALANCE]

        while not done:
            action = agent.act(state, explore=False)
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            episode_portfolio.append(info["portfolio_value"])

        total_rewards.append(total_reward)
        portfolio_values.append(episode_portfolio)

        if episode % 10 == 0:
            logger.info(
                f"Test Episode {episode + 1}/{episodes}, Final Reward: {total_reward:.6f}, Final Portfolio Value: {float(info['portfolio_value']):.2f}"
            )
        
        # Stop if time limit reached
        if time_limit_reached:
            logger.info(f"Testing stopped after {episode + 1} episodes due to time limit.")
            break

    # Calculate statistics
    final_values = [portfolio[-1] for portfolio in portfolio_values]
    mean_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    std_final_value = np.std(final_values)

    logger.info(
        f"Testing completed. Mean final portfolio value: ${mean_final_value:.2f} \u00b1 ${std_final_value:.2f}"
    )

    # Simple plotting
    plt.figure(figsize=(12, 6))
    
    # Plot average and individual episodes
    if portfolio_values:
        max_len = max(len(p) for p in portfolio_values)
        aligned = []
        for p in portfolio_values:
            try:
                p_array = np.array(p, dtype=np.float64).flatten()
                padded = np.pad(p_array, (0, max_len - len(p_array)), 'edge')
                aligned.append(padded)
            except:
                aligned.append(np.full(max_len, INITIAL_BALANCE, dtype=np.float64))
        
        avg_portfolio = np.mean(aligned, axis=0)
        plt.plot(avg_portfolio, "b-", linewidth=2, label="Average")
        
        # Plot a few individual episodes
        for i in range(min(5, len(portfolio_values))):
            plt.plot(portfolio_values[i], "r-", alpha=0.3)
    
    plt.title("Portfolio Performance")
    plt.xlabel("Steps")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("test_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Distribution plot
    plt.figure(figsize=(8, 5))
    plt.hist(final_values, bins=20, alpha=0.7, color="skyblue")
    plt.axvline(mean_final_value, color="red", linestyle="--", label=f"Mean: ${mean_final_value:.2f}")
    plt.title("Final Portfolio Values")
    plt.xlabel("Value ($)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("final_values_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    return agent

def run_live_trading():
    """
    Runs the agent in live trading mode with Alpaca.
    """
    logger.info("Starting live trading...")
    # Load trained agent
    agent = DRLAgent(
        STATE_DIM,
        ACTION_DIM,
        LEARNING_RATE,
        GAMMA,
        MEMORY_SIZE,
        noise_sigma=NOISE_SIGMA,
        noise_theta=NOISE_THETA,
    )
    if not agent.load_models("./models"):
        logger.error("Failed to load model weights. Please train the model first.")
        return

    # Connect Alpaca lazily here
    global trading_client, stock_data_client
    if trading_client is None or stock_data_client is None:
        try:
            trading_client = TradingClient(
                ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER_TRADING
            )
            stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            logger.info("Successfully connected to Alpaca API for live trading")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            return

    trading_system = AlpacaTradingSystem(agent, STOCK_SYMBOLS, trading_client, stock_data_client)

    # Invalidate cache at the start to ensure fresh data
    invalidate_cache(days=1)
    StockTradingEnvironment.clear_in_memory_cache()

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
                    clock = trading_client.get_clock()
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
                            f"Failed to connect to API after {max_retries} attempts"
                        )
                        # Wait longer before trying again to avoid hammering the API
                        time.sleep(300)  # 5 minutes
                        retry_delay = 30  # Reset delay
                        continue

            try:
                if clock.is_open:
                    logger.info(f'\n--- Executing Trades ---')
                    actions = trading_system.execute_trades()

                    # Log the actions in a more readable format
                    action_summary = []
                    if actions:
                        for i, symbol in enumerate(STOCK_SYMBOLS):
                            signal = actions[i]
                            if signal > 0.1:  # Buy signal
                                action_summary.append(f"BUY {symbol}: {signal:.2f}")
                            elif signal < -0.1:  # Sell signal
                                action_summary.append(f"SELL {symbol}: {-signal:.2f}")

                        # Log all weights
                        logger.info("\n--- Model Weights ---")
                        for i, symbol in enumerate(STOCK_SYMBOLS):
                            logger.info(f"{symbol}: Signal: {actions[i]:.2f}")

                    if action_summary:
                        logger.info(f"Actions taken: {', '.join(action_summary)}")
                    else:
                        logger.info("No significant trading actions taken")

                    # Get current portfolio value with retry logic
                    for attempt in range(max_retries):
                        try:
                            account = trading_client.get_account()
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
                logger.error(f"Error: {e}")
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
        choices=["train", "test", "live", "optimize"],
        default="train",
        help="Operation mode: train, test, live trading, or optimize",
    )
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
    parser.add_argument("--api_key", type=str, help="Alpaca API Key")
    parser.add_argument("--api_secret", type=str, help="Alpaca API Secret Key")
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_EPISODES,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TEST_EPISODES,
        help="Number of episodes for testing",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Time limit in hours (alternative to episode/trial counts)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        help="Maximum time in hours (safety limit, stops even mid-episode)",
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
        # Perform health checks before starting
        if not check_system_health():
            logger.error("System health checks failed. Exiting.")
            sys.exit(1)
        
        if args.mode == "train":
            logger.info("Training agent...")
            if args.time_limit or args.max_time:
                logger.info(f"Time limits: soft={args.time_limit}h, hard={args.max_time}h")
            else:
                logger.info(f"Number of epochs: {args.epochs}")
                TRAINING_EPISODES = args.epochs
            agent = train_agent(debug=args.debug, time_limit_hours=args.time_limit, max_time_hours=args.max_time)
        elif args.mode == "test":
            logger.info("Testing agent...")
            if args.time_limit or args.max_time:
                logger.info(f"Time limits: soft={args.time_limit}h, hard={args.max_time}h")
            else:
                logger.info(f"Number of test episodes: {args.episodes}")
            test_agent(episodes=args.episodes, debug=args.debug, time_limit_hours=args.time_limit, max_time_hours=args.max_time)
        elif args.mode == "live":
            logger.info("Starting live trading...")
            run_live_trading()
        elif args.mode == "optimize":
            logger.info("Starting hyperparameter optimization...")
            if args.time_limit or args.max_time:
                logger.info(f"Time limits: soft={args.time_limit}h, hard={args.max_time}h")
            else:
                logger.info(f"Number of trials: {args.trials}")
            optimize_agent(args.trials, time_limit_hours=args.time_limit, max_time_hours=args.max_time)
        # train_fast removed for simplicity; use unified training
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())