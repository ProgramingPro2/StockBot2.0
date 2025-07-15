import os
import pandas as pd
import numpy as np
import time
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
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
import dotenv
from pathlib import Path
import logging
from tqdm import tqdm
import json
import optuna
from concurrent.futures import ThreadPoolExecutor


from noise import OrnsteinUhlenbeckActionNoise
from config import *
from utils import invalidate_cache
from replay_buffer import ReplayBuffer
from drl_agent import DRLAgent
from stock_env import StockTradingEnvironment
from alpaca_trade_system import AlpacaTradingSystem

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

# Load environment variables
dotenv.load_dotenv()

# Connect to Alpaca API
trading_client = TradingClient(
    ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER_TRADING
)
stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def optimize_agent(optimization_episodes=100, optimization_test_trials=5):
    """
    Performs hyperparameter optimization using Optuna.

    Args:
        optimization_episodes: Number of episodes to train the agent for during each trial.
        optimization_test_trials: Number of trials to test the agent for during each optimization.
    """
    logger.info("Starting hyperparameter optimization...")

    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical(
            "batch_size", [256, 512, 1024, 2048, 4096]
        )
        gamma = trial.suggest_float("gamma", 0.9, 0.99)
        noise_sigma = trial.suggest_float("noise_sigma", 0.05, 0.3)
        noise_theta = trial.suggest_float("noise_theta", 0.1, 0.5)
        reward_scaling_factor = trial.suggest_float(
            "reward_scaling_factor", 10.0, 10000.0, log=True
        )

        # Update global parameters or pass them to the agent
        global LEARNING_RATE, BATCH_SIZE, GAMMA
        LEARNING_RATE = learning_rate
        BATCH_SIZE = batch_size
        GAMMA = gamma

        logger.info(
            f"Trial {trial.number}: LR={learning_rate:.6f}, BS={batch_size}, Gamma={gamma:.3f}"
        )

        # Train the agent with suggested hyperparameters
        env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, reward_scaling_factor)
        agent = DRLAgent(
            STATE_DIM,
            ACTION_DIM,
            learning_rate,
            gamma,
            MEMORY_SIZE,
            noise_sigma=noise_sigma,
            noise_theta=noise_theta,
        )
        agent.gamma = gamma  # Ensure agent uses the trial's gamma

        dataset_iterator = None

        for episode in tqdm(
            range(optimization_episodes), desc=f"Trial {trial.number} Training"
        ):  # Add tqdm here
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if len(agent.memory) >= BATCH_SIZE:
                if dataset_iterator is None:
                    dataset = agent.memory.as_dataset(BATCH_SIZE)
                    if dataset:
                        dataset_iterator = iter(dataset)

                if dataset_iterator:
                    for _ in range(2):  # Train less frequently during optimization
                        agent.train(dataset_iterator)

        # Test the agent and return the mean final portfolio value
        test_env = StockTradingEnvironment(
            STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, reward_scaling_factor
        )
        final_values = []
        for _ in tqdm(
            range(optimization_test_trials), desc=f"Trial {trial.number} Testing"
        ):  # Add tqdm here
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
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    logger.info("Optimization finished.")
    logger.info(
        f"""Best trial:
  Value: {study.best_value:.2f}
  Params: {study.best_params}"""
    )

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
    """Trains the DRL agent using historical data with optimized GPU acceleration.

    Returns:
        The trained DRLAgent instance.
    """
    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, REWARD_SCALING_FACTOR)
    agent = DRLAgent(
        STATE_DIM,
        ACTION_DIM,
        LEARNING_RATE,
        GAMMA,
        MEMORY_SIZE,
        noise_sigma=NOISE_SIGMA,
        noise_theta=NOISE_THETA,
    )

    best_reward = -np.inf
    checkpoint_interval = 50  # Save every 50 episodes

    # Create a progress bar for training
    progress_bar = tqdm(range(TRAINING_EPISODES), desc="Training Progress")

    dataset_iterator = None

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
            if dataset_iterator is None:
                dataset = agent.memory.as_dataset(BATCH_SIZE)
                if dataset:
                    dataset_iterator = iter(dataset)

            if dataset_iterator:
                # Multiple training iterations per episode for better learning
                for _ in range(4):  # Train 4x per episode
                    losses = agent.train(dataset_iterator)
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
                    "reward": f"{total_reward:.6f}",  # Display more decimal places
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
    """Tests the trained agent.

    Args:
        agent: The DRLAgent instance to test. If None, a new agent is initialized and models are loaded.
    """
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

    env = StockTradingEnvironment(STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, REWARD_SCALING_FACTOR)

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
        f"Testing completed. Mean final portfolio value: ${mean_final_value:.2f} \u00b1 ${std_final_value:.2f}"
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

    # Create trading system
    trading_system = AlpacaTradingSystem(agent, STOCK_SYMBOLS, trading_client, stock_data_client)

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
        choices=["train", "test", "live", "optimize"],
        default="train",
        help="Operation mode: train, test, live trading, or hyperparameter optimization",
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
        default=25,
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
    except ValueError as e:
        logger.error(f"Error: {e}")
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())