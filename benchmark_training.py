#!/usr/bin/env python3
"""
Benchmark script to test training performance improvements.
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from tqdm import tqdm

from config import *
from drl_agent import DRLAgent
from stock_env import StockTradingEnvironment
from performance_monitor import PerformanceMonitor
from alpaca_trade_system import AlpacaTradingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_training(episodes=50):
    """
    Benchmark the training performance with a small number of episodes.
    
    Args:
        episodes: Number of episodes to run for benchmarking
    """
    logger.info(f"Starting training benchmark with {episodes} episodes...")
    
    # Start performance monitoring
    monitor = PerformanceMonitor(log_interval=5)
    monitor.start_monitoring()
    
    # Initialize environment and agent
    from alpaca.data.historical import StockHistoricalDataClient
    stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    
    env = StockTradingEnvironment(
        STOCK_SYMBOLS, INITIAL_BALANCE, LOOKBACK_WINDOW, 
        stock_data_client, CACHE_DIR, MAX_SHARES_PER_TRADE, REWARD_SCALING_FACTOR
    )
    
    agent = DRLAgent(
        STATE_DIM, ACTION_DIM, LEARNING_RATE, GAMMA, MEMORY_SIZE,
        noise_sigma=NOISE_SIGMA, noise_theta=NOISE_THETA
    )
    
    # Pre-fill buffer
    logger.info("Pre-filling replay buffer...")
    while len(agent.memory) < BATCH_SIZE:
        state = env.reset()
        done = False
        while not done and len(agent.memory) < BATCH_SIZE:
            action = agent.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
    
    # Create dataset
    dataset = agent.memory.as_dataset(BATCH_SIZE)
    dataset_iterator = iter(dataset)
    
    # Training loop
    start_time = time.time()
    episode_times = []
    
    for episode in tqdm(range(episodes), desc="Benchmark Training"):
        episode_start = time.time()
        
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        # Train
        if len(agent.memory) >= BATCH_SIZE:
            for _ in range(4):
                try:
                    agent.train(dataset_iterator)
                except StopIteration:
                    dataset = agent.memory.as_dataset(BATCH_SIZE)
                    dataset_iterator = iter(dataset)
                    break
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        if episode % 10 == 0:
            current_metrics = monitor.get_current_metrics()
            if current_metrics:
                logger.info(f"Episode {episode}: Time={episode_time:.2f}s, "
                          f"GPU={current_metrics['gpu_usage']:.1f}%, "
                          f"Reward={total_reward:.2f}")
    
    total_time = time.time() - start_time
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Total episodes: {episodes}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per episode: {np.mean(episode_times):.2f} seconds")
    print(f"Min episode time: {np.min(episode_times):.2f} seconds")
    print(f"Max episode time: {np.max(episode_times):.2f} seconds")
    print(f"Episodes per hour: {3600/np.mean(episode_times):.1f}")
    
    # Performance summary
    monitor.print_summary()
    monitor.plot_performance("benchmark_performance.png")
    
    return {
        'total_time': total_time,
        'avg_episode_time': np.mean(episode_times),
        'episodes_per_hour': 3600/np.mean(episode_times)
    }

if __name__ == "__main__":
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU detected - using CPU")
    
    # Run benchmark
    results = benchmark_training(episodes=50)
    
    print(f"\nEstimated time for 1000 episodes: {results['total_time'] * 20:.1f} seconds ({results['total_time'] * 20 / 3600:.1f} hours)")
    print(f"Estimated episodes per hour: {results['episodes_per_hour']:.1f}") 