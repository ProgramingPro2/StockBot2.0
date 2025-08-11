"""
StockBot2 - A Deep Reinforcement Learning Trading Bot

A sophisticated algorithmic trading system that uses deep reinforcement learning
to make trading decisions in the stock market.
"""

__version__ = "2.0.0"
__author__ = "StockBot Team"
__email__ = "contact@stockbot.com"

from .drl_agent import DRLAgent
from .stock_env import StockTradingEnvironment
from .alpaca_trade_system import AlpacaTradingSystem
from .performance_monitor import PerformanceMonitor

__all__ = [
    "DRLAgent",
    "StockTradingEnvironment", 
    "AlpacaTradingSystem",
    "PerformanceMonitor",
]
