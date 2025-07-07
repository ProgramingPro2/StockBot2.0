# StockBot 2.0

This is a sophisticated stock trading bot that uses a Deep Reinforcement Learning (DRL) agent to make trading decisions. The bot is designed to be autonomous, with features for training, testing, hyperparameter optimization, and live trading on the Alpaca platform.

## Features

*   **Deep Reinforcement Learning:** Utilizes a DDPG (Deep Deterministic Policy Gradient) agent to learn and adapt its trading strategy.
*   **GPU Acceleration:** Optimized for GPU usage to significantly speed up the training process, leveraging mixed precision training for RTX 3090.
*   **Alpaca Integration:** Seamlessly connects with the Alpaca API for live or paper trading, with improved retry mechanisms for order submission.
*   **Automated Trading:** Includes a bash script for setting up cron jobs to automate daily trading and weekly training sessions.
*   **Robust Data Handling:** Caches historical stock data to minimize API calls and speeds up initialization with parallel fetching. Includes robust retry logic and data completeness checks for `yfinance`.
*   **Hyperparameter Optimization:** Integrates `Optuna` to automatically find optimal hyperparameters for the DRL agent, improving model quality and profitability.
*   **Robustness:** Implements error handling, retry logic, and wash trade prevention to ensure stable operation.
*   **Improved State Representation:** Enhanced historical data padding in state representation for more accurate agent learning.
*   **Enhanced Exploration:** Uses Ornstein-Uhlenbeck noise for better exploration during training.
*   **Stable Training:** Incorporates gradient clipping and soft target updates for more stable DRL agent training.
*   **Easy Configuration:** Uses a `.env` file for managing API keys and a `stocks.json` file for defining the trading universe.
*   **Clearer Visualizations:** Generates improved performance plots during testing for better analysis.

## How It Works

The bot operates based on the interaction between a DRL agent and a simulated stock market environment.

1.  **Data Collection:** The `StockTradingEnvironment` downloads historical stock data for the symbols defined in `stocks.json` using the `yfinance` library, with parallel fetching and robust error handling.
2.  **Training:** In `train` mode, the `DRLAgent` is trained on the historical data. It learns to maximize its rewards (i.e., portfolio value) by making decisions in the simulated environment. The agent's "brain" consists of two neural networks:
    *   An **Actor Network** that decides the action to take (buy, sell, or hold).
    *   A **Critic Network** that evaluates the quality of the actor's actions.
3.  **Hyperparameter Optimization:** In `optimize` mode, `Optuna` is used to systematically search for the best combination of hyperparameters (e.g., learning rate, batch size, gamma, noise parameters) to maximize the bot's performance.
4.  **Testing:** In `test` mode, the trained agent is evaluated on unseen historical data to assess its performance. The results are saved as PNG images (`test_results.png` and `final_values_distribution.png`).
5.  **Live Trading:** In `live` mode, the `AlpacaTradingSystem` connects the agent to a live Alpaca account. The agent uses real-time market data to make trading decisions and execute orders through the Alpaca API, with retry mechanisms for order submission.

## Getting Started

### Prerequisites

*   Python 3.8+
*   An Alpaca paper or live trading account
*   (Optional) An NVIDIA GPU with CUDA and cuDNN installed for accelerated training

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd StockBot2.0
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up your Alpaca API keys:**

    *   Copy the example `.env.example` file to a new `.env` file:

        ```bash
        cp .env.example .env
        ```

    *   Open the `.env` file and replace `"YOUR_API_KEY"` and `"YOUR_SECRET_KEY"` with your actual Alpaca API key and secret key.

2.  **Define your stock universe:**

    *   Edit the `stocks.json` file to include the stock symbols you want the bot to trade. The symbols should be in uppercase.

## Usage

The `main.py` script can be run in several modes:

*   **Train:**

    ```bash
    python main.py --mode train
    ```

    This will train the DRL agent using the historical data and save the trained model to the `./models` directory.

*   **Test:**

    ```bash
    python main.py --mode test
    ```

    This will test the trained agent and generate performance plots.

*   **Hyperparameter Optimization:**

    ```bash
    python main.py --mode optimize
    ```

    This will run `Optuna` to find optimal hyperparameters for the DRL agent. The best parameters found will update the global configuration.

*   **Live Trading:**

    ```bash
    python main.py --mode live
    ```

    This will run the bot in live trading mode using your Alpaca account.

### Command-line Arguments

*   `--mode`: The operation mode (`train`, `test`, `live`, `optimize`). Default is `train`.
*   `--epochs`: The number of episodes to train for. Default is `1000`.
*   `--gpu`: Force GPU usage. The script will raise an error if a GPU is not available.

## Automation

The `market_automation.sh` script is provided to automate the execution of the bot.

### Installation

To set up the cron jobs for automated trading and training, run the following command:

```bash
./market_automation.sh install
```

This will add the following cron jobs to your crontab:

*   Run the bot in live mode every weekday at 9 AM.
*   Run the bot in training mode every Friday at 4:30 PM.
*   Run the bot in live mode 5 minutes after a system reboot.

### Manual Execution

You can also run the script manually with a specific mode:

*   **Live Mode:**

    ```bash
    ./market_automation.sh live
    ```

*   **Training Mode:**

    ```bash
    ./market_automation.sh training
    ```

## Files and Directories

```
.
├── .env.example            # Example environment file for API keys
├── .gitignore              # Git ignore file
├── README.md               # This file
├── final_values_distribution.png # Plot of the final portfolio values from testing
├── main.py                 # The main application script
├── market_automation.sh    # Script for automating the bot
├── requirements.txt        # Python package requirements
├── stocks.json             # List of stock symbols to trade
├── checkpoints/            # Directory for saving model checkpoints
├── data_cache/             # Directory for caching historical stock data
└── models/                 # Directory for saving the best trained models
```

## Disclaimer

This project is for educational purposes only and should not be considered financial advice. Algorithmic trading involves significant risks, and you should be fully aware of these risks before using this bot for live trading. The creators of this project are not responsible for any financial losses you may incur.