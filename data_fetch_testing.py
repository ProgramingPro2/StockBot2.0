import os
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import dotenv # Import dotenv

class AlpacaDataFetcher:
    """
    A class to fetch stock bar data from the Alpaca Markets API.
    """
    # This BASE_URL is specific to the stock bars endpoint of the Data API
    BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, api_key: str, secret_key: str, lookback_window_days: int = 30):
        """
        Initializes the data fetcher with API credentials and a lookback window.

        Args:
            api_key (str): Your Alpaca API Key ID.
            secret_key (str): Your Alpaca Secret Key.
            lookback_window_days (int): Additional days to look back beyond one year.
                                        Total lookback will be (365 + lookback_window_days * 2) days.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.lookback_window = lookback_window_days

    def _get_headers(self) -> dict:
        """
        Returns the authentication headers required for Alpaca API requests.
        """
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }

    def get_stock_bars(self, symbol: str = "NVDA", timeframe: str = "1H") -> dict | None:
        """
        Fetches historical stock bar data for a given symbol and timeframe.

        Args:
            symbol (str): The stock symbol (e.g., "NVDA", "AAPL", "COST").
            timeframe (str): The aggregation period for each bar (e.g., "1H", "1D", "30Min").

        Returns:
            dict | None: The JSON response containing bar data, or None if an error occurred.
        """
        # Get the current time directly in UTC and make it timezone-aware.
        end_date = datetime.now(timezone.utc)
        # Calculate start_date based on your specified logic: 1 year plus additional lookback
        start_date = end_date - timedelta(days=365 + self.lookback_window * 2)

        # Format dates to RFC-3339 with 'Z' for UTC and URL-encode them.
        # The .strftime('%Y-%m-%dT%H:%M:%SZ') creates the correct RFC-3339 string for UTC.
        # quote() then URL-encodes special characters (like ':' to '%3A').
        start_date_str = quote(start_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        end_date_str = quote(end_date.strftime('%Y-%m-%dT%H:%M:%SZ'))

        # Construct the URL with parameters
        url = (
            f"{self.BASE_URL}?"
            f"symbols={symbol}&"
            f"timeframe={timeframe}&"
            f"start={start_date_str}&"
            f"end={end_date_str}&"
            f"feed=iex&" # Added '&' here
            f"sort=desc" # <--- Added this parameter to sort in descending order
            # Optional parameters like 'limit', 'adjustment' can be added here
            # f"&limit=1000" # If you uncomment this, ensure it's after sort
            # f"&adjustment=raw"
        )

        print(f"Requesting URL: {url}") # For debugging purposes

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            print(f"Response content: {response.text}")
        except requests.exceptions.RequestException as err:
            print(f"An error occurred during the request: {err}")
        return None

if __name__ == "__main__":
    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Retrieve API keys from environment variables
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY and/or ALPACA_SECRET_KEY not found in environment variables.")
        print("       Please ensure you have a .env file in the same directory")
        print("       with content like:")
        print("       ALPACA_API_KEY=\"YOUR_API_KEY_ID\"")
        print("       ALPACA_SECRET_KEY=\"YOUR_SECRET_KEY\"")
        exit(1)

    # Initialize the data fetcher
    # You can change the lookback_window_days here if needed
    fetcher = AlpacaDataFetcher(ALPACA_API_KEY, ALPACA_SECRET_KEY, lookback_window_days=30)

    print("--- Fetching Stock Bars Data ---")

    # Example 1: Fetching NVDA data
    print("\nAttempting to fetch NVDA 1-hour bars...")
    nvda_bars_data = fetcher.get_stock_bars(symbol="NVDA", timeframe="30Min")
    if nvda_bars_data and 'bars' in nvda_bars_data and nvda_bars_data['bars']:
        print("Successfully fetched NVDA bars data!")
        # Access bars for 'NVDA' specifically
        nvda_bars = nvda_bars_data['bars'].get('NVDA', [])
        if nvda_bars:
            print(f"Number of NVDA bars received: {len(nvda_bars)}")
            # When sorting descending, the first bar will be the most recent
            # and the last bar will be the oldest within the returned set.
            print(f"First NVDA bar (most recent): {nvda_bars[0]}")
            print(f"Last NVDA bar (oldest): {nvda_bars[-1]}")
            # print(f"Full NVDA bars data: {nvda_bars}") # Uncomment if you want to see all data
        else:
            print("No bars found for NVDA in the response.")
    else:
        print("Failed to fetch NVDA bars data or response was empty.")

    # Example 2: Fetching COST data with a different timeframe
    print("\nAttempting to fetch COST 30-minute bars...")
    cost_bars_data = fetcher.get_stock_bars(symbol="COST", timeframe="30Min")
    if cost_bars_data and 'bars' in cost_bars_data and cost_bars_data['bars']:
        print("Successfully fetched COST bars data!")
        cost_bars = cost_bars_data['bars'].get('COST', [])
        if cost_bars:
            print(f"Number of COST bars received: {len(cost_bars)}")
            # When sorting descending, the first bar will be the most recent
            # and the last bar will be the oldest within the returned set.
            print(f"First COST bar (most recent): {cost_bars[0]}")
            print(f"Last COST bar (oldest): {cost_bars[-1]}")
            # print(f"Full COST bars data: {cost_bars}") # Uncomment if you want to see all data
        else:
            print("No bars found for COST in the response.")
    else:
        print("Failed to fetch COST bars data or response was empty.")