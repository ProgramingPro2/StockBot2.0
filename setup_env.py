#!/usr/bin/env python3
"""
Setup script for StockBot2-Experimental
Helps users configure their environment variables and validate setup.
"""

import os
from pathlib import Path

def create_env_file():
    """Create a .env file with user input"""
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return False
    
    print("\n🔧 StockBot2-Experimental Setup")
    print("=" * 40)
    print("You'll need your Alpaca API credentials.")
    print("Get them from: https://app.alpaca.markets/paper/dashboard/overview")
    print()
    
    # Get API credentials
    api_key = input("Enter your Alpaca API Key: ").strip()
    secret_key = input("Enter your Alpaca Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("❌ API credentials are required!")
        return False
    
    # Ask about paper trading
    paper_trading = input("Use paper trading? (Y/n): ").strip().lower()
    paper_trading = "True" if paper_trading != 'n' else "False"
    
    # Create .env file
    env_content = f"""# Alpaca API Configuration
ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}
ALPACA_PAPER_TRADING={paper_trading}

# Optional: Set to "False" to disable yfinance fallback
USE_YFINANCE=True
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created {env_path}")
    print(f"📝 Paper trading enabled: {paper_trading}")
    return True

def validate_setup():
    """Validate the current setup"""
    print("\n🔍 Validating Setup")
    print("=" * 40)
    
    # Check .env file
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check API credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ API credentials not found in .env file")
        return False
    
    print("✅ API credentials found")
    
    # Check paper trading setting
    paper_trading = os.getenv("ALPACA_PAPER_TRADING", "True").lower() == "true"
    print(f"✅ Paper trading: {'enabled' if paper_trading else 'disabled'}")
    
    # Check required directories
    required_dirs = ["./models", "./data_cache", "./checkpoints"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")
    
    # Check stocks.json
    stocks_path = Path("stocks.json")
    if stocks_path.exists():
        print("✅ stocks.json found")
    else:
        print("⚠️  stocks.json not found (will use default symbols)")
    
    print("\n🎉 Setup validation complete!")
    return True

def main():
    """Main setup function"""
    print("🚀 StockBot2-Experimental Setup")
    print("=" * 40)
    
    # Check if .env exists
    env_path = Path(".env")
    if not env_path.exists():
        print("No .env file found. Let's create one!")
        if not create_env_file():
            return
    else:
        print("Found existing .env file")
    
    # Validate setup
    if validate_setup():
        print("\n🎯 You're ready to start!")
        print("\nNext steps:")
        print("1. Activate your virtual environment: source .venv/bin/activate")
        print("2. Train the agent: python main.py --mode train")
        print("3. Test the agent: python main.py --mode test")
        print("4. Start live trading: python main.py --mode live")
    else:
        print("\n❌ Setup validation failed. Please check your configuration.")

if __name__ == "__main__":
    main() 