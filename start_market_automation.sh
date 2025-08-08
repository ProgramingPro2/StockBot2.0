#!/bin/bash
# Market Automation Startup Script
cd /home/jpearce/Documents/Python/StockBot2-Experimental
source .venv/bin/activate
python3 market_automation.py >> market_automation.log 2>&1 &
echo $! > market_automation.pid
