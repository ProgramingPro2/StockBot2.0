#!/bin/bash

# StockBot2 Automation Service Installation Script
# This script installs and configures the stockbot automation service

set -e  # Exit on any error

echo "🚀 Installing StockBot2 Automation Service..."

# Check if running as root for service installation
if [[ $EUID -eq 0 ]]; then
    echo "❌ This script should NOT be run as root!"
    echo "   Run it as your regular user, it will ask for sudo when needed."
    exit 1
fi

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="stockbot-automation"
SERVICE_FILE="${SERVICE_NAME}.service"

echo "📁 Working directory: $SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "❌ Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "   Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if market_automation.py exists
if [ ! -f "$SCRIPT_DIR/market_automation.py" ]; then
    echo "❌ market_automation.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SCRIPT_DIR/$SERVICE_FILE" ]; then
    echo "❌ Service file $SERVICE_FILE not found in $SCRIPT_DIR"
    exit 1
fi

# Stop the service if it's running
echo "🛑 Stopping existing service (if running)..."
sudo systemctl stop $SERVICE_NAME 2>/dev/null || true

# Copy service file to systemd directory
echo "📋 Installing service file..."
sudo cp "$SCRIPT_DIR/$SERVICE_FILE" /etc/systemd/system/

# Set proper permissions
sudo chmod 644 /etc/systemd/system/$SERVICE_FILE

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service to start on boot
echo "✅ Enabling service for startup..."
sudo systemctl enable $SERVICE_NAME

# Start the service
echo "🚀 Starting service..."
sudo systemctl start $SERVICE_NAME

# Wait a moment for the service to start
sleep 3

# Check service status
echo "📊 Service status:"
sudo systemctl status $SERVICE_NAME --no-pager -l

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Useful commands:"
echo "   Check status:    sudo systemctl status $SERVICE_NAME"
echo "   View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "   Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "   Start service:   sudo systemctl start $SERVICE_NAME"
echo "   Restart service: sudo systemctl restart $SERVICE_NAME"
echo "   Disable startup: sudo systemctl disable $SERVICE_NAME"
echo ""
echo "📝 The service will now:"
echo "   • Start automatically on system boot"
echo "   • Run live trading during market hours"
echo "   • Stop live trading when market closes"
echo "   • Run optimization after Friday market close"
echo "   • Start training after optimization completes"
echo "   • Restart automatically if it crashes"
echo ""
echo "📊 To monitor in real-time:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
