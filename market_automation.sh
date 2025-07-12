#!/bin/bash

# market_automation.sh
# Script to manage daily execution of trading algorithm and Friday training

# Define paths and commands
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="$SCRIPT_DIR/.venv"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get current date for logs
CURRENT_DATE=$(date +"%Y-%m-%d")
LOG_FILE="$LOG_DIR/market_script_$CURRENT_DATE.log"

# Logging function
log() {
	echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# Function to kill existing process if running
kill_existing_process() {
	log "Checking for existing main.py processes"
	if pgrep -f "python.*$MAIN_SCRIPT" >/dev/null; then
		log "Found existing process, killing it"
		pkill -f "python.*$MAIN_SCRIPT"
		sleep 5 # Give the process time to terminate
	else
		log "No existing process found"
	fi
}

# Function to run script in live mode
run_live_mode() {
	log "Running main.py in LIVE mode"
	cd "$SCRIPT_DIR"
	nohup "$PYTHON_ENV/bin/python" "$MAIN_SCRIPT" --mode live --gpu >"$LOG_DIR/live_$CURRENT_DATE.log" 2>&1 &
	log "Live mode started with PID: $!"
}

# Function to run script in training mode
run_training_mode() {
	log "Running main.py in TRAINING mode with 2500 epochs"
	cd "$SCRIPT_DIR"
	nohup "$PYTHON_ENV/bin/python" "$MAIN_SCRIPT" --mode train --gpu --epochs 2500 >"$LOG_DIR/training_$CURRENT_DATE.log" 2>&1 &
	log "Training mode started with PID: $!"
}

# Function to run script in optimization mode
run_optimization_mode() {
	log "Running main.py in OPTIMIZATION mode"
	cd "$SCRIPT_DIR"
	nohup "$PYTHON_ENV/bin/python" "$MAIN_SCRIPT" --mode optimize --gpu >"$LOG_DIR/optimization_$CURRENT_DATE.log" 2>&1 &
	log "Optimization mode started with PID: $!"
}

# Setup cron jobs for this script
setup_cron_jobs() {
	log "Setting up cron jobs"

	# Remove existing cron jobs related to this script to prevent duplicates
	crontab -l 2>/dev/null | grep -v "market_automation.sh" | crontab -

	# Create a temporary file for the current crontab
	TEMP_CRON=$(mktemp)
	crontab -l >"$TEMP_CRON" 2>/dev/null

	log "Adding new cron jobs"
	echo "# Market script - run every weekday at 9am" >>"$TEMP_CRON"
	echo "0 9 * * 1-5 $SCRIPT_DIR/market_automation.sh live" >>"$TEMP_CRON"
	echo "# Market script - run Friday at 4:30pm for training" >>"$TEMP_CRON"
	echo "30 16 * * 5 $SCRIPT_DIR/market_automation.sh training" >>"$TEMP_CRON"
	echo "0 22 * * 6 $SCRIPT_DIR/market_automation.sh optimize" >>"$TEMP_CRON"
	echo "# Market script - run 5 minutes after reboot" >>"$TEMP_CRON"
	echo "@reboot sleep 300 && $SCRIPT_DIR/market_automation.sh reboot" >>"$TEMP_CRON"

	# Install new crontab
	crontab "$TEMP_CRON"
	log "Cron jobs installed successfully"

	# Remove the temporary file
	rm "$TEMP_CRON"
}

# Check if this is being run from cron or manually
if [ -z "$1" ]; then
	log "Script started without arguments, determining mode based on time"

	# Get current day of week (1-7, with 1 being Monday)
	DAY_OF_WEEK=$(date +"%u")

	# Get current hour and minute
	HOUR=$(date +"%H")
	MINUTE=$(date +"%M")

	# Check if it's Friday afternoon (day 5) after 4:30pm
	if [ "$DAY_OF_WEEK" -eq 5 ] && [ "$HOUR" -ge 16 ] && [ "$MINUTE" -ge 30 ]; then
		MODE="training"
	else
		# For all other weekdays at any time, default to live mode
		MODE="live"
	fi
else
	# If argument provided, use that as mode
	MODE="$1"
fi

# Check if we're running after reboot
if [ "$MODE" = "reboot" ]; then
	log "Running after system reboot"
	# Check market hours before deciding what mode to run
	DAY_OF_WEEK=$(date +"%u")
	HOUR=$(date +"%H")
	MINUTE=$(date +"%M")

	# Basic check for weekdays and market hours (9:30 AM to 4:00 PM ET)
	# This does NOT account for holidays or early closures.
	if [ "$DAY_OF_WEEK" -ge 1 ] && [ "$DAY_OF_WEEK" -le 5 ]; then # Monday to Friday
		if [ "$HOUR" -gt 9 ] && [ "$HOUR" -lt 16 ]; then # Between 10 AM and 3 PM
			log "Reboot during market hours, starting in live mode"
			MODE="live"
		elif [ "$HOUR" -eq 9 ] && [ "$MINUTE" -ge 30 ]; then # 9:30 AM onwards
			log "Reboot during market hours, starting in live mode"
			MODE="live"
		elif [ "$HOUR" -eq 16 ] && [ "$MINUTE" -eq 0 ]; then # Exactly 4:00 PM
			log "Reboot at market close, no action needed"
			exit 0
		else
			log "Reboot outside market hours, no action needed"
			exit 0
		fi
	else
		log "Reboot on weekend, no action needed"
		exit 0
	fi
fi

# Execute based on mode
case "$MODE" in
"live")
	log "=== STARTING DAILY LIVE MODE SCRIPT ==="
	kill_existing_process

	run_live_mode
	log "=== DAILY LIVE MODE SCRIPT COMPLETE ==="
	;;
"training")
	log "=== STARTING FRIDAY TRAINING MODE SCRIPT ==="
	kill_existing_process

	run_training_mode
	log "=== FRIDAY TRAINING MODE SCRIPT COMPLETE ==="
	;;
"optimize")
	log "=== STARTING SATURDAY OPTIMIZATION SCRIPT ==="
	kill_existing_process

	run_optimization_mode
	log "=== SATURDAY OPTIMIZATION SCRIPT COMPLETE ==="
	;;
"install")
	log "=== INSTALLING CRON JOBS ==="
	setup_cron_jobs
	log "=== INSTALLATION COMPLETE ==="
	;;
"reboot")
	log "=== HANDLING POST-REBOOT EXECUTION ==="
	# The reboot mode logic is already handled above
	# This case is just for completeness
	;;
*)
	log "Invalid mode: $MODE. Valid modes are 'live', 'training', 'install', or 'reboot'"
	exit 1
	;;
esac
