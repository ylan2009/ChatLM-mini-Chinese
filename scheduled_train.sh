#!/bin/bash
# =============================================================================
# Scheduled Training Script
# 
# Automatically start/stop training within configured time windows.
# Uses Ctrl+C (SIGINT) to gracefully stop training at end time,
# and resumes from checkpoint at next start time.
#
# Usage:
#   chmod +x scheduled_train.sh
#   ./scheduled_train.sh              # Start the scheduler
#   ./scheduled_train.sh pause        # Pause: stop current training and hold
#   ./scheduled_train.sh resume       # Resume: continue scheduled training
#   ./scheduled_train.sh status       # Show current scheduler status
#   ./scheduled_train.sh stop         # Fully stop the scheduler
#
# Configuration:
#   Edit the variables below to set your schedule and training parameters.
# =============================================================================

# ========================== Configuration ==========================

# --- Schedule Configuration ---
# Format: HH:MM (24-hour, Beijing time)
# The script will start training at START_TIME and stop at END_TIME each day.
# Supports overnight windows, e.g. START_TIME="22:00" END_TIME="08:00"
START_TIME="10:48"
END_TIME="20:00"

# --- Training Configuration ---
NUM_PROCESSES=3                          # Number of GPUs
TRAIN_SCRIPT="./train.py"               # Training script path
TRAIN_ARGS="train"                       # Base training arguments (--is_keep_training is auto-detected)
STATE_DIR="./model_save/train_latest_state"  # Checkpoint state directory (used to auto-detect resume)

# --- Advanced Configuration ---
POLL_INTERVAL=60                         # Seconds between time checks when waiting
GRACE_PERIOD=30                          # Seconds to wait after SIGINT before SIGTERM
CONDA_ENV="chatlm"                       # Conda environment name (leave empty to skip activation)
TIMEZONE="Asia/Shanghai"                 # Timezone for schedule (Beijing time)

# ========================== End Configuration ==========================

# Export timezone
export TZ="${TIMEZONE}"

# Script directory (for relative paths)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCHEDULE_LOG="${LOG_DIR}/scheduled_train.log"
TRAIN_PID=""

# Control files for pause/resume
PAUSE_FILE="${LOG_DIR}/.scheduler_paused"
PID_FILE="${LOG_DIR}/.scheduler_pid"

# --- Logging ---
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "${msg}"
    echo "${msg}" >> "${SCHEDULE_LOG}"
}

# --- Time Utilities ---
# Get current time in minutes since midnight
get_current_minutes() {
    echo $(( $(date +%H) * 60 + $(date +%M) ))
}

# Convert HH:MM to minutes since midnight
time_to_minutes() {
    local h="${1%%:*}"
    local m="${1##*:}"
    # Remove leading zeros to avoid octal interpretation
    h=$((10#$h))
    m=$((10#$m))
    echo $(( h * 60 + m ))
}

# Check if current time is within the training window
is_within_window() {
    local now
    now=$(get_current_minutes)
    local start
    start=$(time_to_minutes "${START_TIME}")
    local end
    end=$(time_to_minutes "${END_TIME}")

    if [ "${start}" -le "${end}" ]; then
        # Same-day window: e.g. 08:00 - 18:00
        [ "${now}" -ge "${start}" ] && [ "${now}" -lt "${end}" ]
    else
        # Overnight window: e.g. 22:00 - 08:00
        [ "${now}" -ge "${start}" ] || [ "${now}" -lt "${end}" ]
    fi
}

# Calculate seconds until next start time
seconds_until_start() {
    local now
    now=$(get_current_minutes)
    local start
    start=$(time_to_minutes "${START_TIME}")

    local diff=$(( start - now ))
    if [ "${diff}" -le 0 ]; then
        diff=$(( diff + 1440 ))  # Add 24 hours
    fi
    echo $(( diff * 60 ))
}

# Calculate seconds until end time
seconds_until_end() {
    local now
    now=$(get_current_minutes)
    local end
    end=$(time_to_minutes "${END_TIME}")

    local diff=$(( end - now ))
    if [ "${diff}" -le 0 ]; then
        diff=$(( diff + 1440 ))  # Add 24 hours
    fi
    echo $(( diff * 60 ))
}

# --- Training Process Management ---
start_training() {
    local train_log="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"

    # Auto-detect checkpoint: if state directory exists and is not empty, resume from it
    local actual_args="${TRAIN_ARGS}"
    local abs_state_dir="${SCRIPT_DIR}/${STATE_DIR#./}"
    if [ -d "${abs_state_dir}" ] && [ "$(ls -A "${abs_state_dir}" 2>/dev/null)" ]; then
        actual_args="${TRAIN_ARGS} --is_keep_training=True"
        log "Checkpoint found in ${abs_state_dir}, resuming from checkpoint."
    else
        log "No checkpoint found in ${abs_state_dir}, starting fresh training."
    fi

    log "Starting training (log: ${train_log})"
    log "Command: accelerate launch --multi_gpu --num_processes ${NUM_PROCESSES} ${TRAIN_SCRIPT} ${actual_args}"

    # Activate conda environment if specified
    local cmd="accelerate launch --multi_gpu --num_processes ${NUM_PROCESSES} ${TRAIN_SCRIPT} ${actual_args}"
    if [ -n "${CONDA_ENV}" ]; then
        # Try to activate conda
        if command -v conda &>/dev/null; then
            eval "$(conda shell.bash hook)"
            conda activate "${CONDA_ENV}" 2>/dev/null
        fi
    fi

    # Launch training in background, redirect output to log
    cd "${SCRIPT_DIR}"
    ${cmd} >> "${train_log}" 2>&1 &
    TRAIN_PID=$!

    log "Training started with PID: ${TRAIN_PID}"
}

stop_training() {
    if [ -n "${TRAIN_PID}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        log "Stopping training (PID: ${TRAIN_PID}) with SIGINT (graceful shutdown)..."
        
        # Send SIGINT to the entire process group for multi-GPU training
        kill -INT -"${TRAIN_PID}" 2>/dev/null || kill -INT "${TRAIN_PID}" 2>/dev/null
        
        # Wait for graceful shutdown
        local waited=0
        while kill -0 "${TRAIN_PID}" 2>/dev/null && [ "${waited}" -lt "${GRACE_PERIOD}" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        
        if kill -0 "${TRAIN_PID}" 2>/dev/null; then
            log "Training did not stop gracefully after ${GRACE_PERIOD}s, sending SIGTERM..."
            kill -TERM -"${TRAIN_PID}" 2>/dev/null || kill -TERM "${TRAIN_PID}" 2>/dev/null
            sleep 5
            
            if kill -0 "${TRAIN_PID}" 2>/dev/null; then
                log "Force killing training process..."
                kill -9 -"${TRAIN_PID}" 2>/dev/null || kill -9 "${TRAIN_PID}" 2>/dev/null
            fi
        fi
        
        wait "${TRAIN_PID}" 2>/dev/null
        log "Training stopped."
        TRAIN_PID=""
    fi
}

is_training_running() {
    [ -n "${TRAIN_PID}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null
}

# --- Pause/Resume Control ---
is_paused() {
    [ -f "${PAUSE_FILE}" ]
}

do_pause() {
    if ! is_paused; then
        touch "${PAUSE_FILE}"
        log "Scheduler PAUSED by user command. Training will be stopped and held."
        # If training is running, stop it gracefully
        if is_training_running; then
            stop_training
        fi
    else
        log "Scheduler is already paused."
    fi
}

do_resume() {
    if is_paused; then
        rm -f "${PAUSE_FILE}"
        log "Scheduler RESUMED by user command. Training will restart in next window check."
    else
        log "Scheduler is not paused."
    fi
}

# Handle SIGUSR1 as toggle pause/resume
toggle_pause() {
    if is_paused; then
        do_resume
    else
        do_pause
    fi
}

trap toggle_pause SIGUSR1

# --- Signal Handlers ---
cleanup() {
    log "Received termination signal, cleaning up..."
    stop_training
    rm -f "${PID_FILE}"
    rm -f "${PAUSE_FILE}"
    log "Scheduler exiting."
    exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP

# --- CLI Subcommands ---
# Handle pause/resume/status/stop commands sent to a running scheduler
handle_subcommand() {
    local cmd="$1"
    local running_pid=""

    if [ -f "${PID_FILE}" ]; then
        running_pid=$(cat "${PID_FILE}")
        if ! kill -0 "${running_pid}" 2>/dev/null; then
            running_pid=""
            rm -f "${PID_FILE}"
        fi
    fi

    case "${cmd}" in
        pause)
            if [ -z "${running_pid}" ]; then
                echo "No running scheduler found."
                exit 1
            fi
            touch "${PAUSE_FILE}"
            kill -USR1 "${running_pid}" 2>/dev/null
            echo "Pause signal sent to scheduler (PID: ${running_pid})."
            echo "Training will be stopped and scheduler will hold until resumed."
            ;;
        resume)
            if [ -z "${running_pid}" ]; then
                echo "No running scheduler found."
                exit 1
            fi
            rm -f "${PAUSE_FILE}"
            kill -USR1 "${running_pid}" 2>/dev/null
            echo "Resume signal sent to scheduler (PID: ${running_pid})."
            echo "Training will restart at next schedule check."
            ;;
        status)
            if [ -z "${running_pid}" ]; then
                echo "Scheduler: NOT RUNNING"
            else
                echo "Scheduler: RUNNING (PID: ${running_pid})"
                if [ -f "${PAUSE_FILE}" ]; then
                    echo "State:     PAUSED"
                else
                    echo "State:     ACTIVE"
                fi
            fi
            echo "Window:    ${START_TIME} - ${END_TIME} (${TIMEZONE})"
            # Show recent log
            if [ -f "${SCHEDULE_LOG}" ]; then
                echo ""
                echo "--- Recent log (last 10 lines) ---"
                tail -n 10 "${SCHEDULE_LOG}"
            fi
            ;;
        stop)
            if [ -z "${running_pid}" ]; then
                echo "No running scheduler found."
                exit 1
            fi
            echo "Sending stop signal to scheduler (PID: ${running_pid})..."
            kill -TERM "${running_pid}" 2>/dev/null
            echo "Stop signal sent. Scheduler will clean up and exit."
            ;;
        *)
            echo "Unknown command: ${cmd}"
            echo "Usage: $0 [pause|resume|status|stop]"
            exit 1
            ;;
    esac
    exit 0
}

# If a subcommand is given, handle it and exit
if [ $# -gt 0 ]; then
    handle_subcommand "$1"
fi

# --- Main Loop ---
main() {
    # Save our PID for subcommands
    echo $$ > "${PID_FILE}"
    # Clean up any stale pause file from previous run
    rm -f "${PAUSE_FILE}"

    log "=========================================="
    log "Scheduled Training Scheduler Started"
    log "=========================================="
    log "Training window: ${START_TIME} - ${END_TIME} (${TIMEZONE})"
    log "GPUs: ${NUM_PROCESSES}"
    log "Train command: accelerate launch --multi_gpu --num_processes ${NUM_PROCESSES} ${TRAIN_SCRIPT} ${TRAIN_ARGS} (--is_keep_training auto-detected)"
    log "Checkpoint state dir: ${STATE_DIR}"
    log "Poll interval: ${POLL_INTERVAL}s"
    log "Controls: $0 pause|resume|status|stop"
    log "=========================================="

    while true; do
        # Check if paused
        if is_paused; then
            if is_training_running; then
                log "Scheduler is paused, stopping current training..."
                stop_training
            fi
            # Wait until resumed
            while is_paused; do
                sleep "${POLL_INTERVAL}"
            done
            log "Scheduler resumed, continuing normal operation..."
            continue
        fi

        if is_within_window; then
            if ! is_training_running; then
                log "Within training window [${START_TIME} - ${END_TIME}], starting training..."
                start_training
                sleep 5  # Brief pause to let process start
            fi

            # Monitor: wait until either training ends or window expires
            while is_within_window && is_training_running && ! is_paused; do
                sleep "${POLL_INTERVAL}"
            done

            # If paused during training, let the pause block handle it
            if is_paused; then
                continue
            fi

            if is_training_running && ! is_within_window; then
                # Window expired, stop training
                log "Training window [${START_TIME} - ${END_TIME}] ended, stopping training..."
                stop_training
                log "Training will resume at next window start: ${START_TIME}"
            elif ! is_training_running && is_within_window; then
                # Training finished on its own within the window
                log "Training process exited on its own (possibly completed all epochs)."
                log "Waiting for next training window..."
                # Wait until we're outside the window, then continue the main loop
                while is_within_window && ! is_paused; do
                    sleep "${POLL_INTERVAL}"
                done
            fi
        else
            # Outside training window
            local secs
            secs=$(seconds_until_start)
            local hrs=$((secs / 3600))
            local mins=$(( (secs % 3600) / 60 ))
            log "Outside training window. Next start in ~${hrs}h ${mins}m (at ${START_TIME})"

            # Sleep in intervals to stay responsive to signals
            local slept=0
            while [ "${slept}" -lt "${secs}" ] && ! is_within_window && ! is_paused; do
                local chunk="${POLL_INTERVAL}"
                if [ $((secs - slept)) -lt "${chunk}" ]; then
                    chunk=$((secs - slept))
                fi
                sleep "${chunk}"
                slept=$((slept + chunk))
            done
        fi
    done
}

main
