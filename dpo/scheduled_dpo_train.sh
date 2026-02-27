#!/bin/bash
# =============================================================================
# Scheduled DPO Training Script
#
# Automatically start/stop DPO training within configured time windows.
# Uses Ctrl+C (SIGINT) to gracefully stop training at end time,
# and resumes from checkpoint at next start time.
#
# Usage:
#   chmod +x scheduled_dpo_train.sh
#   ./scheduled_dpo_train.sh           # Start the scheduler
#   ./scheduled_dpo_train.sh pause     # Pause: stop current training and hold
#   ./scheduled_dpo_train.sh resume    # Resume: continue scheduled training
#   ./scheduled_dpo_train.sh status    # Show current scheduler status
#   ./scheduled_dpo_train.sh stop      # Fully stop the scheduler
# =============================================================================

# ========================== Configuration ==========================

# --- Schedule Configuration ---
# Format: HH:MM (24-hour, Beijing time)
START_TIME="9:00"
END_TIME="21:00"

# --- Training Configuration ---
TRAIN_SCRIPT="./dpo_train.py"           # DPO training script path (relative to this script)
PYTHON_BIN="python"                     # Python binary (or full path)
# Checkpoint dir: DPOTrainer saves checkpoints here (must match output_dir in DpoConfig)
CHECKPOINT_DIR="../model_save/dpo"

# --- Advanced Configuration ---
POLL_INTERVAL=60                        # Seconds between time checks when waiting
GRACE_PERIOD=300                        # Seconds to wait after SIGINT before SIGTERM (给 Trainer 足够时间保存 checkpoint)
CONDA_ENV="chatlm"                      # Conda environment name (leave empty to skip)
TIMEZONE="Asia/Shanghai"                # Timezone for schedule

# ========================== End Configuration ==========================

export TZ="${TIMEZONE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCHEDULE_LOG="${LOG_DIR}/scheduled_dpo_train.log"
TRAIN_PID=""
PAUSE_FILE="${LOG_DIR}/.dpo_scheduler_paused"
PID_FILE="${LOG_DIR}/.dpo_scheduler_pid"

# --- Logging ---
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "${msg}"
    echo "${msg}" >> "${SCHEDULE_LOG}"
}

# --- Time Utilities ---
get_current_minutes() {
    local h=$(date +%H); local m=$(date +%M)
    echo $(( 10#$h * 60 + 10#$m ))
}

time_to_minutes() {
    local h="${1%%:*}"; local m="${1##*:}"
    echo $(( 10#$h * 60 + 10#$m ))
}

is_within_window() {
    local now; now=$(get_current_minutes)
    local start; start=$(time_to_minutes "${START_TIME}")
    local end;   end=$(time_to_minutes "${END_TIME}")
    if [ "${start}" -le "${end}" ]; then
        [ "${now}" -ge "${start}" ] && [ "${now}" -lt "${end}" ]
    else
        [ "${now}" -ge "${start}" ] || [ "${now}" -lt "${end}" ]
    fi
}

seconds_until_start() {
    local now; now=$(get_current_minutes)
    local start; start=$(time_to_minutes "${START_TIME}")
    local diff=$(( start - now ))
    [ "${diff}" -le 0 ] && diff=$(( diff + 1440 ))
    echo $(( diff * 60 ))
}

# --- Training Process Management ---
start_training() {
    local train_log="${LOG_DIR}/dpo_train_$(date '+%Y%m%d_%H%M%S').log"

    # 断点续训：检测 checkpoint 目录下是否存在 checkpoint-* 子目录
    local resume_arg=""
    local abs_checkpoint_dir
    abs_checkpoint_dir="$(cd "${SCRIPT_DIR}" && cd "${CHECKPOINT_DIR}" 2>/dev/null && pwd)"
    # 找最新的 checkpoint-* 目录
    local latest_ckpt
    latest_ckpt=$(ls -dt "${abs_checkpoint_dir}"/checkpoint-* 2>/dev/null | head -n 1)
    if [ -n "${latest_ckpt}" ] && [ -d "${latest_ckpt}" ]; then
        resume_arg="--resume_from_checkpoint=${latest_ckpt}"
        log "Checkpoint found: ${latest_ckpt}, will resume from it."
    else
        log "No checkpoint found in ${abs_checkpoint_dir}, starting fresh training."
    fi

    log "Starting DPO training (log: ${train_log})"
    log "Command: ${PYTHON_BIN} ${TRAIN_SCRIPT} ${resume_arg}"
    log "To follow training log: tail -f ${train_log}"

    # 激活 conda 环境
    if [ -n "${CONDA_ENV}" ] && command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}" 2>/dev/null
    fi

    # 禁用 tokenizers 并行警告；过滤 \r 进度条字符避免日志混乱
    export TOKENIZERS_PARALLELISM=false

    cd "${SCRIPT_DIR}"
    # 用临时 fifo 管道 + python 过滤器，分离进度条与训练指标，让日志清晰可读
    local fifo="${LOG_DIR}/.dpo_train_fifo_$$"
    mkfifo "${fifo}"

    # Python 过滤器：
    #   - 进度条行（含 it/s 或 %| 的行）：只在终端显示，不写入日志文件
    #   - 训练指标行（含 'loss' 的 JSON 行）：格式化后写入日志，并在终端高亮显示
    #   - 其他行（启动信息、警告等）：同时写入日志和终端
    python3 -u - "${train_log}" < "${fifo}" << 'PYEOF' &
import sys, re, os

log_path = sys.argv[1]
# 进度条行特征：含 it/s 或 %| 或纯空白
progress_pat = re.compile(r'it/s|s/it|\d+%\||\|\s*\d+/\d+')
# 训练指标行特征：含 'loss' 关键字（HuggingFace Trainer 输出的 JSON dict）
metric_pat   = re.compile(r"'loss'|\"loss\"")

with open(log_path, 'a', buffering=1) as f:
    for raw in sys.stdin:
        # 去除 \r 及首尾空白
        line = raw.replace('\r', '').rstrip()
        if not line:
            continue

        if progress_pat.search(line):
            # 进度条：只打印到终端，不写日志
            print(line, flush=True)
        elif metric_pat.search(line):
            # 训练指标：格式化输出，写日志 + 终端高亮
            try:
                import ast
                d = ast.literal_eval(line.strip())
                fmt = (
                    f"[METRICS] step={d.get('step','?'):>5} | "
                    f"epoch={d.get('epoch',0):.3f} | "
                    f"loss={d.get('loss',0):.4f} | "
                    f"acc={d.get('rewards/accuracies',0):.4f} | "
                    f"margin={d.get('rewards/margins',0):.4f} | "
                    f"lr={d.get('learning_rate',0):.2e} | "
                    f"grad={d.get('grad_norm',0):.2f}"
                )
            except Exception:
                fmt = f"[METRICS] {line}"
            f.write(fmt + '\n')
            f.flush()
            print(f"\033[32m{fmt}\033[0m", flush=True)
        else:
            # 普通信息行：写日志 + 终端
            f.write(line + '\n')
            f.flush()
            print(line, flush=True)
PYEOF

    setsid ${PYTHON_BIN} -u ${TRAIN_SCRIPT} ${resume_arg} > "${fifo}" 2>&1 &
    TRAIN_PID=$!
    rm -f "${fifo}"
    log "Training started with PID: ${TRAIN_PID}"
}

kill_tree() {
    local pid=$1; local sig=$2
    local children; children=$(pgrep -P "${pid}" 2>/dev/null)
    for child in ${children}; do kill_tree "${child}" "${sig}"; done
    kill -"${sig}" "${pid}" 2>/dev/null
}

stop_training() {
    if [ -n "${TRAIN_PID}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        log "Stopping DPO training (PID: ${TRAIN_PID}) with SIGINT (waiting for checkpoint save)..."
        # 只发给 Python 进程本身，让 HuggingFace Trainer 捕获 SIGINT 并自动保存 checkpoint
        # 不要用 kill -INT -PID（进程组信号），否则 Trainer 内部子进程也会被打断，无法完成保存
        kill -INT "${TRAIN_PID}" 2>/dev/null

        # 等待 Trainer 保存 checkpoint（保存大模型可能需要较长时间，给 300s）
        local waited=0
        local save_timeout=300
        log "Waiting up to ${save_timeout}s for checkpoint to be saved..."
        while kill -0 "${TRAIN_PID}" 2>/dev/null && [ "${waited}" -lt "${save_timeout}" ]; do
            sleep 2; waited=$((waited + 2))
            # 每 30s 打印一次等待进度
            [ $((waited % 30)) -eq 0 ] && log "Still waiting for training to stop... (${waited}s elapsed)"
        done

        if kill -0 "${TRAIN_PID}" 2>/dev/null; then
            log "Not stopped after ${save_timeout}s, sending SIGTERM..."
            kill -TERM "${TRAIN_PID}" 2>/dev/null
            kill_tree "${TRAIN_PID}" TERM 2>/dev/null
            sleep 10
            if kill -0 "${TRAIN_PID}" 2>/dev/null; then
                log "Force killing (SIGKILL)..."
                kill -9 "${TRAIN_PID}" 2>/dev/null
                kill_tree "${TRAIN_PID}" 9 2>/dev/null
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

# --- Pause/Resume ---
is_paused() { [ -f "${PAUSE_FILE}" ]; }

do_pause() {
    if ! is_paused; then
        touch "${PAUSE_FILE}"
        log "Scheduler PAUSED."
        is_training_running && stop_training
    else
        log "Already paused."
    fi
}

do_resume() {
    if is_paused; then
        rm -f "${PAUSE_FILE}"
        log "Scheduler RESUMED."
    else
        log "Not paused."
    fi
}

toggle_pause() {
    if is_paused; then do_resume; else do_pause; fi
}
trap toggle_pause SIGUSR1

# --- Signal Handlers ---
cleanup() {
    log "Received termination signal, cleaning up..."
    stop_training
    rm -f "${PID_FILE}" "${PAUSE_FILE}"
    log "Scheduler exiting."
    exit 0
}
trap cleanup SIGINT SIGTERM SIGHUP

# --- CLI Subcommands ---
handle_subcommand() {
    local cmd="$1"
    local running_pid=""
    if [ -f "${PID_FILE}" ]; then
        running_pid=$(cat "${PID_FILE}")
        kill -0 "${running_pid}" 2>/dev/null || { running_pid=""; rm -f "${PID_FILE}"; }
    fi

    case "${cmd}" in
        pause)
            [ -z "${running_pid}" ] && { echo "No running scheduler found."; exit 1; }
            [ -f "${PAUSE_FILE}" ] && { echo "Already paused."; exit 0; }
            kill -USR1 "${running_pid}" 2>/dev/null
            echo "Pause signal sent to scheduler (PID: ${running_pid})."
            ;;
        resume)
            [ -z "${running_pid}" ] && { echo "No running scheduler found."; exit 1; }
            [ ! -f "${PAUSE_FILE}" ] && { echo "Not paused."; exit 0; }
            kill -USR1 "${running_pid}" 2>/dev/null
            echo "Resume signal sent to scheduler (PID: ${running_pid})."
            ;;
        status)
            if [ -z "${running_pid}" ]; then
                echo "Scheduler: NOT RUNNING"
            else
                echo "Scheduler: RUNNING (PID: ${running_pid})"
                [ -f "${PAUSE_FILE}" ] && echo "State:     PAUSED" || echo "State:     ACTIVE"
            fi
            echo "Window:    ${START_TIME} - ${END_TIME} (${TIMEZONE})"
            echo "Checkpoint dir: ${CHECKPOINT_DIR}"
            if [ -f "${SCHEDULE_LOG}" ]; then
                echo ""
                echo "--- Recent log (last 10 lines) ---"
                tail -n 10 "${SCHEDULE_LOG}"
            fi
            ;;
        stop)
            [ -z "${running_pid}" ] && { echo "No running scheduler found."; exit 1; }
            kill -TERM "${running_pid}" 2>/dev/null
            echo "Stop signal sent to scheduler (PID: ${running_pid})."
            ;;
        *)
            echo "Unknown command: ${cmd}"
            echo "Usage: $0 [pause|resume|status|stop]"
            exit 1
            ;;
    esac
    exit 0
}

if [ $# -gt 0 ]; then
    handle_subcommand "$1"
fi

# --- Main Loop ---
main() {
    echo $$ > "${PID_FILE}"
    rm -f "${PAUSE_FILE}"

    log "=========================================="
    log "Scheduled DPO Training Scheduler Started"
    log "=========================================="
    log "Training window : ${START_TIME} - ${END_TIME} (${TIMEZONE})"
    log "Train script    : ${TRAIN_SCRIPT}"
    log "Checkpoint dir  : ${CHECKPOINT_DIR}"
    log "Poll interval   : ${POLL_INTERVAL}s"
    log "Controls        : $0 pause|resume|status|stop"
    log "=========================================="

    while true; do
        if is_paused; then
            is_training_running && stop_training
            while is_paused; do sleep "${POLL_INTERVAL}"; done
            log "Scheduler resumed, continuing..."
            continue
        fi

        if is_within_window; then
            if ! is_training_running; then
                log "Within training window [${START_TIME} - ${END_TIME}], starting DPO training..."
                start_training
                sleep 5
            fi

            while is_within_window && is_training_running && ! is_paused; do
                sleep "${POLL_INTERVAL}"
            done

            is_paused && continue

            if is_training_running && ! is_within_window; then
                log "Training window ended, stopping training..."
                stop_training
                log "Will resume at next window start: ${START_TIME}"
            elif ! is_training_running && is_within_window; then
                log "Training process exited on its own (all epochs completed or error)."
                log "Waiting for next training window..."
                while is_within_window && ! is_paused; do sleep "${POLL_INTERVAL}"; done
            fi
        else
            local secs; secs=$(seconds_until_start)
            local hrs=$(( secs / 3600 )); local mins=$(( (secs % 3600) / 60 ))
            log "Outside training window. Next start in ~${hrs}h ${mins}m (at ${START_TIME})"

            local slept=0
            while [ "${slept}" -lt "${secs}" ] && ! is_within_window && ! is_paused; do
                local chunk="${POLL_INTERVAL}"
                [ $((secs - slept)) -lt "${chunk}" ] && chunk=$((secs - slept))
                sleep "${chunk}"
                slept=$(( slept + chunk ))
            done
        fi
    done
}

main
