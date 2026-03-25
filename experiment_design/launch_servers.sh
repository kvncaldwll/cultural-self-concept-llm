#!/usr/bin/env bash
# launch_servers.sh
# -----------------
# Launch four vLLM OpenAI-compatible servers, one per model.
# Designed for Lambda Cloud A100-80GB single-node.
#
# Prerequisites:
#   pip install vllm>=0.4.0
#   huggingface-cli login  (for gated models: Llama-3.1)
#
# Usage:
#   chmod +x launch_servers.sh
#   ./launch_servers.sh           # launch all four servers
#   ./launch_servers.sh qwen      # launch only Qwen
#   ./launch_servers.sh --stop    # kill all managed servers
#
# Logs are written to logs/<model_name>.log
# PIDs are tracked in .vllm_pids for clean shutdown.

set -euo pipefail

LOG_DIR="logs"
PID_FILE=".vllm_pids"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------
# Model definitions: NAME|MODEL_ID|PORT|TENSOR_PARALLEL|DTYPE
# -----------------------------------------------------------------------
declare -a MODELS=(
    "qwen|Qwen/Qwen2.5-7B-Instruct|8001|1|bfloat16"
    "llama|meta-llama/Llama-3.1-8B-Instruct|8002|1|bfloat16"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|8003|1|bfloat16"
    "yi|01-ai/Yi-1.5-9B-Chat|8004|1|bfloat16"
)

# -----------------------------------------------------------------------
# Stop function
# -----------------------------------------------------------------------
stop_servers() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[INFO] No PID file found. Nothing to stop."
        return 0
    fi
    echo "[INFO] Stopping vLLM servers..."
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[INFO]   Killing PID $pid"
            kill "$pid"
        else
            echo "[INFO]   PID $pid already dead"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "[INFO] All servers stopped."
}

# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
wait_for_server() {
    local name="$1"
    local port="$2"
    local max_wait=180  # seconds
    local elapsed=0
    local interval=5

    echo "[INFO] Waiting for $name on port $port..."
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if [[ "$elapsed" -ge "$max_wait" ]]; then
            echo "[ERROR] $name failed to become healthy after ${max_wait}s"
            echo "[INFO]  Check logs/${name}.log for details"
            return 1
        fi
        echo "[INFO]   Still waiting... (${elapsed}s/${max_wait}s)"
    done
    echo "[OK] $name is healthy on port $port"
}

# -----------------------------------------------------------------------
# Launch function
# -----------------------------------------------------------------------
launch_model() {
    local name="$1"
    local model_id="$2"
    local port="$3"
    local tp="$4"
    local dtype="$5"

    echo ""
    echo "[INFO] ============================================================"
    echo "[INFO] Launching: $name"
    echo "[INFO]   Model:  $model_id"
    echo "[INFO]   Port:   $port"
    echo "[INFO]   TP:     $tp"
    echo "[INFO]   dtype:  $dtype"
    echo "[INFO] ============================================================"

    # Check if port already in use
    if lsof -i ":${port}" > /dev/null 2>&1; then
        echo "[WARN] Port $port is already in use. Skipping $name."
        return 0
    fi

    python -m vllm.entrypoints.openai.api_server \
        --model "$model_id" \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --dtype "$dtype" \
        --max-model-len 4096 \
        --disable-log-requests \
        --trust-remote-code \
        > "${LOG_DIR}/${name}.log" 2>&1 &

    local pid=$!
    echo "$pid" >> "$PID_FILE"
    echo "[INFO] Launched $name with PID $pid"
}

# -----------------------------------------------------------------------
# GPU memory check
# -----------------------------------------------------------------------
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        local total_mb
        total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        local total_gb=$((total_mb / 1024))
        echo "[INFO] GPU memory: ~${total_gb}GB detected"
        if [[ "$total_gb" -lt 60 ]]; then
            echo "[WARN] Less than 60GB VRAM detected. Running all 4 models simultaneously"
            echo "[WARN] may cause OOM. Consider running models sequentially:"
            echo "[WARN]   ./launch_servers.sh qwen && python run_experiment.py ... --models Qwen2.5-7B"
        fi
    else
        echo "[WARN] nvidia-smi not found — cannot verify GPU memory"
    fi
}

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
# Handle flags
if [[ "${1:-}" == "--stop" ]]; then
    stop_servers
    exit 0
fi

# Filter by name argument if provided
FILTER="${1:-}"

check_gpu_memory

# Clear PID file
> "$PID_FILE"

# Launch
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model_id port tp dtype <<< "$entry"

    if [[ -n "$FILTER" && "$name" != "$FILTER" ]]; then
        echo "[INFO] Skipping $name (filter: $FILTER)"
        continue
    fi

    launch_model "$name" "$model_id" "$port" "$tp" "$dtype"

    # Stagger launches to avoid simultaneous model weight downloads
    sleep 5
done

echo ""
echo "[INFO] All requested servers launched. Waiting for health checks..."
echo ""

# Health checks
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model_id port tp dtype <<< "$entry"

    if [[ -n "$FILTER" && "$name" != "$FILTER" ]]; then
        continue
    fi

    wait_for_server "$name" "$port" || true
done

echo ""
echo "[INFO] ============================================================"
echo "[INFO] Server status summary:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name model_id port tp dtype <<< "$entry"
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "[OK]   $name  (port $port)"
    else
        echo "[DOWN] $name  (port $port)  — check ${LOG_DIR}/${name}.log"
    fi
done
echo "[INFO] ============================================================"
echo ""
echo "[INFO] To stop all servers:  ./launch_servers.sh --stop"
echo "[INFO] To run experiment:    python run_experiment.py --config config.yaml"
echo ""
