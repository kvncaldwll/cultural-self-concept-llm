#!/usr/bin/env bash
# run_language_matrix.sh
# Runs Chinese and French prompts on all 4 models
# (English already done, Chinese on Eastern already done)
# New runs needed:
#   Chinese: Llama, Mistral
#   French: all 4
set -euo pipefail

declare -a ALL=(
    "qwen|Qwen/Qwen2.5-7B-Instruct|8001|Qwen2.5-7B"
    "llama|meta-llama/Llama-3.1-8B-Instruct|8002|Llama-3.1-8B"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|8003|Mistral-7B-v0.3"
    "yi|01-ai/Yi-1.5-9B-Chat|8004|Yi-1.5-9B"
)

# Only need Chinese for Llama and Mistral (Qwen and Yi already done in round 3)
declare -a WESTERN_ONLY=(
    "llama|meta-llama/Llama-3.1-8B-Instruct|8002|Llama-3.1-8B"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|8003|Mistral-7B-v0.3"
)

launch() {
    echo ""; echo "=== Launching $1 ==="
    sudo docker run -d --gpus all --name "vllm_$1" \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p "$3:8000" -e HF_TOKEN="$HF_TOKEN" \
        vllm/vllm-openai:latest \
        --model "$2" --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85
    for i in $(seq 1 60); do curl -sf "http://localhost:$3/health" >/dev/null 2>&1 && echo "  Ready!" && return; sleep 5; done
    echo "  FAILED"; return 1
}
stop() { sudo docker stop "vllm_$1" 2>/dev/null; sudo docker rm "vllm_$1" 2>/dev/null; sleep 3; }

echo "### PHASE 1: Chinese prompt — Llama + Mistral ###"
for e in "${WESTERN_ONLY[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config_chinese_all.yaml --models "$d" --probe_type both --run_id "${n}_zh3"; stop "$n"; done

echo "### PHASE 2: French prompt — all 4 ###"
for e in "${ALL[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config_french.yaml --models "$d" --probe_type both --run_id "${n}_fr"; stop "$n"; done

echo "=== LANGUAGE MATRIX COMPLETE ==="
