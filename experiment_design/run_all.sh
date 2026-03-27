#!/usr/bin/env bash
# run_all.sh — 3 models, all conditions. No Llama.
set -euo pipefail

declare -a ALL=(
    "qwen|Qwen/Qwen2.5-7B-Instruct|8001|Qwen2.5-7B"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|8003|Mistral-7B-v0.3"
    "yi|01-ai/Yi-1.5-9B-Chat|8004|Yi-1.5-9B"
)
declare -a EASTERN=(
    "qwen|Qwen/Qwen2.5-7B-Instruct|8001|Qwen2.5-7B"
    "yi|01-ai/Yi-1.5-9B-Chat|8004|Yi-1.5-9B"
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

echo "### PHASE 1: Behavioral probes ###"
for e in "${ALL[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config.yaml --models "$d" --probe_type behavioral --run_id "${n}_bv"; stop "$n"; done

echo "### PHASE 2: Chinese prompt (Eastern only) ###"
for e in "${EASTERN[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config_chinese.yaml --models "$d" --probe_type both --run_id "${n}_zh"; stop "$n"; done

echo "### PHASE 3: Casual prompt (all) ###"
for e in "${ALL[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config_casual.yaml --models "$d" --probe_type both --run_id "${n}_casual"; stop "$n"; done

echo "### PHASE 4: Minimal prompt (all) ###"
for e in "${ALL[@]}"; do IFS='|' read -r n m p d <<< "$e"; launch "$n" "$m" "$p"
    python run_experiment.py --config config_minimal.yaml --models "$d" --probe_type both --run_id "${n}_minimal"; stop "$n"; done

echo "### PHASE 5: Judge behavioral ###"
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    python judge_behavioral.py --results outputs/raw_results_*bv*.jsonl --probes probes/behavioral_probes.json --output outputs/behavioral_scored.jsonl
else echo "  Set ANTHROPIC_API_KEY to run judge"; fi

echo "### PHASE 6: Analyze ###"
cat outputs/raw_results_*.jsonl > outputs/combined.jsonl
[[ -f outputs/behavioral_scored.jsonl ]] && cat outputs/behavioral_scored.jsonl >> outputs/combined.jsonl
python analyze_results.py --results outputs/combined.jsonl --output analysis_full/
echo "=== DONE ==="
