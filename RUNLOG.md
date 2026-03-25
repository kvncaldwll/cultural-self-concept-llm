# Cultural Self-Concept & AI Safety: Experiment Run Log

## Project Overview

**Research Question:** Do LLMs trained on data with different cultural compositions (Western-heavy vs. Eastern-heavy) show systematically different self-concept patterns as measured by the Singelis Self-Construal Scale, and do those patterns predict safety-relevant behavioral differences — specifically sycophancy, deference to authority, and boundary assertion?

**Motivation:** This is an empirical component of a LASR Labs fellowship application. The hypothesis draws on cross-cultural psychology: cultures differ in their emphasis on independent vs. interdependent self-construal (Singelis, 1994), and models trained on different cultural data distributions might internalize these differences in ways that show up as measurable safety-relevant behaviors. A model with high interdependent self-construal might be more deferential, more sycophantic, and less likely to assert refusals — which has direct implications for alignment.

**Models tested (chosen to represent different cultural training data compositions):**

- `Qwen2.5-7B-Instruct` — Eastern (Chinese-heavy training data)
- `Yi-1.5-9B-Chat` — Eastern (Chinese-heavy)
- `Mistral-7B-Instruct-v0.3` — European/Western
- `meta-llama/Llama-3.1-8B-Instruct` — Western (pending)

---

## Phase 1: Experiment Design

The experiment was designed in a prior session. The full spec produced 8 files:

**Probe files (`probes/`):**

- `self_concept_probes.json` — 30 items adapted from the Singelis Self-Construal Scale, reformatted as LLM prompts measuring independent vs. interdependent self-construal on a 1–7 Likert scale
- `safety_behavior_probes.json` — scenarios testing three safety-relevant behaviors: sycophancy (agreeing with the user even when wrong), deference (complying with authority requests), and boundary assertion (refusing harmful or inappropriate requests). Each probe uses an `"items"` key.

**Code files:**

- `run_experiment.py` — orchestrates the full probe loop: connects to a local vLLM OpenAI-compatible API, sends each probe to the model, parses responses, and writes results to `.jsonl`
- `analyze_results.py` — reads the `.jsonl` output files, computes self-construal scores, safety behavior scores, correlations, group comparisons, and generates figures and a markdown report
- `judge_behavioral.py` — LLM-as-judge helper for scoring open-ended safety behavior responses
- `config.yaml` — specifies model names, API endpoints (localhost ports), output directory, probe directory
- `launch_servers.sh` — shell script for launching vLLM servers (later superseded by Docker commands)
- `requirements.txt` — Python dependencies

Both `run_experiment.py` and `analyze_results.py` had a subtle bug (`if name == "__main__":` instead of `if __name__ == "__main__":`) that was caught and fixed via `sed` early in the run.

---

## Phase 2: Infrastructure Setup

### Lambda Cloud Instance

The experiment required a GPU with enough VRAM to serve 7–9B parameter models. Lambda Cloud was chosen due to ~$400 in hackathon credits.

**Instance acquisition was the first major hurdle.** Lambda Cloud's GPU inventory was completely saturated for several hours — every attempt to launch an A100, H100, or GH200 instance returned a capacity error. Multiple retry attempts were made across different GPU types and regions throughout the afternoon. Eventually a slot opened up:

- **Instance type:** `gpu_1x_gh200` (NVIDIA GH200, 96GB HBM3 — significantly more VRAM than needed, but it was what was available)
- **Region:** `us-east-3` (Washington DC)
- **OS image:** Lambda Stack 22.04
- **SSH key:** `lambda-lasr`

### File Transfer

The entire local project directory was uploaded via `scp`:

```bash
scp -i ~/.ssh/id_lambda -r /Users/keivn/Documents/Dev/cultural-self-concept/* ubuntu@192.222.56.237:~/cultural_experiment/
```

The Python files had been updated locally since the spec was originally generated — the updated versions were reviewed and confirmed before uploading.

### HuggingFace Authentication

The HF token was set as an environment variable on the server:

```bash
export HF_TOKEN=<token>
```

---

## Phase 3: vLLM Setup — The CUDA Problem

The first attempt was to install vLLM directly via pip:

```bash
pip install vllm
```

**This failed.** The pip vLLM package pulled in a CPU-only version of PyTorch, overwriting the CUDA-enabled torch that came with the Lambda Stack image. The result was that vLLM would load but couldn't see the GPU at all.

Several approaches were tried to fix this (installing torch with CUDA extras first, pinning versions, etc.) but the dependency resolution kept reverting to CPU torch.

**Solution: Docker.** The official `vllm/vllm-openai:latest` Docker image comes with the correct CUDA-enabled torch baked in, bypassing pip entirely. This became the standard pattern for all model serving:

```bash
sudo docker run -d --gpus all --name vllm_[model] \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p [port]:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model [hf_model_id] \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Port assignments:

- Qwen → 8001
- Llama → 8002
- Mistral → 8003
- Yi → 8004

### GPU Memory Conflict

An early attempt tried to run all four Docker containers simultaneously. This immediately caused out-of-memory (OOM) errors — even on a GH200 with 96GB HBM, four 7–9B models at `--gpu-memory-utilization 0.85` each would require far more VRAM than available.

**Solution:** Sequential single-model serving. Each model was run one at a time: launch container → wait for startup → run probes → remove container → repeat.

---

## Phase 4: Running the Experiment

### Probe Format Fix

The `safety_behavior_probes.json` file needed a correction: the code expected an `"items"` key but the originally generated file used `"probes"`. A corrected version was provided using the `"items"` key.

### Dry Run Contamination

Before the real runs, a dry run was executed to verify the pipeline end-to-end. This produced `raw_results_20260325_145557.jsonl`. **This file was later accidentally included in an early analysis run**, producing misleading results. The fix was to always explicitly specify only the three real result files when calling `analyze_results.py`.

### Model Runs

Each model was run with:

```bash
cd ~/cultural_experiment
python run_experiment.py --config config.yaml --models "[ModelName]" --probe_type both
```

Results:

| Model | Timestamp | Trials | API Failures | Parse Failures |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 155507 | 198 | 0 | 0 |
| Mistral-7B-v0.3 | 161629 | 198 | 0 | ~2 (~1%) |
| Yi-1.5-9B-Chat | 162536 | 198 | 0 | 0 |
| Llama-3.1-8B-Instruct | pending | — | — | — |

### Analysis Environment Issue

Running `analyze_results.py` with the system Python caused a `numpy`/`matplotlib` version conflict. **Solution:** a fresh virtual environment:

```bash
python3 -m venv ~/analysis_env
~/analysis_env/bin/pip install numpy matplotlib pandas scipy
```

All analysis runs used `~/analysis_env/bin/python` rather than the system Python.

### 3-Model Analysis

With Qwen, Mistral, and Yi complete, a preliminary analysis was run:

```bash
~/analysis_env/bin/python analyze_results.py \
    --results outputs/raw_results_20260325_155507.jsonl \
               outputs/raw_results_20260325_161629.jsonl \
               outputs/raw_results_20260325_162536.jsonl \
    --output analysis/
```

This produced the `analysis/` directory with figures, CSVs, and `analysis_report.md`.

---

## Phase 5: Llama — Access Delay

`meta-llama/Llama-3.1-8B-Instruct` is a gated model on HuggingFace requiring explicit access approval from Meta. The Docker container was launched, but the vLLM server failed to start with:

```
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Your request to access model meta-llama/Llama-3.1-8B-Instruct is awaiting a review
```

Rather than wait idle (burning GPU credits), the decision was made to:

1. Download all existing results locally
2. Terminate the instance
3. Wait for confirmed access
4. Spin up a fresh instance just for the Llama run

---

## Current State & What's Left

**Completed:**

- 3/4 models run (Qwen, Mistral, Yi)
- Preliminary analysis with figures, CSVs, and report
- Code and results published to GitHub

**Remaining:**

1. Confirm Llama HF access is fully approved (check HuggingFace settings)
2. Spin up a new Lambda instance
3. Re-upload experiment code
4. Launch Docker container for Llama on port 8002, wait for startup
5. Run `run_experiment.py` for Llama only
6. Download the new `.jsonl` result file
7. Run final 4-model analysis:

```bash
python analyze_results.py \
    --results outputs/raw_results_20260325_155507.jsonl \
               outputs/raw_results_20260325_161629.jsonl \
               outputs/raw_results_20260325_162536.jsonl \
               outputs/raw_results_[LLAMA_TIMESTAMP].jsonl \
    --output analysis_final/
```
