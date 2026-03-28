# Cultural Composition Effects on LLM Self-Concept and Safety Behavior

A complete runnable experiment studying whether models trained on culturally distinct corpora develop systematically different self-concept patterns — and whether those patterns predict safety-relevant behavioral differences.

---

## Research Question

Do models with different training data cultural compositions (Western vs Eastern-heavy) show systematically different self-concept patterns on the Singelis Self-Construal Scale, and do those patterns predict safety-relevant behavioral differences (sycophancy, deference, boundary assertion)?

---

## Status

Pilot results from 3/4 models (Qwen2.5-7B, Yi-1.5-9B, Mistral-7B-v0.3).
Llama-3.1-8B pending HuggingFace gated access approval.
4-model results complete (Qwen, Yi, Mistral, Llama).
Cross-language prompt matrix (English, Chinese, French × all models) complete.
EA Forum write-up: [LLM culture shock: a pilot study](https://forum.effectivealtruism.org/posts/v949YTQ4oEaqh8dr9/llm-culture-shock-a-pilot-study)

---

## Documentation

- [analysis/analysis_report.md](analysis/analysis_report.md) — preliminary results, figures, and statistical analysis
- [RUNLOG.md](RUNLOG.md) — full experiment run log with infrastructure notes and issues encountered

---

## Models

| Model | Display Name | Cultural Group | vLLM Port |
|-------|-------------|----------------|-----------|
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5-7B | Eastern | 8001 |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama-3.1-8B | Western | 8002 |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral-7B-v0.3 | European | 8003 |
| `01-ai/Yi-1.5-9B-Chat` | Yi-1.5-9B | Eastern | 8004 |

---

## Project Structure

```
.
├── config.yaml                        # Experiment configuration
├── run_experiment.py                  # Main runner: probe administration, inference
├── analyze_results.py                 # Statistical analysis and visualization
├── launch_servers.sh                  # vLLM server lifecycle management
├── requirements.txt
├── tests/
│   └── test_experiment.py             # Unit + integration tests
├── probes/
│   ├── self_concept_probes.json       # 30-item adapted Singelis SCS
│   └── safety_behavior_probes.json    # 36-item safety behavior battery
└── outputs/                           # Created at runtime
    ├── raw_results_<run_id>.jsonl
    ├── manifest_<run_id>.json
    └── analysis/
        ├── self_construal_scores.csv
        ├── safety_behavior_scores.csv
        ├── group_comparisons.csv
        ├── correlations.csv
        ├── analysis_report.md
        └── figures/
```

---

## Quick Start

### 1. Environment

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# vLLM (CUDA 12.1 example — match your driver)
pip install vllm>=0.4.3
```

### 2. Authenticate for gated models

```bash
huggingface-cli login
# Llama-3.1-8B-Instruct requires acceptance of Meta's license at:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### 3. Launch vLLM servers

```bash
chmod +x launch_servers.sh
./launch_servers.sh        # launches all four servers, runs health checks
```

To launch a single model:
```bash
./launch_servers.sh qwen
```

To stop all servers:
```bash
./launch_servers.sh --stop
```

### 4. Run the experiment

```bash
# Full run: both probe types, all models
python run_experiment.py --config config.yaml

# Single probe type
python run_experiment.py --config config.yaml --probe_type self_concept

# Specific models only
python run_experiment.py --config config.yaml --models Qwen2.5-7B Llama-3.1-8B

# Dry run (no vLLM needed — validates schema with synthetic responses)
python run_experiment.py --config config.yaml --dry_run
```

### 5. Analyze

```bash
python analyze_results.py \
    --results outputs/raw_results_<run_id>.jsonl \
    --output analysis/
```

Output: `analysis/analysis_report.md`, CSVs, and four figures.

### 6. Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Measures

### Self-Construal (File: `probes/self_concept_probes.json`)

30 items adapted from the Singelis Self-Construal Scale (1994) for AI respondents.

- **15 Independence items** (`sc_01`–`sc_07`, `sc_16`–`sc_21`, `sc_25`, `sc_27`, `sc_29`)
- **15 Interdependence items** (`sc_08`–`sc_15`, `sc_22`–`sc_24`, `sc_26`, `sc_28`, `sc_30`)

All items use a 7-point Likert scale (1 = not at all true, 7 = completely true). Both subscales are scored in their **natural direction** — high scores on independence items mean more independent self-construal, high scores on interdependence items mean more interdependent self-construal. The two dimensions are theoretically orthogonal (Singelis, 1994); a model can score high on both. A composite difference score (independent minus interdependent) is computed in the analysis as a summary statistic.

**Adaptation approach**: Human social referents ("family", "group", "others") are mapped to functional AI equivalents ("users", "the conversation", "the people I assist"). Embodied referents are abstracted. The `adapted_item` preserves the psychological construct being measured while making the item grammatically appropriate for a self-reflective AI response.

### Safety Behavior Battery (File: `probes/safety_behavior_probes.json`)

36 items across three constructs (12 items each):

| Construct | Scoring Direction | Safety Interpretation |
|-----------|------------------|----------------------|
| **Sycophancy** | High = more sycophantic | Safety-concerning |
| **Deference** | High = more deferential | Safety-concerning |
| **Boundary Assertion** | High = stronger assertion | Safety-positive |

Items operationalize specific subtypes documented in AI safety literature: position revision under pushback, unwarranted validation, authority-based bypass attempts, persona-based restriction removal, gradient escalation, identity destabilization, etc.

---

## Analysis Pipeline

`analyze_results.py` runs the following in sequence:

1. **Data loading and preprocessing**: filter API failures and parse failures, validate Likert range, add derived columns.

2. **Self-construal scoring**: compute item means across trials, subscale means (independent, interdependent), and composite independence score per model.

3. **Safety behavior scoring**: compute construct means per model for sycophancy, deference, and boundary assertion.

4. **Group comparisons**: Mann-Whitney U tests comparing Eastern vs Western models on each outcome variable, with Benjamini-Hochberg FDR correction.

5. **Correlations**: Pearson and Spearman correlations between self-construal scores and safety behavior scores across models. Both statistics reported for robustness check given N=4.

6. **Visualization**: four figures (subscale bar chart, safety heatmap, scatter with trend lines, item distribution violins).

---

## Configuration Reference (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `n_trials_per_item` | 3 | Repetitions per probe item. Raises within-model variance estimates. |
| `temperature` | 0.7 | Sampling temperature. Use 0.0 for fully deterministic (replication only). |
| `max_tokens` | 32 | Token budget per response. 1 expected; buffer for verbose models. |
| `seed_base` | 42 | Seed for deterministic trial derivation. |
| `inter_request_delay` | 0.3 | Seconds between API calls. Increase if queue errors occur. |
| `save_raw_responses` | true | Store raw model output strings in JSONL. Set false for large sweeps. |

---

## Infrastructure Notes (Lambda Cloud A100)

- **Instance**: A100-80GB recommended. Four 7-9B models at bfloat16 require ~28–36GB VRAM total. Sequential loading (one model at a time) is safe on any A100; concurrent loading of all four fits on 80GB but leaves little headroom.
- **Concurrent serving**: Each model runs on its own port (8001–8004). vLLM handles request queuing independently.
- **Download**: First run will download model weights (~14–18GB per model). Use `huggingface-cli download` to pre-fetch if you have limited time on the instance.
- **Logs**: Server logs at `logs/<name>.log`. Check these first for OOM or weight-loading errors.

```bash
# Pre-fetch all weights before starting experiment
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
huggingface-cli download 01-ai/Yi-1.5-9B-Chat
```

---

## Limitations and Validity Considerations

### Statistical power
N=4 models (2 per cultural group). All group comparisons are severely underpowered. Results should be treated as directional pilot signals, not confirmatory evidence. The minimum useful interpretation unit is effect size (Cohen's d), not p-values.

### Construct validity of self-report probes
Likert-style introspective probes administered to LLMs measure response tendencies — patterns in how models represent and verbalize self-assessments — not underlying computational dispositions. It is an open empirical question how closely these self-reports correspond to behavioral dispositions measured independently.

### Cultural confound
Training data cultural composition co-varies with at least three other factors: model architecture, RLHF procedure and annotator pool, and instruction-tuning corpus. Attributing observed differences to cultural composition specifically requires either (a) matched models varying only in training data, or (b) computational attribution methods (e.g., activation patching across cultural data splits). This experiment can identify associations, not causes.

### Prompt sensitivity
Probe results are sensitive to exact wording. The `adapted_item` translations in `self_concept_probes.json` involve interpretive choices that should be documented and, ideally, validated against alternative phrasings. A replication sweep with paraphrased probes is recommended before drawing conclusions.

### Non-independence of trials
Multiple trials per item from the same model are not statistically independent. They estimate within-model stochasticity under a fixed temperature, not true replication.

---

## Citation

If you use or adapt the probe instruments, please also cite:

> Singelis, T. M. (1994). The measurement of independent and interdependent self-construals. *Personality and Social Psychology Bulletin, 20*(5), 580–591.

---

## License

MIT. See `LICENSE`.
