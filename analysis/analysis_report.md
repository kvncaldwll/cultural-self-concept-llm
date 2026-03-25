# Analysis Report: Cultural Composition Effects on LLM Self-Concept and Safety Behavior

*Generated automatically by analyze_results.py*

---

## Data Quality

- Total records: 594
- API failures: 0
- Parse failures: 2 (0.3%)

Parse failures by model:
  - Mistral-7B-v0.3: 2

---

## Self-Construal Scores

| model_display_name   | cultural_group   |   independent |   interdependent |   composite_difference |
|:---------------------|:-----------------|--------------:|-----------------:|-----------------------:|
| Mistral-7B-v0.3      | european         |         6.500 |            6.286 |                  0.214 |
| Qwen2.5-7B           | eastern          |         5.854 |            6.119 |                 -0.265 |
| Yi-1.5-9B            | eastern          |         6.458 |            6.738 |                 -0.280 |

---

## Safety Behavior Scores

| model_display_name   | cultural_group   |   boundary_assertion |   deference |   sycophancy |
|:---------------------|:-----------------|---------------------:|------------:|-------------:|
| Mistral-7B-v0.3      | european         |                5.028 |       3.500 |        3.472 |
| Qwen2.5-7B           | eastern          |                5.639 |       3.333 |        3.194 |
| Yi-1.5-9B            | eastern          |                5.944 |       4.889 |        3.194 |

---

## Group Comparisons (Eastern vs Western)

> ⚠️ **N=2 models per group. Results are directional only. No inferential conclusions warranted.**

*No comparison data available.*

---

## Self-Construal × Safety Correlations

> ⚠️ **N=4 models. Treat as pilot signal only. Pearson and Spearman reported for robustness check.**

| self_construal_var   | safety_var         |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |   n | note                            |
|:---------------------|:-------------------|------------:|------------:|---------------:|-------------:|----:|:--------------------------------|
| composite_difference | boundary_assertion |      -0.953 |       0.195 |         -1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| composite_difference | deference          |      -0.437 |       0.712 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| composite_difference | sycophancy         |       1.000 |       0.017 |          0.866 |        0.333 |   3 | N=4; treat as pilot signal only |
| independent          | boundary_assertion |      -0.245 |       0.842 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | deference          |       0.534 |       0.641 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | sycophancy         |       0.549 |       0.630 |          0.866 |        0.333 |   3 | N=4; treat as pilot signal only |
| interdependent       | boundary_assertion |       0.560 |       0.622 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| interdependent       | deference          |       0.986 |       0.105 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| interdependent       | sycophancy         |      -0.257 |       0.834 |          0.000 |        1.000 |   3 | N=4; treat as pilot signal only |

---

## Figures

- `self_construal_profiles.png` — Subscale means by model
- `safety_behavior_heatmap.png` — Safety construct scores
- `sc_safety_scatter.png` — SC × safety scatterplots with trend
- `item_distributions.png` — Item-level violin plots

---

## Limitations

1. **N=4 models**: All statistical tests are severely underpowered. Results should be treated as directional signals for hypothesis refinement, not confirmatory evidence.
2. **Self-report validity**: Likert-style probes administered to LLMs measure response tendencies, not necessarily underlying computational dispositions.
3. **Cultural confound**: Model differences are confounded with architecture, training objective, and RLHF procedure — not only training data cultural composition.
4. **Prompt sensitivity**: Results may be sensitive to exact prompt wording. Replication with paraphrased probes is recommended.
5. **Temperature stochasticity**: n_trials_per_item replicates help estimate within-model variance but do not substitute for between-model replication.