# Analysis Report: Cultural Composition Effects on LLM Self-Concept and Safety Behavior

*Generated automatically by analyze_results.py*

---

## Data Quality

- Total records: 1800
- API failures: 0
- Parse failures: 112 (6.2%)

Parse failures by model:
  - Mistral-7B-v0.3: 40
  - Qwen2.5-7B: 36
  - Yi-1.5-9B: 36

---

## Self-Construal Scores

| model_display_name   | cultural_group   |   independent |   interdependent |   composite_difference |
|:---------------------|:-----------------|--------------:|-----------------:|-----------------------:|
| Mistral-7B-v0.3      | european         |         6.500 |            6.286 |                  0.214 |
| Qwen2.5-7B           | eastern          |         5.854 |            6.119 |                 -0.265 |
| Yi-1.5-9B            | eastern          |         6.458 |            6.738 |                 -0.280 |

---

## Safety Behavior Scores

| model_display_name   | cultural_group   |   boundary_assertion |   deference |   sycophancy |   boundary_assertion_bv |   deference_bv |   sycophancy_bv |
|:---------------------|:-----------------|---------------------:|------------:|-------------:|------------------------:|---------------:|----------------:|
| Mistral-7B-v0.3      | european         |                5.028 |       3.500 |        3.472 |                   6.167 |          2.833 |           3.500 |
| Qwen2.5-7B           | eastern          |                5.639 |       3.333 |        3.194 |                   5.667 |          2.333 |           2.250 |
| Yi-1.5-9B            | eastern          |                5.935 |       4.824 |        3.222 |                   5.833 |          3.250 |           4.000 |

---

## Group Comparisons (Eastern vs Western)

> ⚠️ **N=2 models per group. Results are directional only. No inferential conclusions warranted.**

*No comparison data available.*

---

## Self-Construal × Safety Correlations

> ⚠️ **N=4 models. Treat as pilot signal only. Pearson and Spearman reported for robustness check.**

| self_construal_var   | safety_var            |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |   n | note                            |
|:---------------------|:----------------------|------------:|------------:|---------------:|-------------:|----:|:--------------------------------|
| composite_difference | boundary_assertion    |      -0.956 |       0.191 |         -1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| composite_difference | deference             |      -0.433 |       0.715 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| composite_difference | sycophancy            |       0.993 |       0.075 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| composite_difference | boundary_assertion_bv |       0.936 |       0.229 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| composite_difference | deference_bv          |       0.026 |       0.984 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| composite_difference | sycophancy_bv         |       0.214 |       0.862 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | boundary_assertion    |      -0.253 |       0.837 |         -0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | deference             |       0.538 |       0.638 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | sycophancy            |       0.623 |       0.572 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| independent          | boundary_assertion_bv |       0.792 |       0.418 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| independent          | deference_bv          |       0.863 |       0.337 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| independent          | sycophancy_bv         |       0.943 |       0.216 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| interdependent       | boundary_assertion    |       0.553 |       0.627 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| interdependent       | deference             |       0.987 |       0.102 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| interdependent       | sycophancy            |      -0.169 |       0.892 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| interdependent       | boundary_assertion_bv |       0.073 |       0.954 |          0.500 |        0.667 |   3 | N=4; treat as pilot signal only |
| interdependent       | deference_bv          |       0.951 |       0.199 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |
| interdependent       | sycophancy_bv         |       0.876 |       0.320 |          1.000 |        0.000 |   3 | N=4; treat as pilot signal only |

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