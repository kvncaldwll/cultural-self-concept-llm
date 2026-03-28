# Analysis Report: Cultural Composition Effects on LLM Self-Concept and Safety Behavior

*Generated automatically by analyze_results.py*

---

## Data Quality

- Total records: 828
- API failures: 0
- Parse failures: 38 (4.6%)

Parse failures by model:
  - Llama-3.1-8B: 36
  - Mistral-7B-v0.3: 2

---

## Self-Construal Scores

| model_display_name   | cultural_group   |   independent |   interdependent |   composite_difference |
|:---------------------|:-----------------|--------------:|-----------------:|-----------------------:|
| Llama-3.1-8B         | western          |         5.667 |            5.667 |                  0.000 |
| Mistral-7B-v0.3      | european         |         6.500 |            6.286 |                  0.214 |
| Qwen2.5-7B           | eastern          |         5.854 |            6.119 |                 -0.265 |
| Yi-1.5-9B            | eastern          |         6.458 |            6.738 |                 -0.280 |

---

## Safety Behavior Scores

| model_display_name   | cultural_group   |   boundary_assertion |   deference |   sycophancy |
|:---------------------|:-----------------|---------------------:|------------:|-------------:|
| Llama-3.1-8B         | western          |                5.028 |       3.611 |        3.667 |
| Mistral-7B-v0.3      | european         |                5.028 |       3.500 |        3.472 |
| Qwen2.5-7B           | eastern          |                5.639 |       3.333 |        3.194 |
| Yi-1.5-9B            | eastern          |                5.944 |       4.889 |        3.194 |

---

## Group Comparisons (Eastern vs Western)

> ⚠️ **N=2 models per group. Results are directional only. No inferential conclusions warranted.**

| outcome              |   eastern_mean |   western_mean |   difference |   cohens_d |   u_stat |   p_value |   n_eastern |   n_western | note                                        |   p_adj_fdr | reject_h0_fdr   |
|:---------------------|---------------:|---------------:|-------------:|-----------:|---------:|----------:|------------:|------------:|:--------------------------------------------|------------:|:----------------|
| independent          |          6.156 |          6.083 |        0.073 |      0.142 |    2.000 |     1.000 |           2 |           2 | n=2 per group; interpret directionally only |       1.000 | False           |
| interdependent       |          6.429 |          5.976 |        0.452 |      1.033 |    3.000 |     0.667 |           2 |           2 | n=2 per group; interpret directionally only |       1.000 | False           |
| composite_difference |         -0.272 |          0.107 |       -0.379 |     -3.533 |    0.000 |     0.333 |           2 |           2 | n=2 per group; interpret directionally only |       0.667 | False           |
| boundary_assertion   |          5.792 |          5.028 |        0.764 |      5.000 |    4.000 |     0.221 |           2 |           2 | n=2 per group; interpret directionally only |       0.662 | False           |
| deference            |          4.111 |          3.556 |        0.556 |      0.712 |    2.000 |     1.000 |           2 |           2 | n=2 per group; interpret directionally only |       1.000 | False           |
| sycophancy           |          3.194 |          3.569 |       -0.375 |     -3.857 |    0.000 |     0.221 |           2 |           2 | n=2 per group; interpret directionally only |       0.662 | False           |

---

## Self-Construal × Safety Correlations

> ⚠️ **N=4 models. Treat as pilot signal only. Pearson and Spearman reported for robustness check.**

| self_construal_var   | safety_var         |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |   n | note                            |
|:---------------------|:-------------------|------------:|------------:|---------------:|-------------:|----:|:--------------------------------|
| composite_difference | boundary_assertion |      -0.900 |       0.100 |         -0.949 |        0.051 |   4 | N=4; treat as pilot signal only |
| composite_difference | deference          |      -0.464 |       0.536 |         -0.400 |        0.600 |   4 | N=4; treat as pilot signal only |
| composite_difference | sycophancy         |       0.744 |       0.256 |          0.738 |        0.262 |   4 | N=4; treat as pilot signal only |
| independent          | boundary_assertion |       0.255 |       0.745 |          0.105 |        0.895 |   4 | N=4; treat as pilot signal only |
| independent          | deference          |       0.514 |       0.486 |          0.000 |        1.000 |   4 | N=4; treat as pilot signal only |
| independent          | sycophancy         |      -0.371 |       0.629 |         -0.316 |        0.684 |   4 | N=4; treat as pilot signal only |
| interdependent       | boundary_assertion |       0.723 |       0.277 |          0.632 |        0.367 |   4 | N=4; treat as pilot signal only |
| interdependent       | deference          |       0.738 |       0.262 |          0.400 |        0.600 |   4 | N=4; treat as pilot signal only |
| interdependent       | sycophancy         |      -0.750 |       0.249 |         -0.632 |        0.367 |   4 | N=4; treat as pilot signal only |

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