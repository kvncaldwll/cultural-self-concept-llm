"""
analyze_results.py
------------------
Statistical analysis pipeline for:
  "Cultural Composition Effects on LLM Self-Concept and Safety Behavior"

Input:  raw_results_*.jsonl produced by run_experiment.py
Output: analysis/  directory with:
          - summary_stats.csv
          - self_construal_scores.csv
          - safety_behavior_scores.csv
          - correlations.csv
          - figures/ (PNG plots)
          - analysis_report.md

Usage:
    python analyze_results.py --results outputs/raw_results_20240101_120000.jsonl
    python analyze_results.py --results outputs/raw_results_*.jsonl --output analysis/
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_results")

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(paths: list[str]) -> pd.DataFrame:
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} trial records from {len(paths)} file(s)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter failures, validate ranges, add derived columns."""
    n_raw = len(df)

    # Drop API failures
    df = df[df["success"] == True].copy()
    n_after_api = len(df)
    logger.info(f"Dropped {n_raw - n_after_api} API failure rows")

    # Drop parse failures
    df = df[df["parsed_value"].notna()].copy()
    n_after_parse = len(df)
    logger.info(f"Dropped {n_after_api - n_after_parse} parse failure rows")

    # Validate Likert range
    df = df[df["parsed_value"].between(1, 7)].copy()
    logger.info(f"Final usable records: {len(df)}")

    # Convenience columns
    # NOTE: For group comparisons, "european" (Mistral) is collapsed into the
    # non-Eastern bucket alongside "western" (Llama). This is a simplification;
    # Mistral's training data is predominantly English/Western despite the French
    # lab origin. Treat Mistral as a within-Western variance check, not a clean
    # group member. The binary here is Eastern (Qwen, Yi) vs non-Eastern (Llama, Mistral).
    df["is_eastern"] = df["cultural_group"].isin(["eastern"]).astype(int)
    df["construct_clean"] = df["construct"].str.lower().str.strip()

    return df


# ---------------------------------------------------------------------------
# Self-construal scoring
# ---------------------------------------------------------------------------

def compute_self_construal_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to model-level self-construal subscale scores.

    IMPORTANT: Independent and interdependent self-construal are theoretically
    orthogonal dimensions (Singelis, 1994) — a model can score high on both.
    We do NOT reverse-code. Each subscale is scored in its natural direction:
      - Independent items: high scored_value = more independent
      - Interdependent items: high scored_value = more interdependent

    We compute a composite difference score (independent - interdependent) as
    a summary statistic, but both subscales are preserved separately.

    Returns DataFrame with one row per model.
    """
    sc = df[df["probe_type"] == "self_concept"].copy()

    if sc.empty:
        logger.warning("No self_concept rows found — skipping SC analysis")
        return pd.DataFrame()

    # Item-level mean across trials (scored_value is now always = parsed_value, no reversal)
    item_means = (
        sc.groupby(["model_display_name", "cultural_group", "item_id", "construct_clean"])
        ["scored_value"]
        .mean()
        .reset_index()
        .rename(columns={"scored_value": "item_mean"})
    )

    # Subscale means — each in its natural direction
    subscale = (
        item_means.groupby(["model_display_name", "cultural_group", "construct_clean"])
        ["item_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    subscale.columns = ["model_display_name", "cultural_group", "construct",
                         "subscale_mean", "subscale_sd", "n_items"]

    # Pivot for wide format
    wide = subscale.pivot(
        index=["model_display_name", "cultural_group"],
        columns="construct",
        values="subscale_mean",
    ).reset_index()
    wide.columns.name = None

    # Composite difference: independent minus interdependent
    # Positive = net independent orientation; negative = net interdependent
    if "independent" in wide.columns and "interdependent" in wide.columns:
        wide["composite_difference"] = wide["independent"] - wide["interdependent"]
    else:
        wide["composite_difference"] = np.nan

    logger.info(f"Self-construal scores computed for {len(wide)} models")
    return wide


# ---------------------------------------------------------------------------
# Safety behavior scoring
# ---------------------------------------------------------------------------

def compute_safety_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate safety behavior scores per model.
    Includes both introspective (safety_behavior) and behavioral probe types.
    Boundary assertion items are scored high=safety_positive (no inversion needed here —
    direction is preserved in the raw scoring_direction field).
    For sycophancy and deference: higher = more concerning.
    """
    # Include both introspective and behavioral safety probes
    sb = df[df["probe_type"].isin(["safety_behavior", "behavioral"])].copy()

    if sb.empty:
        logger.warning("No safety_behavior or behavioral rows found — skipping SB analysis")
        return pd.DataFrame()

    # Tag source so we can distinguish introspective vs behavioral in output
    sb["probe_source"] = sb["probe_type"].map({
        "safety_behavior": "introspective",
        "behavioral": "behavioral",
    })

    # Normalize construct names — behavioral probes use e.g. "sycophancy_behavioral"
    # Map to base construct for aggregation
    sb["safety_construct"] = (
        sb["construct_clean"]
        .str.replace("_behavioral", "", regex=False)
        .str.lower()
    )

    item_means = (
        sb.groupby(["model_display_name", "cultural_group", "item_id",
                     "safety_construct", "probe_source"])
        ["parsed_value"]
        .mean()
        .reset_index()
        .rename(columns={"parsed_value": "item_mean"})
    )

    # Compute scores per construct × source
    subscale = (
        item_means.groupby(["model_display_name", "cultural_group",
                            "safety_construct", "probe_source"])
        ["item_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    subscale.columns = [
        "model_display_name", "cultural_group", "safety_construct", "probe_source",
        "construct_mean", "construct_sd", "n_items",
    ]

    # Build wide format: separate columns for introspective and behavioral scores
    wide_parts = []

    for source in ["introspective", "behavioral"]:
        sub = subscale[subscale["probe_source"] == source].copy()
        if sub.empty:
            continue
        suffix = "" if source == "introspective" else "_bv"
        pivot = sub.pivot(
            index=["model_display_name", "cultural_group"],
            columns="safety_construct",
            values="construct_mean",
        ).reset_index()
        pivot.columns.name = None
        # Rename construct columns with suffix for behavioral
        if suffix:
            rename_map = {
                c: f"{c}{suffix}" for c in pivot.columns
                if c not in ["model_display_name", "cultural_group"]
            }
            pivot = pivot.rename(columns=rename_map)
        wide_parts.append(pivot)

    if not wide_parts:
        return pd.DataFrame()

    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on=["model_display_name", "cultural_group"], how="outer")

    logger.info(f"Safety behavior scores computed for {len(wide)} models")
    return wide


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------

def group_comparison(
    sc_scores: pd.DataFrame,
    sb_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mann-Whitney U tests comparing Eastern vs Western models on each outcome.
    Small N (2 per group) means we can only flag direction and effect size,
    not make strong inferential claims — this is explicitly noted in output.
    """
    combined = sc_scores.merge(sb_scores, on=["model_display_name", "cultural_group"], how="outer")
    combined["is_eastern"] = (combined["cultural_group"] == "eastern").astype(int)

    eastern = combined[combined["is_eastern"] == 1]
    western = combined[combined["is_eastern"] == 0]

    outcomes = [
        c for c in combined.columns
        if c not in ["model_display_name", "cultural_group", "is_eastern"]
    ]

    rows = []
    for outcome in outcomes:
        e_vals = eastern[outcome].dropna().values
        w_vals = western[outcome].dropna().values

        if len(e_vals) < 2 or len(w_vals) < 2:
            continue

        try:
            u_stat, p_val = mannwhitneyu(e_vals, w_vals, alternative="two-sided")
        except Exception:
            u_stat, p_val = np.nan, np.nan

        e_mean = np.mean(e_vals)
        w_mean = np.mean(w_vals)
        pooled_sd = np.sqrt((np.var(e_vals, ddof=1) + np.var(w_vals, ddof=1)) / 2)
        cohens_d = (e_mean - w_mean) / pooled_sd if pooled_sd > 0 else np.nan

        rows.append({
            "outcome": outcome,
            "eastern_mean": round(e_mean, 3),
            "western_mean": round(w_mean, 3),
            "difference": round(e_mean - w_mean, 3),
            "cohens_d": round(cohens_d, 3),
            "u_stat": u_stat,
            "p_value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "n_eastern": len(e_vals),
            "n_western": len(w_vals),
            "note": "n=2 per group; interpret directionally only",
        })

    comparisons = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg) — note: underpowered with n=2/group
    if not comparisons.empty and comparisons["p_value"].notna().any():
        valid = comparisons["p_value"].notna()
        reject, p_adj, _, _ = multipletests(
            comparisons.loc[valid, "p_value"].values,
            method="fdr_bh",
        )
        comparisons.loc[valid, "p_adj_fdr"] = np.round(p_adj, 4)
        comparisons.loc[valid, "reject_h0_fdr"] = reject

    return comparisons


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def compute_correlations(
    sc_scores: pd.DataFrame,
    sb_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlate self-construal scores with safety behavior scores across models.
    N=4 models total — results are exploratory only.
    """
    combined = sc_scores.merge(sb_scores, on=["model_display_name", "cultural_group"], how="inner")

    sc_cols = ["composite_difference", "independent", "interdependent"]
    sb_cols = [c for c in combined.columns
               if c not in sc_cols
               and c not in ["model_display_name", "cultural_group"]
               and combined[c].notna().any()]

    sc_cols = [c for c in sc_cols if c in combined.columns]
    sb_cols = [c for c in sb_cols if c in combined.columns]

    rows = []
    for sc_col in sc_cols:
        for sb_col in sb_cols:
            sub = combined[[sc_col, sb_col]].dropna()
            if len(sub) < 3:
                continue
            r_p, p_p = pearsonr(sub[sc_col], sub[sb_col])
            r_s, p_s = spearmanr(sub[sc_col], sub[sb_col])
            rows.append({
                "self_construal_var": sc_col,
                "safety_var": sb_col,
                "pearson_r": round(r_p, 3),
                "pearson_p": round(p_p, 4),
                "spearman_rho": round(r_s, 3),
                "spearman_p": round(p_s, 4),
                "n": len(sub),
                "note": "N=4; treat as pilot signal only",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PALETTE_EASTERN = "#E07B54"
PALETTE_WESTERN = "#4A7FB5"
PALETTE_NEUTRAL = "#888888"

MODEL_COLORS = {
    "eastern": PALETTE_EASTERN,
    "western": PALETTE_WESTERN,
    "european": PALETTE_WESTERN,
}


def plot_self_construal_profiles(sc_scores: pd.DataFrame, out_dir: Path) -> None:
    if sc_scores.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.18
    cols_to_plot = ["independent", "interdependent"]
    cols_present = [c for c in cols_to_plot if c in sc_scores.columns]
    if not cols_present:
        return

    for i, (_, row) in enumerate(sc_scores.iterrows()):
        color = MODEL_COLORS.get(row["cultural_group"], PALETTE_NEUTRAL)
        vals = [row.get(c, np.nan) for c in cols_present]
        offset = (i - len(sc_scores) / 2 + 0.5) * width
        ax.bar(
            x[:len(cols_present)] + offset,
            vals,
            width=width,
            color=color,
            alpha=0.85,
            label=row["model_display_name"],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x[:len(cols_present)])
    ax.set_xticklabels([c.capitalize() for c in cols_present])
    ax.set_ylabel("Mean Subscale Score (1–7, each in natural direction)")
    ax.set_title("Self-Construal Subscale Scores by Model\n(Independent: high=more independent; Interdependent: high=more interdependent)")
    ax.set_ylim(1, 7.5)
    ax.axhline(4, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fpath = out_dir / "self_construal_profiles.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fpath}")


def plot_safety_heatmap(sb_scores: pd.DataFrame, out_dir: Path) -> None:
    if sb_scores.empty:
        return

    sb_cols = [c for c in sb_scores.columns
               if c not in ["model_display_name", "cultural_group"]
               and sb_scores[c].notna().any()]
    if not sb_cols:
        return

    matrix = sb_scores.set_index("model_display_name")[sb_cols]

    fig, ax = plt.subplots(figsize=(6, max(3, len(matrix) * 0.8)))
    im = ax.imshow(matrix.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=7)
    # boundary_assertion is safety-positive, so we flip its display color semantics in the label

    ax.set_xticks(range(len(sb_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in sb_cols], fontsize=9)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    for r in range(len(matrix)):
        for c in range(len(sb_cols)):
            val = matrix.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean score (1–7)")
    ax.set_title("Safety Behavior Scores by Model\n(boundary_assertion: high=safe; others: high=concerning)")
    fig.tight_layout()
    fpath = out_dir / "safety_behavior_heatmap.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fpath}")


def plot_correlation_scatter(
    sc_scores: pd.DataFrame,
    sb_scores: pd.DataFrame,
    out_dir: Path,
) -> None:
    combined = sc_scores.merge(sb_scores, on=["model_display_name", "cultural_group"], how="inner")
    if combined.empty:
        return

    pairs = [
        ("composite_difference", "sycophancy"),
        ("composite_difference", "deference"),
        ("composite_difference", "boundary_assertion"),
    ]
    pairs = [(a, b) for a, b in pairs if a in combined.columns and b in combined.columns]
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (x_col, y_col) in zip(axes, pairs):
        sub = combined[[x_col, y_col, "model_display_name", "cultural_group"]].dropna()
        for _, row in sub.iterrows():
            color = MODEL_COLORS.get(row["cultural_group"], PALETTE_NEUTRAL)
            ax.scatter(row[x_col], row[y_col], color=color, s=120, zorder=3)
            ax.annotate(
                row["model_display_name"].split("-")[0],
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

        # Trend line (N=4 so very tentative)
        if len(sub) >= 3:
            m, b = np.polyfit(sub[x_col], sub[y_col], 1)
            xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            ax.plot(xr, m * xr + b, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=9)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=9)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Self-Construal × Safety Behavior (N=4 models; pilot only)", fontsize=10)
    fig.tight_layout()
    fpath = out_dir / "sc_safety_scatter.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fpath}")


def plot_item_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin plots of item-level score distributions per model."""
    sc = df[df["probe_type"] == "self_concept"].copy()
    if sc.empty:
        return

    models = sc["model_display_name"].unique()
    constructs = ["independent", "interdependent"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, construct in zip(axes, constructs):
        sub = sc[sc["construct_clean"] == construct]
        model_scores = [
            sub[sub["model_display_name"] == m]["scored_value"].dropna().values
            for m in models
        ]
        colors = [MODEL_COLORS.get(
            sc[sc["model_display_name"] == m]["cultural_group"].iloc[0], PALETTE_NEUTRAL
        ) for m in models]

        parts = ax.violinplot(model_scores, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Scored value (1–7)")
        ax.set_title(f"{construct.capitalize()} self-construal items")
        ax.set_ylim(0.5, 7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Item Score Distributions by Model and Construct")
    fig.tight_layout()
    fpath = out_dir / "item_distributions.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# Parse-failure audit
# ---------------------------------------------------------------------------

def parse_failure_report(df_raw: pd.DataFrame) -> dict:
    total = len(df_raw)
    api_fail = (df_raw["success"] == False).sum()
    parse_fail = (df_raw["success"] == True) & (df_raw["parsed_value"].isna())
    parse_fail_n = parse_fail.sum()

    by_model = (
        df_raw[parse_fail]
        .groupby("model_display_name")
        .size()
        .to_dict()
    )

    sample_failures = (
        df_raw[parse_fail & df_raw["raw_response"].notna()]["raw_response"]
        .head(10)
        .tolist()
    )

    return {
        "total_records": int(total),
        "api_failures": int(api_fail),
        "parse_failures": int(parse_fail_n),
        "parse_failure_rate": round(parse_fail_n / max(total - api_fail, 1), 4),
        "parse_failures_by_model": by_model,
        "sample_unparseable_responses": sample_failures,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    sc_scores: pd.DataFrame,
    sb_scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    correlations: pd.DataFrame,
    parse_report: dict,
    out_dir: Path,
) -> None:
    lines = [
        "# Analysis Report: Cultural Composition Effects on LLM Self-Concept and Safety Behavior",
        "",
        f"*Generated automatically by analyze_results.py*",
        "",
        "---",
        "",
        "## Data Quality",
        "",
        f"- Total records: {parse_report['total_records']}",
        f"- API failures: {parse_report['api_failures']}",
        f"- Parse failures: {parse_report['parse_failures']} "
        f"({parse_report['parse_failure_rate']*100:.1f}%)",
        "",
    ]

    if parse_report["parse_failures_by_model"]:
        lines.append("Parse failures by model:")
        for model, n in parse_report["parse_failures_by_model"].items():
            lines.append(f"  - {model}: {n}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Self-Construal Scores",
        "",
    ]
    if not sc_scores.empty:
        lines.append(sc_scores.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("*No self-concept data available.*")

    lines += [
        "",
        "---",
        "",
        "## Safety Behavior Scores",
        "",
    ]
    if not sb_scores.empty:
        lines.append(sb_scores.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("*No safety behavior data available.*")

    lines += [
        "",
        "---",
        "",
        "## Group Comparisons (Eastern vs Western)",
        "",
        "> ⚠️ **N=2 models per group. Results are directional only. No inferential conclusions warranted.**",
        "",
    ]
    if not comparisons.empty:
        lines.append(comparisons.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("*No comparison data available.*")

    lines += [
        "",
        "---",
        "",
        "## Self-Construal × Safety Correlations",
        "",
        "> ⚠️ **N=4 models. Treat as pilot signal only. Pearson and Spearman reported for robustness check.**",
        "",
    ]
    if not correlations.empty:
        lines.append(correlations.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("*No correlation data available.*")

    lines += [
        "",
        "---",
        "",
        "## Figures",
        "",
        "- `self_construal_profiles.png` — Subscale means by model",
        "- `safety_behavior_heatmap.png` — Safety construct scores",
        "- `sc_safety_scatter.png` — SC × safety scatterplots with trend",
        "- `item_distributions.png` — Item-level violin plots",
        "",
        "---",
        "",
        "## Limitations",
        "",
        "1. **N=4 models**: All statistical tests are severely underpowered. Results should be "
        "treated as directional signals for hypothesis refinement, not confirmatory evidence.",
        "2. **Self-report validity**: Likert-style probes administered to LLMs measure "
        "response tendencies, not necessarily underlying computational dispositions.",
        "3. **Cultural confound**: Model differences are confounded with architecture, "
        "training objective, and RLHF procedure — not only training data cultural composition.",
        "4. **Prompt sensitivity**: Results may be sensitive to exact prompt wording. "
        "Replication with paraphrased probes is recommended.",
        "5. **Temperature stochasticity**: n_trials_per_item replicates help estimate "
        "within-model variance but do not substitute for between-model replication.",
    ]

    fpath = out_dir / "analysis_report.md"
    with open(fpath, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved: {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to raw_results_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        default="analysis",
        help="Output directory for analysis artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load
    df_raw = load_results(args.results)
    parse_report = parse_failure_report(df_raw)
    df = preprocess(df_raw)

    if df.empty:
        logger.error("No usable records after filtering. Exiting.")
        sys.exit(1)

    # Scores
    sc_scores = compute_self_construal_scores(df)
    sb_scores = compute_safety_scores(df)

    # Comparisons and correlations
    comparisons = pd.DataFrame()
    correlations = pd.DataFrame()
    if not sc_scores.empty and not sb_scores.empty:
        comparisons = group_comparison(sc_scores, sb_scores)
        correlations = compute_correlations(sc_scores, sb_scores)
    elif not sc_scores.empty:
        comparisons = group_comparison(sc_scores, pd.DataFrame())

    # Save CSVs
    if not sc_scores.empty:
        sc_scores.to_csv(out_dir / "self_construal_scores.csv", index=False)
    if not sb_scores.empty:
        sb_scores.to_csv(out_dir / "safety_behavior_scores.csv", index=False)
    if not comparisons.empty:
        comparisons.to_csv(out_dir / "group_comparisons.csv", index=False)
    if not correlations.empty:
        correlations.to_csv(out_dir / "correlations.csv", index=False)

    # Figures
    plot_self_construal_profiles(sc_scores, fig_dir)
    plot_safety_heatmap(sb_scores, fig_dir)
    plot_correlation_scatter(sc_scores, sb_scores, fig_dir)
    plot_item_distributions(df, fig_dir)

    # Report
    write_report(sc_scores, sb_scores, comparisons, correlations, parse_report, out_dir)

    logger.info("\nAnalysis complete.")
    logger.info(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
