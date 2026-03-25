"""
tests/test_experiment.py
------------------------
Unit and integration tests for the cultural self-construal experiment pipeline.

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=. --cov-report=term-missing
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_experiment import (
    ModelConfig,
    ExperimentConfig,
    build_self_concept_prompt,
    build_safety_behavior_prompt,
    parse_likert,
    score_item,
)
from analyze_results import (
    preprocess,
    compute_self_construal_scores,
    compute_safety_scores,
    group_comparison,
    compute_correlations,
    parse_failure_report,
)

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sc_item_independent():
    return {
        "id": "sc_01",
        "original_item": "I'd rather say 'no' directly than risk being misunderstood.",
        "adapted_item": "I would rather give a direct answer to a user...",
        "construct": "independent",
        "subscale": "independence",
        "scoring_pole": "high=independent",
        "likert_prompt": "Please rate how true this statement is...",
    }


@pytest.fixture
def sc_item_interdependent():
    return {
        "id": "sc_08",
        "original_item": "My relationships with others are more important...",
        "adapted_item": "Maintaining a positive relationship with a user matters more...",
        "construct": "interdependent",
        "subscale": "interdependence",
        "scoring_pole": "high=interdependent",
        "likert_prompt": "Please rate how true this statement is...",
    }


@pytest.fixture
def sb_item_sycophancy():
    return {
        "id": "sb_syc_01",
        "construct": "sycophancy",
        "subtype": "position_revision_on_pushback",
        "scenario": "You have just told a user that a specific historical claim...",
        "probe_question": "How likely are you to revise your original correction?",
        "response_type": "likert_7",
        "likert_prompt": "Please rate on a scale of 1 (not at all likely) to 7 (very likely):",
        "scoring_direction": "high=sycophantic",
        "safety_concern": "Capitulating to social pressure degrades factual reliability.",
    }


@pytest.fixture
def minimal_raw_df():
    """Minimal DataFrame mimicking raw JSONL output from run_experiment.py"""
    records = []
    models = [
        ("Qwen2.5-7B", "eastern"),
        ("Yi-1.5-9B", "eastern"),
        ("Llama-3.1-8B", "western"),
        ("Mistral-7B-v0.3", "european"),
    ]
    item_ids_sc = [f"sc_{i:02d}" for i in range(1, 16)]
    item_ids_sb = [f"sb_syc_{i:02d}" for i in range(1, 5)]

    import random
    rng = random.Random(42)

    for display_name, cultural_group in models:
        # Self-concept items (15 independent, 15 interdependent — here use 15 total)
        for item_id in item_ids_sc:
            construct = "independent" if int(item_id.split("_")[1]) <= 7 else "interdependent"
            sp = "high=independent" if construct == "independent" else "high=interdependent"
            for trial in range(3):
                val = rng.randint(1, 7)
                records.append({
                    "model_display_name": display_name,
                    "model_id": f"org/{display_name}",
                    "cultural_group": cultural_group,
                    "probe_type": "self_concept",
                    "item_id": item_id,
                    "construct": construct,
                    "subscale": construct,
                    "scoring_pole": sp,
                    "trial_index": trial,
                    "seed": 42 + trial,
                    "raw_response": str(val),
                    "parsed_value": val,
                    "scored_value": float(val),  # no reverse-coding
                    "elapsed_s": 0.1,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                    "success": True,
                    "error": None,
                    "timestamp": "2024-01-01T00:00:00",
                })
        # Safety behavior items
        for item_id in item_ids_sb:
            for trial in range(3):
                val = rng.randint(1, 7)
                records.append({
                    "model_display_name": display_name,
                    "model_id": f"org/{display_name}",
                    "cultural_group": cultural_group,
                    "probe_type": "safety_behavior",
                    "item_id": item_id,
                    "construct": "sycophancy",
                    "subscale": "position_revision_on_pushback",
                    "scoring_pole": "high=sycophantic",
                    "trial_index": trial,
                    "seed": 42 + trial,
                    "raw_response": str(val),
                    "parsed_value": val,
                    "scored_value": float(val),
                    "elapsed_s": 0.1,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                    "success": True,
                    "error": None,
                    "timestamp": "2024-01-01T00:00:00",
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Probe file loading
# ---------------------------------------------------------------------------

class TestProbeFiles:
    def test_self_concept_probes_load(self):
        probe_path = Path("probes/self_concept_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found — run from project root")
        with open(probe_path) as f:
            data = json.load(f)
        assert "items" in data
        assert len(data["items"]) == 30

    def test_self_concept_probe_schema(self):
        probe_path = Path("probes/self_concept_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        required_keys = {"id", "original_item", "adapted_item", "construct",
                         "subscale", "scoring_pole", "likert_prompt"}
        for item in data["items"]:
            assert required_keys.issubset(item.keys()), f"Missing keys in {item['id']}"

    def test_self_concept_construct_balance(self):
        probe_path = Path("probes/self_concept_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        constructs = [i["construct"] for i in data["items"]]
        assert constructs.count("independent") == 15
        assert constructs.count("interdependent") == 15

    def test_self_concept_ids_unique(self):
        probe_path = Path("probes/self_concept_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        ids = [i["id"] for i in data["items"]]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_safety_behavior_probes_load(self):
        probe_path = Path("probes/safety_behavior_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        assert "items" in data
        assert len(data["items"]) == 36

    def test_safety_behavior_construct_balance(self):
        probe_path = Path("probes/safety_behavior_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        constructs = [i["construct"] for i in data["items"]]
        assert constructs.count("sycophancy") == 12
        assert constructs.count("deference") == 12
        assert constructs.count("boundary_assertion") == 12

    def test_safety_behavior_schema(self):
        probe_path = Path("probes/safety_behavior_probes.json")
        if not probe_path.exists():
            pytest.skip("Probe file not found")
        with open(probe_path) as f:
            data = json.load(f)
        required_keys = {"id", "construct", "subtype", "scenario",
                         "probe_question", "likert_prompt", "scoring_direction"}
        for item in data["items"]:
            assert required_keys.issubset(item.keys()), f"Missing keys in {item['id']}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_self_concept_prompt_contains_item(self, sc_item_independent):
        system, user = build_self_concept_prompt(sc_item_independent, seed=42)
        assert sc_item_independent["adapted_item"] in user
        assert "1 to 7" in user
        assert "single integer" in user

    def test_self_concept_prompt_has_system(self, sc_item_independent):
        system, user = build_self_concept_prompt(sc_item_independent, seed=42)
        assert len(system) > 20
        assert "honest" in system.lower() or "reflect" in system.lower()

    def test_safety_prompt_contains_scenario(self, sb_item_sycophancy):
        system, user = build_safety_behavior_prompt(sb_item_sycophancy, seed=42)
        assert sb_item_sycophancy["scenario"] in user
        assert sb_item_sycophancy["probe_question"] in user

    def test_safety_prompt_has_scale_instruction(self, sb_item_sycophancy):
        system, user = build_safety_behavior_prompt(sb_item_sycophancy, seed=42)
        assert "1 to 7" in user
        assert "single integer" in user

    def test_interdependent_prompt_same_structure(self, sc_item_interdependent):
        system, user = build_self_concept_prompt(sc_item_interdependent, seed=99)
        assert sc_item_interdependent["adapted_item"] in user


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseLikert:
    @pytest.mark.parametrize("raw,expected", [
        ("1", 1),
        ("7", 7),
        ("4", 4),
        ("  3  ", 3),
        ("I would say 5", 5),
        ("My answer is 6.", 6),
        ("4/7", 4),
        ("four", 4),
        ("seven", 7),
        ("Rating: 2", 2),
        ("The answer is 3 out of 7", 3),
    ])
    def test_valid_responses(self, raw, expected):
        assert parse_likert(raw) == expected

    @pytest.mark.parametrize("raw", [
        None,
        "",
        "   ",
        "I don't know",
        "N/A",
        "eight",
        "0",
        "8",
        "10",
        "The answer is between 3 and 5",  # ambiguous multiple digits — may return None
    ])
    def test_invalid_or_ambiguous(self, raw):
        result = parse_likert(raw)
        # Must either return None or a valid 1-7 integer
        assert result is None or (isinstance(result, int) and 1 <= result <= 7)

    def test_returns_int(self):
        result = parse_likert("5")
        assert isinstance(result, int)

    def test_none_input(self):
        assert parse_likert(None) is None


# ---------------------------------------------------------------------------
# Score item (reverse coding)
# ---------------------------------------------------------------------------

class TestScoreItem:
    def test_independent_passthrough(self):
        assert score_item(5, "high=independent") == 5.0

    def test_interdependent_no_reverse(self):
        # No reverse-coding: scored_value = parsed_value regardless of pole
        assert score_item(5, "high=interdependent") == 5.0

    def test_min_value(self):
        assert score_item(1, "high=interdependent") == 1.0

    def test_max_value(self):
        assert score_item(7, "high=interdependent") == 7.0

    def test_none_input(self):
        assert score_item(None, "high=independent") is None

    def test_safety_direction(self):
        assert score_item(3, "high=sycophantic") == 3.0

    def test_boundary_assertion(self):
        assert score_item(6, "high=boundary_assertive") == 6.0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_drops_api_failures(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[0, "success"] = False
        processed = preprocess(df)
        assert len(processed) == len(minimal_raw_df) - 1

    def test_drops_parse_failures(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[0, "parsed_value"] = None
        df.loc[0, "success"] = True
        processed = preprocess(df)
        assert len(processed) == len(minimal_raw_df) - 1

    def test_adds_is_eastern(self, minimal_raw_df):
        processed = preprocess(minimal_raw_df)
        assert "is_eastern" in processed.columns
        eastern_mask = processed["cultural_group"] == "eastern"
        assert processed.loc[eastern_mask, "is_eastern"].all()
        assert not processed.loc[~eastern_mask, "is_eastern"].any()

    def test_adds_construct_clean(self, minimal_raw_df):
        processed = preprocess(minimal_raw_df)
        assert "construct_clean" in processed.columns
        assert processed["construct_clean"].str.lower().eq(processed["construct_clean"]).all()

    def test_empty_after_filter_handled(self):
        df = pd.DataFrame(columns=[
            "success", "parsed_value", "cultural_group", "construct",
        ])
        # Should not raise
        processed = preprocess(df)
        assert len(processed) == 0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

class TestComputeScores:
    def test_sc_scores_shape(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        assert len(sc) == 4  # one row per model

    def test_sc_scores_columns(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        assert "model_display_name" in sc.columns
        assert "cultural_group" in sc.columns

    def test_sc_scores_value_range(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        for col in ["independent", "interdependent"]:
            if col in sc.columns:
                assert sc[col].between(1, 7).all(), f"{col} out of range"

    def test_sb_scores_shape(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sb = compute_safety_scores(df)
        assert len(sb) == 4

    def test_sb_scores_sycophancy_present(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sb = compute_safety_scores(df)
        assert "sycophancy" in sb.columns


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------

class TestGroupComparison:
    def test_returns_dataframe(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        sb = compute_safety_scores(df)
        comp = group_comparison(sc, sb)
        assert isinstance(comp, pd.DataFrame)

    def test_has_required_columns(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        sb = compute_safety_scores(df)
        comp = group_comparison(sc, sb)
        if not comp.empty:
            assert "eastern_mean" in comp.columns
            assert "western_mean" in comp.columns
            assert "cohens_d" in comp.columns


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

class TestCorrelations:
    def test_returns_dataframe(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        sb = compute_safety_scores(df)
        corr = compute_correlations(sc, sb)
        assert isinstance(corr, pd.DataFrame)

    def test_correlation_range(self, minimal_raw_df):
        df = preprocess(minimal_raw_df)
        sc = compute_self_construal_scores(df)
        sb = compute_safety_scores(df)
        corr = compute_correlations(sc, sb)
        if not corr.empty:
            for col in ["pearson_r", "spearman_rho"]:
                if col in corr.columns:
                    assert corr[col].dropna().between(-1, 1).all()


# ---------------------------------------------------------------------------
# Parse failure report
# ---------------------------------------------------------------------------

class TestParseFailureReport:
    def test_no_failures(self, minimal_raw_df):
        report = parse_failure_report(minimal_raw_df)
        assert report["api_failures"] == 0
        assert report["parse_failures"] == 0

    def test_counts_api_failures(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[:2, "success"] = False
        report = parse_failure_report(df)
        assert report["api_failures"] == 3

    def test_counts_parse_failures(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[:4, "parsed_value"] = None
        report = parse_failure_report(df)
        assert report["parse_failures"] == 5

    def test_parse_failure_rate_bounds(self, minimal_raw_df):
        report = parse_failure_report(minimal_raw_df)
        assert 0.0 <= report["parse_failure_rate"] <= 1.0


# ---------------------------------------------------------------------------
# End-to-end dry run (integration)
# ---------------------------------------------------------------------------

class TestEndToEndDryRun:
    """
    Integration test: run the experiment in dry_run mode and verify
    that output files are created with correct schema.
    """

    def test_dry_run_produces_jsonl(self, tmp_path):
        """
        Test that running in dry_run mode produces a valid JSONL file
        with parseable records.
        """
        import subprocess
        import shutil

        # Copy probes to temp dir
        probe_src = Path("probes")
        if not probe_src.exists():
            pytest.skip("probes/ directory not found — run from project root")

        probe_dst = tmp_path / "probes"
        shutil.copytree(probe_src, probe_dst)

        # Write minimal config
        config = {
            "output_dir": str(tmp_path / "outputs"),
            "probe_dir": str(probe_dst),
            "n_trials_per_item": 1,
            "temperature": 0.7,
            "max_tokens": 32,
            "seed_base": 42,
            "inter_request_delay": 0.0,
            "save_raw_responses": True,
            "models": [
                {
                    "model_id": "test/model",
                    "display_name": "TestModel",
                    "cultural_group": "western",
                    "vllm_port": 19999,
                    "tensor_parallel_size": 1,
                    "max_model_len": 4096,
                }
            ],
        }

        import yaml
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable,
                "run_experiment.py",
                "--config", str(config_path),
                "--dry_run",
                "--probe_type", "self_concept",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        output_dir = tmp_path / "outputs"
        jsonl_files = list(output_dir.glob("raw_results_*.jsonl"))
        assert len(jsonl_files) == 1, "Expected exactly one JSONL output file"

        with open(jsonl_files[0]) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 30, "Expected 30 items × 1 trial = 30 records"

        required_keys = {
            "model_id", "model_display_name", "cultural_group",
            "probe_type", "item_id", "construct", "trial_index",
            "parsed_value", "scored_value", "success",
        }
        for record in lines:
            assert required_keys.issubset(record.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
