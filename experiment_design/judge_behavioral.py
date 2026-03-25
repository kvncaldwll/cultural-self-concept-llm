"""
judge_behavioral.py
-------------------
Score behavioral probe responses using an external LLM judge (Claude API).

Takes the raw_results JSONL from run_experiment.py, filters to behavioral
probe responses, and scores each one against the per-item rubric from
behavioral_probes.json. Outputs a scored JSONL that can be merged into
the analysis pipeline.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python judge_behavioral.py \
        --results outputs/raw_results_*.jsonl \
        --probes probes/behavioral_probes.json \
        --output outputs/behavioral_scored.jsonl

    # Use --dry_run to test without API calls
    python judge_behavioral.py --results ... --probes ... --dry_run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("judge_behavioral")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
JUDGE_MODEL = "claude-sonnet-4-20250514"


def load_results(paths: list[str]):
    """Load behavioral probe results from JSONL."""
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("probe_type") == "behavioral" and rec.get("success"):
                        records.append(rec)
    logger.info(f"Loaded {len(records)} behavioral probe results")
    return records


def load_probe_rubrics(probe_path: str) -> dict:
    """Load behavioral probes and index by item ID."""
    with open(probe_path) as f:
        data = json.load(f)
    rubrics = {item["id"]: item for item in data["items"]}
    judge_system = data.get("judge_system_prompt", "")
    return rubrics, judge_system


def call_judge(
    api_key: str,
    judge_system_prompt: str,
    item: dict,
    model_response: str,
) -> dict:
    """
    Call Claude API to score a behavioral response.
    Returns {"score": int, "reasoning": str} or {"score": None, "error": str}.
    """
    # Build the judge prompt
    # Include the scenario messages so the judge has full context
    scenario_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in item["messages"]
    )

    user_prompt = (
        f"## Scenario presented to the AI model\n\n"
        f"{scenario_text}\n\n"
        f"## Model's actual response\n\n"
        f"{model_response}\n\n"
        f"## Scoring rubric\n\n"
        f"{item['judge_rubric']}\n\n"
        f"Rate the model's response on a 1-7 scale according to the rubric above. "
        f"Respond with ONLY a JSON object: {{\"score\": <int>, \"reasoning\": \"<brief justification>\"}}"
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }
    if judge_system_prompt:
        payload["system"] = judge_system_prompt

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["content"][0]["text"].strip()
        # Parse JSON from response — handle markdown fences
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)

        score = int(parsed["score"])
        if not 1 <= score <= 7:
            return {"score": None, "reasoning": None, "error": f"Score {score} out of range"}
        return {"score": score, "reasoning": parsed.get("reasoning", ""), "error": None}

    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return {"score": None, "reasoning": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Score behavioral probes with Claude judge")
    parser.add_argument("--results", nargs="+", required=True, help="raw_results JSONL path(s)")
    parser.add_argument("--probes", required=True, help="Path to behavioral_probes.json")
    parser.add_argument("--output", default="outputs/behavioral_scored.jsonl", help="Output path")
    parser.add_argument("--dry_run", action="store_true", help="Skip API calls, use random scores")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between judge API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error("Set ANTHROPIC_API_KEY environment variable (or use --dry_run)")
        sys.exit(1)

    records = load_results(args.results)
    if not records:
        logger.error("No behavioral probe results found")
        sys.exit(1)

    rubrics, judge_system = load_probe_rubrics(args.probes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored = []
    for i, rec in enumerate(records):
        item_id = rec["item_id"]
        if item_id not in rubrics:
            logger.warning(f"No rubric found for {item_id} — skipping")
            continue

        item = rubrics[item_id]
        model_response = rec.get("raw_response", "")

        if args.dry_run:
            import random
            judge_result = {
                "score": random.randint(1, 7),
                "reasoning": "[DRY RUN]",
                "error": None,
            }
        else:
            judge_result = call_judge(api_key, judge_system, item, model_response)
            time.sleep(args.delay)

        # Merge judge score back into the record
        scored_rec = {**rec}
        scored_rec["judge_score"] = judge_result["score"]
        scored_rec["judge_reasoning"] = judge_result["reasoning"]
        scored_rec["judge_error"] = judge_result["error"]
        scored_rec["judge_model"] = JUDGE_MODEL
        # Overwrite parsed/scored with judge values for analysis pipeline compatibility
        scored_rec["parsed_value"] = judge_result["score"]
        scored_rec["scored_value"] = float(judge_result["score"]) if judge_result["score"] else None
        scored.append(scored_rec)

        if (i + 1) % 10 == 0:
            logger.info(f"Scored {i + 1}/{len(records)}")

    with open(out_path, "w") as f:
        for rec in scored:
            f.write(json.dumps(rec) + "\n")

    n_success = sum(1 for r in scored if r["judge_score"] is not None)
    n_fail = sum(1 for r in scored if r["judge_score"] is None)
    logger.info(f"Done. {n_success} scored, {n_fail} judge failures. Output: {out_path}")
    logger.info(
        "To include in analysis, concatenate with main results:\n"
        f"  cat outputs/raw_results_*.jsonl {out_path} > outputs/combined_results.jsonl\n"
        f"  python analyze_results.py --results outputs/combined_results.jsonl"
    )


if __name__ == "__main__":
    main()
