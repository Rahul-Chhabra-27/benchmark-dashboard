#!/usr/bin/env python3
"""Build a standalone dashboard from completed LOFT, RULER, and Synthetic-KV sweeps."""

import ast
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "kvpress" / "evaluation"
OUT = Path(__file__).with_name("index.html")
DOWNLOADS = Path(__file__).with_name("downloads")

SOURCES = [
    {
        "id": "loft32k-fp",
        "title": "LOFT 32K · Non-quantized",
        "group": "loft32k",
        "group_title": "LOFT 32K",
        "precision": "Non-quantized",
        "path": "results_loft32k_qwen_yarn4_fullcontext_all_budgets",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "YaRN-4 · full context (max_context_length=null)",
        "budget_provenance": {},
        "expected_tasks": ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"],
    },
    {
        "id": "loft32k-awq",
        "title": "LOFT 32K · AWQ",
        "group": "loft32k",
        "group_title": "LOFT 32K",
        "precision": "AWQ 4-bit",
        "path": "results_loft32k_qwen_awq_yarn4_fullcontext_all_budgets",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B-AWQ · 4-bit weights · FP16 KV cache",
        "budget_provenance": {},
        "expected_tasks": ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"],
        "publish_predictions": True,
    },
    {
        "id": "loft128k",
        "title": "LOFT 128K · Non-quantized",
        "group": "loft128k",
        "group_title": "LOFT 128K",
        "precision": "Non-quantized",
        "paths": [
            "results_loft128k_qwen_yarn4",
            "results_loft128k_qwen_yarn4_dgx_4gb",
        ],
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "YaRN-4",
    },
    {
        "id": "ruler32k",
        "title": "RULER 32K",
        "group": "ruler32k",
        "group_title": "RULER 32K",
        "precision": "Unquantized",
        "path": "results_ruler32k_qwen_yarn4_full_seed42",
        "kind": "ruler",
        "budgets": ["unbounded", "0.2 GB", "0.4 GB", "0.6 GB", "0.8 GB", "1 GB"],
        "expected_tasks": [
            "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
            "niah_single_3", "qa_1", "qa_2", "vt",
        ],
    },
    {
        "id": "synthetic64k-awq",
        "title": "Synthetic-KV 64K · AWQ",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "AWQ 4-bit",
        "path": "results_synthetic_kv_awq_no_prefix_all_budgets",
        "kind": "synthetic",
        "budgets": ["No compression", "512", "1024", "2048", "4096", "8192"],
        "provenance": "Qwen3-8B-AWQ · YaRN-4 · max context 65,536 · full 2,340-sample dataset",
        "expected_tasks": ["64k"],
    },
    {
        "id": "synthetic64k-nonquantized",
        "title": "Synthetic-KV 64K · Non-quantized",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "Non-quantized",
        "paths": [
            "results_synthetic_kv_direct_no_prefix",
            "results_synthetic_kv_no_prefix_all_budgets",
        ],
        "kind": "synthetic",
        "budgets": ["No compression", "512", "1024", "2048", "4096", "8192"],
        "provenance": "Qwen3-8B BF16 · YaRN-4 · max context 65,536 · full 2,340-sample dataset",
        "expected_tasks": ["64k"],
    },
]


def budget_label(name: str, kind: str) -> str:
    match = re.search(r"__memory_budget([0-9.]+)(MB|GB)(?:__|$)", name)
    if not match:
        if kind in ("loft", "synthetic"):
            return "No compression" if "__no_press__" in name else "KVzip 1% prune"
        return "unbounded"
    value, unit = match.groups()
    if kind in ("loft", "synthetic"):
        return f"{float(value) * (1024 if unit == 'GB' else 1):g}"
    value = value.rstrip("0").rstrip(".") if "." in value else value
    return f"{value} {unit}"


def task_name(name, kind):
    prefixes = {
        "loft": "new_loft__",
        "ruler": "new_ruler32k__",
        "synthetic": "new_synthetic_kv__",
    }
    prefix = prefixes[kind]
    if not name.startswith(prefix) or "__--home" not in name:
        return None
    return name[len(prefix) :].split("__--home", 1)[0]


def score_fields(metrics, kind):
    if kind == "ruler":
        nested = next((v for v in metrics.values() if isinstance(v, dict)), {})
        return {k: float(v) for k, v in nested.items() if isinstance(v, (int, float))}
    if kind == "synthetic":
        nested = next((v for v in metrics.values() if isinstance(v, dict)), {})
        return {
            key: float(nested[key])
            for key in ("exact_match", "string_match")
            if key in nested
        }
    keys = ("em", "subspan_em", "f1", "coverage")
    scores = {key: float(metrics[key]) * 100 for key in keys if key in metrics}
    # LOFT has task-specific official metrics: list-answer tasks expose coverage,
    # while short-answer QA tasks expose token F1. This unified field ensures a
    # complete per-task chart without pretending an unavailable metric is zero.
    primary = metrics.get("coverage", metrics.get("f1", metrics.get("subspan_em", metrics.get("em"))))
    if primary is not None:
        scores["primary_score"] = float(primary) * 100
    return scores


def publish_predictions(metric_file: Path, source_id: str, budget: str, task: str = ""):
    prediction_file = metric_file.with_name("predictions.csv")
    if not prediction_file.exists():
        return None, []

    DOWNLOADS.mkdir(exist_ok=True)
    safe_budget = re.sub(r"[^a-z0-9]+", "-", budget.lower()).strip("-")
    safe_task = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    task_part = f"-{safe_task}" if safe_task else ""
    public_name = f"{source_id}{task_part}-{safe_budget}.csv"
    shutil.copyfile(prediction_file, DOWNLOADS / public_name)

    with prediction_file.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return f"downloads/{public_name}", []

    # Evenly spaced rows provide a deterministic, non-cherry-picked preview.
    count = min(8, len(rows))
    indices = sorted({round(i * (len(rows) - 1) / max(count - 1, 1)) for i in range(count)})
    preview = []
    for index in indices:
        row = rows[index]
        reference = row.get("answer") or row.get("answers") or ""
        try:
            parsed = ast.literal_eval(reference)
            if isinstance(parsed, list) and parsed:
                reference = str(parsed[0])
        except (SyntaxError, ValueError):
            pass
        prediction = row.get("predicted_answer", "")
        preview.append(
            {
                "row": index + 1,
                "question": row.get("question", ""),
                "reference": reference,
                "prediction": prediction,
                "raw_match": prediction.strip().casefold() == reference.strip().casefold(),
            }
        )
    return f"downloads/{public_name}", preview


def collect(source):
    grouped = {}
    newest = 0.0
    for relative_path in source.get("paths", [source.get("path")]):
        directory = EVAL / relative_path
        for metric_file in directory.rglob("metrics.json"):
            run_dir = next(
                (parent for parent in metric_file.parents if parent.parent == directory),
                metric_file.parent,
            )
            task = task_name(run_dir.name, source["kind"])
            if task is None:
                continue
            try:
                metrics = json.loads(metric_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            budget = budget_label(run_dir.name, source["kind"])
            newest = max(newest, metric_file.stat().st_mtime)
            nested = next((v for v in metrics.values() if isinstance(v, dict)), {})
            prediction_url, prediction_preview = (
                publish_predictions(
                    metric_file,
                    source["id"],
                    budget,
                    "" if source["kind"] == "synthetic" else task,
                )
                if source["kind"] == "synthetic" or source.get("publish_predictions")
                else (None, [])
            )
            run = {
                "scores": score_fields(metrics, source["kind"]),
                "samples": int(metrics.get("num_samples", nested.get(
                    "num_samples", 200 if source["kind"] == "ruler" else 0
                ))),
                "retained_tokens": metrics.get("average_retained_context_tokens"),
                "original_tokens": metrics.get("average_original_context_tokens"),
                "retained_gb": metrics.get("average_retained_kv_memory_gb"),
                "compression": metrics.get("average_compression_ratio"),
                "prediction_url": prediction_url,
                "prediction_preview": prediction_preview,
            }
            grouped.setdefault(task, {})[budget] = run

    # Display compression relative to true no-press when available, otherwise
    # relative to the least-compressed reference included in that sweep.
    for runs in grouped.values():
        reference = "No compression" if "No compression" in runs else (
            "KVzip 1% prune" if "KVzip 1% prune" in runs else "unbounded"
        )
        baseline = runs.get(reference, {}).get("compression")
        if baseline is None or baseline >= 1:
            continue
        baseline_retained = 1 - baseline
        for run in runs.values():
            if run["compression"] is not None:
                run["compression"] = 1 - ((1 - run["compression"]) / baseline_retained)

    expected = set(source["budgets"])
    allowed_tasks = set(source.get("expected_tasks", grouped))
    complete = {
        task: {budget: runs[budget] for budget in source["budgets"]}
        for task, runs in sorted(grouped.items())
        if task in allowed_tasks and expected.issubset(runs)
    }
    expected_tasks = set(source.get("expected_tasks", grouped))
    incomplete = sorted(task for task in expected_tasks if not expected.issubset(grouped.get(task, {})))
    metrics = sorted({key for runs in complete.values() for run in runs.values() for key in run["scores"]})
    return {
        **{k: source[k] for k in ("id", "title", "group", "group_title", "precision", "kind", "budgets")},
        "provenance": source.get("provenance", ""),
        "budget_provenance": source.get("budget_provenance", {}),
        "tasks": complete,
        "metrics": metrics,
        "excluded": incomplete,
        "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
    }


def build() -> None:
    datasets = [collect(source) for source in SOURCES]
    template = (Path(__file__).with_name("template.html")).read_text()
    output = template.replace("__DASHBOARD_DATA__", json.dumps(datasets, separators=(",", ":")))
    OUT.write_text(output)
    print(f"Wrote {OUT}")
    for dataset in datasets:
        print(f"{dataset['title']}: {len(dataset['tasks'])} complete, {len(dataset['excluded'])} excluded")


if __name__ == "__main__":
    build()
