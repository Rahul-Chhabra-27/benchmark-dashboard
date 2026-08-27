#!/usr/bin/env python3
"""Build a standalone dashboard from completed KVPress and RLM sweeps."""

import ast
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "kvpress" / "evaluation" / "results"
# `evaluation/results` is a symlink that currently only reaches the LOFT-32K
# matrices; LOFT-128K, the 27B-model runs, and the 27B synthetic-KV runs live
# directly under benchmark_artifacts instead. Sources opt into this base with
# "base": "benchmark_artifacts" rather than repointing the existing symlink.
BENCH = ROOT / "kvpress" / "benchmark_artifacts" / "results"
OUT = Path(__file__).with_name("index.html")
DOWNLOADS = Path(__file__).with_name("downloads")

SOURCES = [
    {
        "id": "loft32k-fp",
        "title": "LOFT 32K · Non-quantized",
        "group": "loft32k",
        "group_title": "LOFT 32K",
        "precision": "Plain KVzip · Non-quantized",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
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
        "precision": "Plain KVzip · AWQ 4-bit",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
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
        "precision": "Plain KVzip · Non-quantized",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "paths": [
            "results_loft128k_qwen_yarn4",
            "results_loft128k_qwen_yarn4_dgx_4gb",
        ],
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "YaRN-4",
    },
    {
        "id": "loft128k-awq",
        "title": "LOFT 128K · AWQ",
        "group": "loft128k",
        "group_title": "LOFT 128K",
        "precision": "Plain KVzip · AWQ 4-bit",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "path": "results_loft128k_qwen_awq_yarn4_all_budgets",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B-AWQ · 4-bit weights · FP16 KV cache",
        "budget_provenance": {},
        "expected_tasks": ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"],
        "publish_predictions": True,
    },
    {
        "id": "loft32k-qwen3-4b-instruct",
        "title": "LOFT 32K · Plain KVzip · Qwen3-4B-Instruct-2507",
        "group": "loft32k",
        "group_title": "LOFT 32K",
        "precision": "Plain KVzip · Qwen3-4B-Instruct-2507",
        "model": "qwen3-4b-instruct-2507",
        "model_title": "Qwen3-4B-Instruct-2507",
        "path": "loft/32k/matrices/results_loft32k_qwen3_4b_instruct_2507_native_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-4B-Instruct-2507 BF16 · native RoPE (no YaRN needed at 32K)",
        "budget_provenance": {},
        "expected_tasks": ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"],
        "publish_predictions": True,
    },
    {
        "id": "loft128k-qwen3-4b-instruct",
        "title": "LOFT 128K · Plain KVzip · Qwen3-4B-Instruct-2507",
        "group": "loft128k",
        "group_title": "LOFT 128K",
        "precision": "Plain KVzip · Qwen3-4B-Instruct-2507",
        "model": "qwen3-4b-instruct-2507",
        "model_title": "Qwen3-4B-Instruct-2507",
        "path": "loft/128k/matrices/results_loft128k_qwen3_4b_instruct_2507_native_dgx_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-4B-Instruct-2507 BF16 · native RoPE",
        "budget_provenance": {},
        "expected_tasks": ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"],
        "publish_predictions": True,
    },
    {
        "id": "loft32k-qwen35-27b-gptq",
        "title": "LOFT 32K · Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "group": "loft32k",
        "group_title": "LOFT 32K",
        "precision": "Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "model": "qwen35-27b-gptq-int4",
        "model_title": "Qwen3.5-27B-GPTQ-Int4",
        "path": "loft/32k/runs/results_loft32k_qwen35_27b_gptq_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3.5-27B-GPTQ-Int4 · 4-bit GPTQ weights · FP16 KV cache",
        "budget_provenance": {},
        "expected_tasks": ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"],
        "publish_predictions": True,
    },
    {
        "id": "loft128k-qwen35-27b-gptq",
        "title": "LOFT 128K · Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "group": "loft128k",
        "group_title": "LOFT 128K",
        "precision": "Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "model": "qwen35-27b-gptq-int4",
        "model_title": "Qwen3.5-27B-GPTQ-Int4",
        "path": "loft/128k/runs/results_loft128k_qwen35_27b_gptq_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "loft",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3.5-27B-GPTQ-Int4 · 4-bit GPTQ weights · FP16 KV cache",
        "budget_provenance": {},
        "expected_tasks": ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"],
        "publish_predictions": True,
    },
    {
        "id": "synthetic32k-qwen35-27b-gptq",
        "title": "Synthetic-KV 32K · Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "group": "synthetic32k",
        "group_title": "Synthetic-KV 32K",
        "precision": "Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "model": "qwen35-27b-gptq-int4",
        "model_title": "Qwen3.5-27B-GPTQ-Int4",
        "path": "synthetic_kv/32k/runs/results_synthetic_kv_32k_qwen35_27b_gptq_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096", "8192"],
        "provenance": "Qwen3.5-27B-GPTQ-Int4 · 4-bit GPTQ weights · max context 32,768",
        "expected_tasks": ["32k"],
    },
    {
        "id": "synthetic64k-qwen35-27b-gptq",
        "title": "Synthetic-KV 64K · Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "model": "qwen35-27b-gptq-int4",
        "model_title": "Qwen3.5-27B-GPTQ-Int4",
        "path": "synthetic_kv/64k/runs/results_synthetic_kv_64k_qwen35_27b_gptq_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096", "8192"],
        "provenance": "Qwen3.5-27B-GPTQ-Int4 · 4-bit GPTQ weights · max context 65,536",
        "expected_tasks": ["64k"],
    },
    {
        "id": "ruler32k",
        "title": "RULER 32K",
        "group": "ruler32k",
        "group_title": "RULER 32K",
        "precision": "Plain KVzip · Unquantized",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
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
        "id": "ruler32k-qwen35-27b-gptq",
        "title": "RULER 32K · Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "group": "ruler32k",
        "group_title": "RULER 32K",
        "precision": "Plain KVzip · Qwen3.5-27B-GPTQ-Int4",
        "model": "qwen35-27b-gptq-int4",
        "model_title": "Qwen3.5-27B-GPTQ-Int4",
        "path": "ruler32k/runs/results_ruler32k_qwen35_27b_gptq_all_budgets",
        "base": "benchmark_artifacts",
        "kind": "ruler",
        "budgets": ["unbounded", "0.2 GB", "0.4 GB", "0.6 GB", "0.8 GB", "1 GB"],
        "provenance": "Qwen3.5-27B-GPTQ-Int4 · 4-bit GPTQ weights · FP16 KV cache",
        "expected_tasks": [
            "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
            "niah_single_3", "qa_1", "qa_2", "vt",
        ],
        "publish_predictions": True,
    },
    {
        "id": "synthetic32k-nonquantized",
        "title": "Synthetic-KV 32K · Non-quantized",
        "group": "synthetic32k",
        "group_title": "Synthetic-KV 32K",
        "precision": "Plain KVzip · Non-quantized",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "path": "results_synthetic_kv_32k_native_no_yarn",
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B BF16 · max context 32,768 · 866-sample dataset",
        "expected_tasks": ["32k"],
    },
    {
        "id": "synthetic32k-awq",
        "title": "Synthetic-KV 32K · AWQ",
        "group": "synthetic32k",
        "group_title": "Synthetic-KV 32K",
        "precision": "Plain KVzip · AWQ 4-bit",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "path": "results_synthetic_kv_32k_native_awq_all_budgets",
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B-AWQ · max context 32,768 · 866-sample dataset",
        "expected_tasks": ["32k"],
    },
    {
        "id": "synthetic64k-awq",
        "title": "Synthetic-KV 64K · AWQ",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "Plain KVzip · AWQ 4-bit",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "path": "results_synthetic_kv_64k_yarn2_awq_all_budgets",
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B-AWQ · YaRN-2 · max context 65,536 · 1,707-sample dataset",
        "expected_tasks": ["64k"],
    },
    {
        "id": "synthetic64k-nonquantized",
        "title": "Synthetic-KV 64K · Non-quantized",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "Plain KVzip · Non-quantized",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "paths": [
            "results_synthetic_kv_64k_prefixed_yarn2",
        ],
        "kind": "synthetic",
        "budgets": ["No compression", "256", "512", "1024", "2048", "4096"],
        "provenance": "Qwen3-8B BF16 · YaRN-2 · max context 65,536 · 1,707-sample dataset",
        "expected_tasks": ["64k"],
    },
    {
        "id": "synthetic64k-native-no-yarn",
        "title": "Synthetic-KV 64K · No YaRN baseline",
        "group": "synthetic64k",
        "group_title": "Synthetic-KV 64K",
        "precision": "Plain KVzip · No YaRN · native RoPE",
        "model": "qwen3-8b",
        "model_title": "Qwen3-8B",
        "path": "results_synthetic_kv_64k_native_qwen3_8b_no_yarn_baseline",
        "kind": "synthetic",
        "budgets": ["No compression"],
        "provenance": "Qwen3-8B BF16 · no YaRN · max_context_length=null · native positional config 40,960",
        "expected_tasks": ["64k"],
    },
]


def rlm_sources():
    """Build dashboard variants from completed RLM JSONL checkpoints.

    RLM outputs are intentionally ignored by Git because they can be large and
    are produced on the cluster.  This collector keeps the dashboard build
    reproducible: any ``evaluation/results/rlm/**/*.jsonl`` files present at
    build time become an RLM-vs-vanilla comparison group automatically.
    """
    result_root = EVAL / "rlm"
    if not result_root.exists():
        return []

    grouped = {}
    for path in sorted(result_root.rglob("*.jsonl")):
        try:
            task, mode, model = path.stem.split(".", 2)
        except ValueError:
            # Ignore unrelated JSONL files rather than failing the whole site.
            continue
        condition = path.parent.name if path.parent != result_root else "default"
        key = (condition, mode, model)
        rows = []
        try:
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows.append(record)
        except OSError:
            continue
        if not rows:
            continue
        bucket = grouped.setdefault(
            key,
            {"condition": condition, "mode": mode, "model": model, "tasks": {}, "newest": 0.0},
        )
        bucket["tasks"].setdefault(task, []).extend(rows)
        bucket["newest"] = max(bucket["newest"], path.stat().st_mtime)

    sources = []
    for (condition, mode, model), data in sorted(grouped.items()):
        # Keep the labels short enough for the chart legend while retaining
        # the condition/model needed to identify an experiment.
        mode_label = "RLM" if mode == "rlm" else "Vanilla"
        precision = f"{mode_label} · {condition} · {model}"
        tasks = {}
        for task, rows in sorted(data["tasks"].items()):
            correct = sum(bool(row.get("correct")) for row in rows)
            finished = sum(bool(row.get("finished", True)) for row in rows)
            tokens = [row.get("tokens") for row in rows if isinstance(row.get("tokens"), (int, float))]
            latency = [row.get("latency_s") for row in rows if isinstance(row.get("latency_s"), (int, float))]
            tasks[task] = {
                "All examples": {
                    "scores": {
                        "accuracy": 100 * correct / len(rows),
                        "finished_rate": 100 * finished / len(rows),
                    },
                    "samples": len(rows),
                    "retained_tokens": None,
                    "original_tokens": None,
                    "retained_gb": None,
                    "compression": None,
                    "prediction_url": None,
                    "prediction_preview": [],
                    "tokens_per_query": sum(tokens) / len(tokens) if tokens else None,
                    "latency_per_query": sum(latency) / len(latency) if latency else None,
                }
            }
        sources.append(
            {
                "id": "rlm-" + re.sub(r"[^a-z0-9]+", "-", "-".join((condition, mode, model).lower())).strip("-"),
                "title": precision,
                "group": "rlm",
                "group_title": "RLM vs Vanilla",
                "precision": precision,
                "kind": "rlm",
                "budgets": ["All examples"],
                "provenance": f"RLM benchmark · condition {condition} · root model {model}",
                "budget_provenance": {},
                "tasks": tasks,
                "metrics": ["accuracy", "finished_rate"],
                "excluded": [],
                "updated": datetime.fromtimestamp(data["newest"]).strftime("%Y-%m-%d %H:%M") if data["newest"] else "—",
            }
        )
    return sources


RLM_KVZIP_VARIANTS = [
    (
        "32k",
        "loft32k",
        "LOFT 32K",
        ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"],
    ),
    (
        "128k",
        "loft128k",
        "LOFT 128K",
        ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"],
    ),
]

RLM_RUN_DIR_RE = re.compile(r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-(?P<press>kvzip|no_press)(?P<value>[0-9.]*)(?P<unit>MB|GB)?$")


def rlm_kvzip_sources():
    """RLM + KVzip sub-call compression vs. an RLM-no-press ablation.

    Reads the same evaluation/results/rlm/ tree rlm_sources() looks at, but
    parses run_benchmark.py's actual nested `<run-dir>/metrics.json` output
    (rlm_sources() expects a flat task.mode.model.jsonl layout that no longer
    matches). Uses "loft" as the kind and the plain-LOFT task names/group ids
    so these variants land in the same LOFT 32K/128K views as the plain-KVzip
    sources above and can be directly compared via the precision chips.
    """
    sources = []
    for ctx, group, group_title, expected_tasks in RLM_KVZIP_VARIANTS:
        for kind_id, precision, run_dir_name, budgets, provenance in [
            (
                "rlm-kvzip",
                "RLM + KVzip · Qwen3-4B-Instruct-2507",
                f"rlm/loft{ctx}_full_batch",
                ["256", "512", "1024", "2048", "4096"],
                "RLM keeps the document out of the prompt; each llm_query sub-call is "
                "compressed in-process via kvpress's KVzipPress, budgeted the same way "
                "as the plain-KVzip sweep above (converted to a per-sub-call token cap).",
            ),
            (
                "rlm-nopress",
                "RLM (no press, ablation) · Qwen3-4B-Instruct-2507",
                f"rlm/loft{ctx}_nopress_ablation",
                ["No press"],
                "Ablation: identical RLM harness and sub-call backend, but press=None so "
                "sub-calls are never compressed -- isolates RLM's search strategy from "
                "KVzip's contribution.",
            ),
        ]:
            directory = EVAL / run_dir_name
            if not directory.is_dir():
                continue
            grouped = {}
            newest = 0.0
            for metric_file in sorted(directory.glob("*/metrics.json")):
                match = RLM_RUN_DIR_RE.match(metric_file.parent.name)
                if not match or match.group("press") != ("kvzip" if kind_id == "rlm-kvzip" else "no_press"):
                    continue
                task = match.group("task")
                try:
                    metrics = json.loads(metric_file.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                newest = max(newest, metric_file.stat().st_mtime)
                if kind_id == "rlm-kvzip":
                    value, unit = match.group("value"), match.group("unit")
                    budget = f"{float(value) * (1024 if unit == 'GB' else 1):g}"
                else:
                    budget = "No press"
                runtime = metrics.get("runtime", {})
                run = {
                    "scores": score_fields(metrics, "loft"),
                    "samples": int(metrics.get("num_samples", 0)),
                    "retained_tokens": runtime.get("average_sub_retained_context_tokens"),
                    "original_tokens": runtime.get("average_sub_context_tokens"),
                    "retained_gb": None,
                    "compression": runtime.get("average_sub_compression_ratio"),
                    "prediction_url": None,
                    "prediction_preview": [],
                }
                grouped.setdefault(task, {})[budget] = run

            expected = set(budgets)
            complete = {
                task: {budget: runs[budget] for budget in budgets}
                for task, runs in sorted(grouped.items())
                if task in expected_tasks and expected.issubset(runs)
            }
            incomplete = sorted(task for task in expected_tasks if not expected.issubset(grouped.get(task, {})))
            metric_keys = sorted({key for runs in complete.values() for run in runs.values() for key in run["scores"]})
            sources.append(
                {
                    "id": f"{kind_id}-loft{ctx}",
                    "title": precision,
                    "group": group,
                    "group_title": group_title,
                    "precision": precision,
                    "model": "qwen3-4b-instruct-2507",
                    "model_title": "Qwen3-4B-Instruct-2507",
                    "kind": "loft",
                    "budgets": budgets,
                    "provenance": provenance,
                    "budget_provenance": {},
                    "tasks": complete,
                    "metrics": metric_keys,
                    "excluded": incomplete,
                    "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
                }
            )
    return sources


AUTOSUB_RUN_DIR_RE = re.compile(
    r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-kvzip(?P<value>[0-9.]+)(?P<unit>MB|GB)__sub\d+__autosub(?P<target>[0-9.]+)$"
)
AUTOSUB_NOPRESS_RUN_DIR_RE = re.compile(
    r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-no_press(?P<value>[0-9.]+)(?P<unit>MB|GB)__sub\d+__autosub(?P<target>[0-9.]+)$"
)


def rlm_autosub_sources():
    """Collect today's budget-derived RLM + KVzip and no-press matrices.

    Unlike the older 5-budget sweep, these runs use 100 MB through 2 GB.  Keep
    partially completed tasks visible so the dashboard reflects live progress
    instead of hiding an entire task until every budget finishes.
    """
    expected_tasks = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]
    budgets = ["100", "256", "400", "512", "750", "1024", "2048"]
    variants = [
        (
            "rlm/loft128k_autosub_target0.5",
            AUTOSUB_RUN_DIR_RE,
            "rlm-autosub-today-kvzip-loft128k",
            "RLM + KVzip, auto-chunk target {target} · Qwen3-4B-Instruct-2507",
            "Sub-call chunk size is derived from the memory budget with target compression "
            "{target}; realized KV removal is reported from each completed run.",
        ),
        (
            "rlm/loft128k_autosub_nopress_target0.0",
            AUTOSUB_NOPRESS_RUN_DIR_RE,
            "rlm-autosub-today-nopress-loft128k",
            "RLM only, auto-chunk no-press · Qwen3-4B-Instruct-2507",
            "Matched RLM auto-chunk ablation: the same budget-derived sub-call sizes, "
            "but with press=None so no KV compression is applied.",
        ),
    ]
    sources = []
    for run_dir_name, pattern, source_id, title_template, provenance_template in variants:
        directory = EVAL / run_dir_name
        if not directory.is_dir():
            continue
        grouped, newest, target_seen = {}, 0.0, None
        for metric_file in sorted(directory.glob("*/metrics.json")):
            match = pattern.match(metric_file.parent.name)
            if not match:
                continue
            try:
                metrics = json.loads(metric_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            task = match.group("task")
            target_seen = match.group("target")
            newest = max(newest, metric_file.stat().st_mtime)
            value, unit = match.group("value"), match.group("unit")
            budget = f"{float(value) * (1024 if unit == 'GB' else 1):g}"
            runtime = metrics.get("runtime", {})
            grouped.setdefault(task, {})[budget] = {
                "scores": score_fields(metrics, "loft"),
                "samples": int(metrics.get("num_samples", 0)),
                "retained_tokens": runtime.get("average_sub_retained_context_tokens"),
                "original_tokens": runtime.get("average_sub_context_tokens"),
                "retained_gb": None,
                "compression": runtime.get("average_sub_compression_ratio"),
                "prediction_url": None,
                "prediction_preview": [],
            }
        tasks = {
            task: {budget: runs[budget] for budget in budgets if budget in runs}
            for task, runs in sorted(grouped.items())
            if task in expected_tasks
        }
        incomplete = sorted(task for task in expected_tasks if set(budgets) - set(grouped.get(task, {})))
        metric_keys = sorted({key for runs in tasks.values() for run in runs.values() for key in run["scores"]})
        target = target_seen or "0.5"
        title = title_template.format(target=target)
        sources.append(
            {
                "id": source_id,
                "title": title,
                "group": "loft128k",
                "group_title": "LOFT 128K",
                "precision": title,
                "model": "qwen3-4b-instruct-2507",
                "model_title": "Qwen3-4B-Instruct-2507",
                "kind": "loft",
                "budgets": budgets,
                "provenance": provenance_template.format(target=target),
                "budget_provenance": {},
                "tasks": tasks,
                "metrics": metric_keys,
                "excluded": incomplete,
                "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
            }
        )
    return sources


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
    synthetic_32k_prefix = "new_synthetic_kv_32k__"
    if kind == "synthetic" and name.startswith(synthetic_32k_prefix):
        prefix = synthetic_32k_prefix
    if not name.startswith(prefix) or "__--home" not in name:
        return None
    task = name[len(prefix) :].split("__--home", 1)[0]
    return task


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
    # The full set is available via prediction_url -- the dashboard fetches
    # and renders it client-side on demand rather than embedding every row
    # of every task/budget into the page (which bloated index.html to ~20MB).
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
    base = BENCH if source.get("base") == "benchmark_artifacts" else EVAL
    for relative_path in source.get("paths", [source.get("path")]):
        directory = base / relative_path
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
                if (
                    budget in source["budgets"]
                    and (source["kind"] == "synthetic" or source.get("publish_predictions"))
                )
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
        "model": source.get("model", ""),
        "model_title": source.get("model_title", source.get("precision", "")),
        "provenance": source.get("provenance", ""),
        "budget_provenance": source.get("budget_provenance", {}),
        "tasks": complete,
        "metrics": metrics,
        "excluded": incomplete,
        "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
    }


def build() -> None:
    datasets = [collect(source) for source in SOURCES] + rlm_sources() + rlm_kvzip_sources() + rlm_autosub_sources()
    # Remove downloads from budgets that are no longer published (for example,
    # an interrupted 8 GB run intentionally excluded from the dashboard).
    published = {
        run["prediction_url"]
        for dataset in datasets
        for runs in dataset["tasks"].values()
        for run in runs.values()
        if run["prediction_url"]
    }
    source_ids = {source["id"] for source in SOURCES}
    for file in DOWNLOADS.glob("*.csv"):
        if any(file.name.startswith(f"{source_id}-") for source_id in source_ids):
            relative = f"downloads/{file.name}"
            if relative not in published:
                file.unlink()
    template = (Path(__file__).with_name("template.html")).read_text()
    output = template.replace("__DASHBOARD_DATA__", json.dumps(datasets, separators=(",", ":")))
    OUT.write_text(output)
    print(f"Wrote {OUT}")
    for dataset in datasets:
        print(f"{dataset['title']}: {len(dataset['tasks'])} complete, {len(dataset['excluded'])} excluded")


if __name__ == "__main__":
    build()
