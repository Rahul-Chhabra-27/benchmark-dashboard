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



AUTOSUB_RUN_DIR_RE = re.compile(
    r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-kvzip(?P<value>[0-9.]+)(?P<unit>MB|GB)__sub\d+__autosub(?P<target>[0-9.]+)$"
)
AUTOSUB_NOPRESS_RUN_DIR_RE = re.compile(
    r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-no_press(?P<value>[0-9.]+)(?P<unit>MB|GB)__sub\d+__autosub(?P<target>[0-9.]+)$"
)


RLM_NOPRESS_RUN_DIR_RE = re.compile(r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-no_press(?:[0-9.]*)(?:MB|GB)?$")

RLM_UNBOUNDED_VARIANTS = [
    ("32k", "loft32k", "LOFT 32K", ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"]),
    ("128k", "loft128k", "LOFT 128K", ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]),
]


def rlm_unbounded_nopress_sources():
    """RLM with a fixed 32,000-char sub-call chunk and press=None ("Unbounded
    RLM"): the pre-budget-derived-sizing no-press ablation. Kept separately
    from the pressed "RLM + KVzip" family (removed -- its chunk size was
    fixed and budget-independent, which is what made it worth dropping) and
    from rlm_autosub_sources()'s budget-derived no-press arm below, which
    sweeps real budgets rather than using one fixed, unbounded chunk size.
    """
    sources = []
    for ctx, group, group_title, expected_tasks in RLM_UNBOUNDED_VARIANTS:
        run_dir_name = f"rlm/loft{ctx}_nopress_ablation"
        directory = EVAL / run_dir_name
        if not directory.is_dir():
            continue
        grouped = {}
        newest = 0.0
        for metric_file in sorted(directory.glob("*/metrics.json")):
            match = RLM_NOPRESS_RUN_DIR_RE.match(metric_file.parent.name)
            if not match:
                continue
            task = match.group("task")
            try:
                metrics = json.loads(metric_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            newest = max(newest, metric_file.stat().st_mtime)
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
            grouped.setdefault(task, {})["Unbounded RLM"] = run

        expected = {"Unbounded RLM"}
        complete = {
            task: runs for task, runs in sorted(grouped.items())
            if task in expected_tasks and expected.issubset(runs)
        }
        incomplete = sorted(task for task in expected_tasks if not expected.issubset(grouped.get(task, {})))
        metric_keys = sorted({key for runs in complete.values() for run in runs.values() for key in run["scores"]})
        precision = "RLM (no press, ablation) · Qwen3-4B-Instruct-2507"
        sources.append(
            {
                "id": f"rlm-nopress-{ctx}",
                "title": precision,
                "group": group,
                "group_title": group_title,
                "precision": precision,
                "model": "qwen3-4b-instruct-2507",
                "model_title": "Qwen3-4B-Instruct-2507",
                "kind": "loft",
                "budgets": ["Unbounded RLM"],
                "provenance": (
                    "Ablation: identical RLM harness and sub-call backend, but press=None so "
                    "sub-calls are never compressed -- isolates RLM's search strategy from "
                    "KVzip's contribution. Sub-call chunk is a fixed 32,000-char size, not "
                    "derived from a memory budget."
                ),
                "budget_provenance": {},
                "tasks": complete,
                "metrics": metric_keys,
                "excluded": incomplete,
                "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
            }
        )
    return sources


def rlm_autosub_sources():
    """Collect today's budget-derived RLM + KVzip and no-press matrices.

    Unlike the older 5-budget sweep, these runs use 100 MB through 2 GB.  Keep
    partially completed tasks visible so the dashboard reflects live progress
    instead of hiding an entire task until every budget finishes.
    """
    expected_tasks = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]
    budgets = ["256", "512", "750", "1024", "2048"]
    variants = [
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


FIXEDGRID_RUN_DIR_RE = re.compile(
    r"^loft__(?P<task>.+?)_{3}home.*?__rlm__kvzip-(?P<press>kvzip|no_press)(?P<tokens>[0-9]+)tokens__autosubx(?P<factor>[0-9]+)__fixed$"
)


def rlm_fixedgrid_sources():
    """Collect the B x F fixed-chunk grid: for budget B and factor F, a sub-call
    reads N=B*F tokens and KVzip prunes back to B. F=1 cells run no_press as the
    uncompressed control. Each (B, F) pair becomes its own budget chip, e.g.
    "256MB x1", since the dashboard's budget axis is one-dimensional.
    """
    expected_tasks = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]
    token_to_mb = {1736: 256, 3472: 512, 5086: 750, 6944: 1024, 13888: 2048, 27126: 4096}
    variants = [
        ("rlm/loft128k_fixedgrid_full", "hotpotqa_128k only, 100% data"),
        ("rlm/loft128k_fixedgrid_50pct", "nq/musique/qampari/quest, seeded 50% sample (seed 42)"),
    ]
    grouped, newest = {}, 0.0
    for run_dir_name, _ in variants:
        directory = EVAL / run_dir_name
        if not directory.is_dir():
            continue
        for metric_file in sorted(directory.glob("*/metrics.json")):
            match = FIXEDGRID_RUN_DIR_RE.match(metric_file.parent.name)
            if not match:
                continue
            try:
                metrics = json.loads(metric_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            task = match.group("task")
            tokens = int(match.group("tokens"))
            factor = int(match.group("factor"))
            if factor == 16:
                # F=16 is consistently the weakest cell everywhere it's been
                # tested (well below each task's own best), and it's also
                # where every permanently-impossible cell lives (chunk bigger
                # than the document itself). Dropped from the dashboard grid
                # rather than kept as a column of mostly gaps and weak scores.
                continue
            mb = token_to_mb.get(tokens)
            budget = f"{mb}MB x{factor}" if mb else f"{tokens}tok x{factor}"
            newest = max(newest, metric_file.stat().st_mtime)
            runtime = metrics.get("runtime", {})
            scores = score_fields(metrics, "loft")
            # RLM-only, not the shared canonical scorer: a looser substring-
            # containment check ("does a gold answer string appear anywhere in
            # the prediction"). Kept as its own field, never folded into
            # primary_score, so it's never silently compared against plain
            # KVzip's em/subspan_em/f1/coverage as if it were the same metric.
            if runtime.get("progress_match_loose") is not None:
                scores["progress_match_loose"] = float(runtime["progress_match_loose"]) * 100
            grouped.setdefault(task, {})[budget] = {
                "scores": scores,
                "samples": int(metrics.get("num_samples", 0)),
                "retained_tokens": runtime.get("average_sub_retained_context_tokens"),
                "original_tokens": runtime.get("average_sub_context_tokens"),
                "retained_gb": None,
                "compression": runtime.get("average_sub_compression_ratio"),
                "prediction_url": None,
                "prediction_preview": [],
                "fraction": 1.0 if task == "hotpotqa_128k" else 0.5,
            }
    if not grouped:
        return []
    all_budgets = sorted(
        {b for runs in grouped.values() for b in runs},
        key=lambda b: (int(b.split("MB")[0]) if "MB" in b else 0, int(b.split("x")[-1])),
    )
    tasks = {task: runs for task, runs in sorted(grouped.items()) if task in expected_tasks}
    incomplete = sorted(task for task in expected_tasks if set(all_budgets) - set(grouped.get(task, {})))
    metric_keys = sorted({key for runs in tasks.values() for run in runs.values() for key in run["scores"]})
    return [
        {
            "id": "rlm-fixedgrid-loft128k",
            "title": "RLM + KVzip, fixed-chunk B x F grid · Qwen3-4B-Instruct-2507",
            "group": "loft128k",
            "group_title": "LOFT 128K",
            "precision": "RLM + KVzip, fixed-chunk B x F grid · Qwen3-4B-Instruct-2507",
            "model": "qwen3-4b-instruct-2507",
            "model_title": "Qwen3-4B-Instruct-2507",
            "kind": "loft",
            "budgets": all_budgets,
            # quest_128k's 2048MB x8 cell hit a pathological multi-hour hang (many
            # RLM sub-call retries on one hard example) and was accepted as a gap
            # rather than waited out further -- exempted here so the chip enables
            # without the other 4 tasks' completed data waiting on it forever.
            # Per-(task, budget) exemption, not the source-wide leniency flag,
            # so genuinely still-computing budgets (the 4GB row) stay disabled.
            #
            # hotpotqa_128k's 4096MB x8 cell is a TEMPORARY exemption, not a
            # permanent gap: its post-widenfix-clamp rerun crashed on a GPU
            # preflight check (47.3GiB then 39.5GiB free vs ~48GiB needed --
            # transient node contention, not a real failure) and has been
            # resubmitted (job 304903, busy nodes excluded). Exempted so the
            # rest of the 4GB row (nq/musique/qampari/quest, all already
            # complete after the fix) isn't held back waiting on a rerun
            # that's actively in flight. Remove this entry once it lands.
            "gap_exempt": {"quest_128k": ["2048MB x8"], "hotpotqa_128k": ["4096MB x8"]},
            "provenance": "Each budget chip is one (B, F) cell: a sub-call reads N=B*F tokens of the "
            "document, then KVzip prunes it back to B tokens. x1 cells run no_press as the "
            "uncompressed control at that budget. hotpotqa_128k runs at 100% data; the other "
            "four tasks run a reproducible 50% sample (seed 42). hotpotqa_128k's 4096MB x8 "
            "cell is a rerun in progress (GPU preflight crash, resubmitted) -- expect it to "
            "appear within this row shortly.",
            "budget_provenance": {},
            "tasks": tasks,
            "metrics": metric_keys,
            "excluded": incomplete,
            "updated": datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "—",
        }
    ]


RLM_NEW_CSV = Path(__file__).with_name("data") / "rlm_new_loft128k.csv"
RLM_NEW_EXPECTED_TASKS = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]


def rlm_new_loft128k_sources():
    """The September 5-7 RLM campaign (source commit c448c97), read from a CSV
    committed alongside this script rather than from a run tree.

    Unlike every other source here, the raw run directories for this campaign
    are not reachable from this repo, so the exported per-cell metrics CSV is
    the record of truth. That also makes this source regenerable from a clean
    checkout on any machine.

    Each budget chip is one (logical KV budget, chunk factor) cell, labelled in
    GB rather than the fixed-chunk grid's MB so the two campaigns never share a
    chip: their budgets are quoted under conversions this repo cannot verify
    against each other, and silently merging them would compare cells that only
    look alike.
    """
    if not RLM_NEW_CSV.is_file():
        return []
    grouped = {}
    with RLM_NEW_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["dataset"]
            gb = float(row["logical_kv_budget_gb"])
            gb_label = int(gb) if gb == int(gb) else gb
            budget = f"{gb_label}GB x{int(row['chunk_factor'])}"
            factor = float(row["effective_compression_factor"])
            grouped.setdefault(task, {})[budget] = {
                "scores": {
                    "em": float(row["em"]) * 100,
                    "subspan_em": float(row["subspan_em"]) * 100,
                    "f1": float(row["f1"]) * 100,
                    # LOFT's primary_score is f1 for these tasks, matching the
                    # other LOFT sources on this dashboard.
                    "primary_score": float(row["f1"]) * 100,
                    # How much of the document the RLM actually read. This is an
                    # RLM sizing statistic, NOT the LOFT "coverage" scorer, so it
                    # gets its own key and is never folded into primary_score.
                    "document_coverage": float(row["document_coverage_fraction"]) * 100,
                },
                "samples": int(row["num_samples"]),
                "retained_tokens": int(row["kv_budget_tokens"]),
                "original_tokens": int(row["chunk_tokens"]),
                "retained_gb": gb,
                # Fraction of the chunk's KV removed by the press; the x1 cells
                # run no_press, so nothing is removed.
                "compression": 0.0 if factor <= 1 else 1 - 1 / factor,
                "prediction_url": None,
                "prediction_preview": [],
                # Not a subsample: every cell is the same 55-example set, so
                # these runs opt out of the sample-fraction filter entirely.
                "fraction": None,
            }
    if not grouped:
        return []
    tasks = {task: runs for task, runs in sorted(grouped.items())}
    all_budgets = sorted(
        {b for runs in tasks.values() for b in runs},
        key=lambda b: (float(b.split("GB")[0]), int(b.split("x")[-1])),
    )
    return [
        {
            "id": "rlm-new-loft128k",
            "title": "RLM(new) · post-c448c97 · Qwen3-4B-Instruct-2507",
            "group": "loft128k",
            "group_title": "LOFT 128K",
            "precision": "RLM(new), post-c448c97 · Qwen3-4B-Instruct-2507",
            "model": "qwen3-4b-instruct-2507",
            "model_title": "Qwen3-4B-Instruct-2507",
            "kind": "loft",
            "budgets": all_budgets,
            "provenance": "RLM(new), source commit c448c97 · 1,024-token scratchpad · search_k=0 · "
            "each chip is one (logical KV budget, chunk factor) cell: a sub-call reads "
            "budget x factor tokens, then KVzip prunes back to the budget; x1 cells run "
            "no_press as the uncompressed control. Logical budgets are decimal GB at "
            "147,456 bytes/token -- simulated KV retention, not measured GPU memory. "
            "16 example-runs across 10 cells hit a GPU-fit error and 14 predictions were "
            "unfinished; both are included in these scores (see RLM-new-loft128k.csv).",
            "budget_provenance": {},
            "tasks": tasks,
            # "coverage" is declared but never populated: no cell in this campaign
            # computed LOFT's coverage scorer. It stays listed so that selecting
            # this source alongside the other LOFT-128K sources does not drop
            # coverage out of the shared metric dropdown for all of them.
            "metrics": ["coverage", "document_coverage", "em", "f1", "primary_score", "subspan_em"],
            "excluded": [t for t in RLM_NEW_EXPECTED_TASKS if t not in tasks],
            "updated": "2026-09-07 15:45",
        }
    ]


STRUCTURED_RLM_CSV = Path(__file__).with_name("data") / "structured_rlm_loft128k.csv"
STRUCTURED_GEOMETRY_CSV = Path(__file__).with_name("data") / "structured_geometry_loft128k.csv"

# One entry per source id in the CSV's `source` column. Each is published as its
# OWN dashboard source rather than as chips on a shared one, because the four
# differ in which rows they scored: 55 per subset, the 51-row hard slice, and two
# pooled-only row sets. Merging them would put scores with different denominators
# under one task label, which is exactly the comparison the campaign write-up
# spends its caveats warning against.
STRUCTURED_RLM_SOURCES = {
    # Listed FIRST so its chip leads the group and setGroup preselects it: the
    # template puts 'No compression'/'unbounded' first everywhere else, and this
    # is the same idea under a name budgetSortKey has no rule for.
    "baseline": {
        "title": "Baseline · whole 128k in-window, no RLM",
        "precision": "Vanilla (no RLM) · Qwen3-4B-Instruct-2507",
        "blurb": "The no-recursion control: the entire LOFT document in the model's "
        "window, answered in one call. Qwen3-4B-2507 is natively 262,144 and was "
        "served at --max-model-len 139264, so nothing was truncated "
        "(context_retained = 1.0) and there is no context wall at 128k for the RLM "
        "to be worth paying for -- vanilla outscores every RLM arm here. The "
        "defensible RLM result is the cost axis beside it: ~2,000 peak tokens "
        "against ~120,000, for 71-82% of vanilla's score. DIFFERENT ROW SET: n=110 "
        "(LOFT dev 10 + test 100) from the 2026-08-27 arms 1-3 campaign, where every "
        "other source here is 55 test rows. Read it as a reference line, NOT as a "
        "paired comparison -- no significance test on this dashboard crosses that "
        "boundary.",
    },
    "full55": {
        "title": "Structured vs agentic · full 55 rows",
        "precision": "Structured RLM · full 55 · Qwen3-4B-Instruct-2507",
        "blurb": "All 55 rows of each subset, nothing selected on. Arms A/B are the "
        "agentic REPL; B' is the same BM25 chunks read by the fixed map->reduce "
        "pipeline. Ablations 1 and 2 each remove one revision-2 difference.",
    },
    "hardslice": {
        "title": "Structured vs agentic · hard slice (n=51)",
        "precision": "Structured RLM · hard slice · Qwen3-4B-Instruct-2507",
        "blurb": "The 51 rows where arm B retrieved the gold answer string and still "
        "answered wrongly (nq 22 of 55, hotpotqa 29 of 29+). SELECTED ON ARM B BEING "
        "WRONG, so B' -- which shares B's chunks -- is disadvantaged by construction "
        "and C is not; arm A is reported beside it as the unselected comparator.",
    },
    "rev2_2x2": {
        "title": "Revision-2 ablation · clustering x answer shape",
        "precision": "Structured RLM · rev-2 2x2 pooled · Qwen3-4B-Instruct-2507",
        "blurb": "The 2x2 that separates revision 2's two confounded changes, pooled "
        "over both subsets (n=110). Clustering is worth +0.036 subspan at both prompt "
        "settings; requiring the answer shape costs -0.036 subspan and buys em. "
        "Neither is significant alone at this n.",
    },
    "transfer": {
        "title": "Selector chunks read by the agentic root",
        "precision": "Structured RLM · chunk transfer pooled · Qwen3-4B-Instruct-2507",
        "blurb": "The hard slice's 51 rows, pooled. The same windows are worth 0.4706 "
        "subspan to the fixed pipeline (C) and 0.2353 to the agentic loop (run 4 "
        "capped) -- the selector's advantage is a property of the pipeline, not the "
        "chunks. Uncapped run 4 re-searched 21.4x per example against the frozen "
        "control's 1.7; the capped rows are the honest version.",
    },
}



STRUCTURED_GEOMETRY_SOURCES = {
    "geometry": {
        "title": "Chunk geometry \u00b7 single-answer subsets",
        "precision": "Structured RLM \u00b7 geometry sweep \u00b7 Qwen3-4B-Instruct-2507",
        "metric": "subspan_em",
        "group": "structured-geometry",
        "group_title": "Structured RLM \u00b7 chunk geometry (LOFT 128K)",
        "blurb": "What the map\u2192reduce pipeline reads, swept for the first time. Every "
        "structured arm before this ran at one setting -- 2000-character windows, 400 of "
        "overlap, 5 of them -- which is 512 tokens at 20% overlap, the retrieval optimum "
        "the chunking literature lands on. What had never been chosen is the READ size: "
        "--search-window set the BM25 unit and the reader's payload as one number. "
        "Chips are CONFIGURATIONS, not KV budgets; no arm here runs a press, so the "
        "KV-removed columns stay empty and \"retained tokens\" is mean peak ROOT context.",
    },
    "geometry_mv": {
        "title": "Chunk geometry \u00b7 qampari (multi-answer)",
        "precision": "Structured RLM \u00b7 geometry sweep \u00b7 Qwen3-4B-Instruct-2507",
        "metric": "coverage",
        "group": "structured-geometry-mv",
        "group_title": "Structured RLM \u00b7 chunk geometry, multi-answer (LOFT 128K)",
        "blurb": "qampari in its OWN group because it is scored on `coverage`, not "
        "`subspan_em`. Its gold is a list of entities scattered across passages, and LOFT's "
        "multi-value subspan metric demands every one be matched, which pins it near zero; "
        "coverage is the fraction of the list recovered. Kept apart from the single-answer "
        "subsets so the two metrics can never be read down one column -- within a group the "
        "metric dropdown is the intersection across selected sources.",
    },
}


def structured_geometry_sources():
    """The 2026-09-09 chunk-geometry campaign (source commit d0a1ad6).

    CSV-backed for the same reason every RLM source here is: the run trees live on
    the compute host. 80 cells, 4,400 examples, zero errors; the chips published are
    the ten configurations that exist for EVERY subset, out of the twenty run.

    Three things worth knowing before reading the numbers:

    * **`BARE query` is a retrieval FIX, not a tuning knob.** The structured arm had
      been handing BM25 LOFT's whole task string -- a page of "print the TITLE and
      ID... format the answers into a list" -- so windows were scored on boilerplate
      that occurs in every one of them. Asking it the question instead moves recall of
      the gold answer at depth 2 from 0.00 to 0.95 on nq. Runs before this campaign,
      including the `structured-rlm-*` sources on this dashboard, all used the task
      string.
    * **`oracle chunks` is a ceiling, not an arm.** Its windows are the ones that
      provably contain a gold answer, so retrieval is perfect by construction and
      whatever is missing is the reader.
    * **qampari's metric is `coverage`,** which is why it is a separate group.

    `primary_score` is published as whichever of subspan_em / coverage is that
    subset's LOFT headline, matching what evaluation/compare.py now selects.
    """
    if not STRUCTURED_GEOMETRY_CSV.is_file():
        return []
    grouped = {}
    with STRUCTURED_GEOMETRY_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"]) * 100
            scores = {row["metric"]: score, "primary_score": score}
            if row["em"]:
                scores["em"] = float(row["em"]) * 100
            peak = row["peak_context_tokens"]
            grouped.setdefault(row["source"], {}).setdefault(row["task"], {})[row["arm"]] = {
                "scores": scores,
                "samples": int(row["samples"]),
                "retained_tokens": int(peak) if peak else None,
                "original_tokens": None,
                "retained_gb": None,
                "compression": None,
                "prediction_url": None,
                "prediction_preview": [],
                "fraction": None,
            }
    datasets = []
    for source_id, meta in STRUCTURED_GEOMETRY_SOURCES.items():
        tasks = grouped.get(source_id)
        if not tasks:
            continue
        arms = []
        for runs in tasks.values():
            for arm in runs:
                if arm not in arms:
                    arms.append(arm)
        # A chip is greyed out unless every task of every source declaring it has a
        # run, so assert rather than trust: a missing cell disables the chip for the
        # whole group and the loss is silent.
        for task, runs in tasks.items():
            missing = [arm for arm in arms if arm not in runs]
            assert not missing, f"{source_id}/{task} is missing arms {missing}"
        metrics = ["primary_score", meta["metric"]]
        if any("em" in run["scores"] for runs in tasks.values() for run in runs.values()):
            metrics.append("em")
        datasets.append(
            {
                "id": f"structured-geometry-{source_id.replace('_', '-')}",
                "title": meta["title"],
                "group": meta["group"],
                "group_title": meta["group_title"],
                "precision": meta["precision"],
                "model": "qwen3-4b-instruct-2507",
                "model_title": "Qwen3-4B-Instruct-2507",
                "kind": "loft",
                "budgets": arms,
                "provenance": meta["blurb"]
                + " 55 test rows per subset, one served vLLM at temperature 0.0, 0 errors. "
                "Source commit d0a1ad6; rebuilt from data/structured_geometry_loft128k.csv. "
                "NOT a paired comparison with the n=110 vanilla reference line on the "
                "Structured RLM tab.",
                "budget_provenance": {},
                "tasks": dict(sorted(tasks.items())),
                "metrics": sorted(metrics),
                "excluded": [],
                "updated": "2026-09-09 21:00",
            }
        )
    return datasets

def structured_rlm_sources():
    """The 2026-09-08 structured-chunks campaign (source commit 24f54e4).

    Like `rlm_new_loft128k_sources`, and for the same reason, this is rebuilt
    from a CSV committed beside this script: the run trees live on the compute
    host and are not reachable from this repo. Unlike that source, the record of
    truth behind the CSV is the campaign write-up
    (`evaluation/rlm/STRUCTURED_RESULTS.md` in the benchmark repo), whose own
    tables were re-derived from `metrics.json`.

    Two things this campaign does NOT carry, and why they are absent rather than
    zero-filled:

    * **No f1, and therefore no LOFT `primary_score`.** The write-up scores
      `subspan_em` (LOFT's own headline for these tasks) and `em`; f1 was never
      computed. `primary_score` is published here AS `subspan_em` so the metric
      selector has a working default, and the provenance string says so. This is
      why the campaign gets its own group: within a group the metric dropdown is
      the intersection across selected sources, so declaring a short metric list
      next to the LOFT-128K sources would strip f1 and coverage from all of them.
    * **No compression figures.** No arm here runs a press -- the axis the
      campaign measures is peak ROOT context, which is published in the
      `retained_tokens` column. `compression` is left None so the KV-removed
      columns render as "--" instead of implying an uncompressed run was
      measured at 0%.
    """
    if not STRUCTURED_RLM_CSV.is_file():
        return []
    grouped = {}
    with STRUCTURED_RLM_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scores = {"subspan_em": float(row["subspan_em"]) * 100}
            # Published as the default metric, NOT as a separate measurement:
            # see the docstring. Kept in lockstep with subspan_em.
            scores["primary_score"] = scores["subspan_em"]
            if row["em"]:
                scores["em"] = float(row["em"]) * 100
            peak = row["peak_context_tokens"]
            grouped.setdefault(row["source"], {}).setdefault(row["task"], {})[row["arm"]] = {
                "scores": scores,
                "samples": int(row["samples"]),
                # Mean peak ROOT context, the axis compare.py puts against
                # KVPress's retained tokens. Absent for the arms whose peak was
                # not measured on that row set. The baseline's figure is the
                # document MEASURED with the Qwen3-4B-Instruct-2507 tokenizer
                # (118,606 / 121,229 tokens plus the question), not a number read
                # back from that run's metrics.json -- its run tree is not
                # reachable from here either.
                "retained_tokens": int(peak) if peak else None,
                "original_tokens": None,
                "retained_gb": None,
                "compression": None,
                "prediction_url": None,
                "prediction_preview": [],
                "fraction": None,
            }
    datasets = []
    for source_id, meta in STRUCTURED_RLM_SOURCES.items():
        tasks = grouped.get(source_id)
        if not tasks:
            continue
        # Chip order is the CSV's order, which is the write-up's: arms first,
        # then the ablations that were run against them. budgetSortKey has no
        # rule for these labels, so the template's sort leaves this order alone.
        arms = []
        for runs in tasks.values():
            for arm in runs:
                if arm not in arms:
                    arms.append(arm)
        # A chip is greyed out unless every task of every source declaring it has
        # a run there. Each source here holds exactly one row set, so its tasks
        # all carry the same arms and no gap_exempt is needed -- assert it rather
        # than trusting the CSV, since a missing cell would silently disable the
        # chip for the whole group.
        for task, runs in tasks.items():
            missing = [arm for arm in arms if arm not in runs]
            assert not missing, f"{source_id}/{task} is missing arms {missing}"
        datasets.append(
            {
                "id": f"structured-rlm-{source_id.replace('_', '-')}",
                "title": meta["title"],
                "group": "structured-rlm",
                "group_title": "Structured RLM · LOFT 128K",
                "precision": meta["precision"],
                "model": "qwen3-4b-instruct-2507",
                "model_title": "Qwen3-4B-Instruct-2507",
                "kind": "loft",
                "budgets": arms,
                "provenance": meta["blurb"]
                + " Source commit 24f54e4, one served vLLM at temperature 0.0, scored by "
                "the shared score_prediction_frame. Chips are ARMS, not KV budgets: no arm "
                "here runs a press, so the KV-removed columns stay empty. \"Retained "
                "tokens\" is mean peak ROOT context. primary_score IS subspan_em -- f1 was "
                "never computed for this campaign. Rebuilt from "
                "data/structured_rlm_loft128k.csv; the record of truth is "
                "evaluation/rlm/STRUCTURED_RESULTS.md in the benchmark repo.",
                "budget_provenance": {},
                "tasks": dict(sorted(tasks.items())),
                "metrics": ["em", "primary_score", "subspan_em"],
                "excluded": [],
                "updated": "2026-09-08 15:00",
            }
        )
    return datasets


def apply_postfix_infolab_overrides(datasets):
    """Overwrite/add loft128k-qwen3-4b-instruct budget entries from post-fix
    infolab CSV reruns, instead of publishing them as separate sources. The
    attention_patch key-restoration fix only changes behavior at >=1GB
    budgets (verified empirically against pre-fix LOFT-32K predictions);
    750MB is below that line and unaffected by the fix, but wasn't part of
    the original published matrix, so it's a new budget column, not an
    override. Each CSV row's own memory_budget/memory_budget_unit decides
    the target budget key -- no per-file hardcoding of values.
    """
    csv_paths = [
        ROOT / "kvpress/benchmark_artifacts/logs/running_log/loft128k_1gb_2gb_metrics.csv",
        ROOT / "kvpress/benchmark_artifacts/logs/running_log/loft128k_750mb_metrics.csv",
        ROOT / "kvpress/benchmark_artifacts/loft128k_750mb_4gb_metrics.csv",
    ]
    target = next((d for d in datasets if d["id"] == "loft128k-qwen3-4b-instruct"), None)
    if target is None:
        return
    import csv as csv_module

    newest = 0.0
    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        newest = max(newest, csv_path.stat().st_mtime)
        with open(csv_path, newline="") as f:
            for row in csv_module.DictReader(f):
                task = row["task"]
                if task not in target["tasks"]:
                    continue
                value, unit = float(row["memory_budget"]), row["memory_budget_unit"]
                budget = f"{value * (1024 if unit == 'GB' else 1):g}"
                scores = {
                    key: float(row[key]) * 100 for key in ("em", "subspan_em", "f1", "coverage") if row.get(key)
                }
                primary = scores.get("coverage", scores.get("f1", scores.get("subspan_em", scores.get("em"))))
                if primary is not None:
                    scores["primary_score"] = primary
                target["tasks"][task][budget] = {
                    "scores": scores,
                    "samples": int(float(row["num_samples"])),
                    "retained_tokens": float(row["average_retained_context_tokens"]),
                    "original_tokens": float(row["average_original_context_tokens"]),
                    "retained_gb": float(row["average_retained_kv_memory_gb"]),
                    "compression": float(row["average_compression_ratio"]),
                    "prediction_url": None,
                    "prediction_preview": [],
                    "fraction": 0.5,
                }
                if budget not in target["budgets"]:
                    target["budgets"].append(budget)
    if newest:
        target["updated"] = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
    # Recompute excluded now that budgets may have grown (e.g. 750MB added for
    # only 4 of 5 tasks): a task missing any declared budget is a known,
    # permanent gap here, not "still computing" -- the UI treats excluded
    # tasks as already-accounted-for rather than blocking the whole budget.
    target["excluded"] = sorted(
        task for task, runs in target["tasks"].items() if set(target["budgets"]) - set(runs)
    )
    # Distinct from rlm_fixedgrid_sources()'s "excluded" (every task, while its
    # campaign is still running -- not a permanent gap): this source's gaps
    # come from a one-off manual CSV addition, so a missing task is known and
    # final, not "still computing." Only sources built by this override get
    # the budget-chip leniency in the frontend.
    target["budget_gate_lenient"] = True


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
    datasets = (
        [collect(source) for source in SOURCES]
        + rlm_sources()
        + rlm_fixedgrid_sources()
        + rlm_new_loft128k_sources()
        + structured_rlm_sources()
    )
    apply_postfix_infolab_overrides(datasets)
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
