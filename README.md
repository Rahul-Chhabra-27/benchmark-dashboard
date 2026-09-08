# Benchmark dashboard

This standalone report visualizes completed LOFT, RULER32K, Synthetic-KV, and
RLM sweeps. LOFT
shows true no-compression baselines where available and omits the older KVzip
1%-pruned reference. It also shows the positional configuration used by each
result.

LOFT provenance is displayed directly in the dashboard:

- All canonical non-quantized LOFT 32K results use YaRN-4 and full
  context (`max_context_length: null`), including the no-compression
  baseline and 256, 512, 1024, 2048, and 4096 budget labels (MiB).
- All canonical non-quantized LOFT 128K results use YaRN-4.
- Synthetic-KV 32K and 64K include the latest non-quantized and Qwen3-8B-AWQ
  sweeps through 4 GB. The interrupted 8 GB AWQ run is intentionally omitted.
- Completed Synthetic-KV configurations include deterministic prediction
  previews and downloadable CSV files containing every question, reference,
  prediction, and run statistic.

The `RLM(new)` LOFT-128K source (September 5-7 campaign, source commit
`c448c97`) is the one exception to the "regenerate from a run tree" rule: its
raw runs are not reachable from this repo, so it is rebuilt from
`data/rlm_new_loft128k.csv`, which is the record of truth for those cells. Its
budget chips are labelled in GB (`2GB x4`) rather than the fixed-chunk grid's MB
so the two RLM campaigns never share a chip -- their budget conversions are not
verified against each other here.

The `Structured RLM` tab (2026-09-08 campaign, source commit `24f54e4`) is the
second CSV-backed exception, rebuilt from `data/structured_rlm_loft128k.csv` by
`splice_structured_rlm.py`. Read it with three caveats:

- Its **chips are arms, not KV budgets**. No arm runs a press, so the KV-removed
  and KV-retained columns stay empty; "Retained tokens" is mean peak *root*
  context, the axis the campaign actually measures.
- It publishes **`primary_score` as `subspan_em`** -- LOFT's own headline for
  these tasks. f1 was never computed for this campaign, which is why the tab is
  its own group: the metric dropdown is the intersection across the sources
  selected in a group, so putting a short metric list beside the LOFT-128K
  sources would strip f1 and coverage from all of them.
- It is **four sources, not one**, because the campaign scored four different row
  sets: all 55 rows per subset, the 51-row hard slice, and two pooled-only
  ablations. They are kept apart so no single task label ever mixes denominators.
  The hard-slice sources are selected on arm B answering wrongly, which
  disadvantages the arm sharing B's chunks by construction -- the per-source
  provenance line says so in the dashboard itself.

The record of truth behind that CSV is `evaluation/rlm/STRUCTURED_RESULTS.md` in
the benchmark repo, whose tables were re-derived from `metrics.json`.

Regenerate after evaluations finish:

```bash
python3 benchmark_dashboard/generate_dashboard.py
```

Completed RLM checkpoints are imported automatically from
`kvpress/evaluation/results/rlm/**/*.jsonl`. They appear in an `RLM vs Vanilla`
tab with accuracy and completion-rate metrics; copy or sync the cluster output
there before rebuilding the site.

Then open `benchmark_dashboard/index.html` in a browser, or serve the repository
with any static HTTP server.
