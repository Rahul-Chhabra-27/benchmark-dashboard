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

The **Structured RLM** tab now has one explorer for the September 10 query-selection
study (84 runs), September 9 geometry study (117 completed runs), and five historical
comparison groups (34 summaries). The historical groups keep their different sample
sets separate. Eleven incomplete geometry settings and the unrun notes/holdout
answering experiments are explicitly identified.

- Filter by campaign, subset, selection method and read depth. The default compares
  corrected BM25 against exhaustive reading. “Question-independent” describes window
  selection only: the reader still receives the question.
- Chart score, mean tokens per query, or score versus tokens on a logarithmic token
  axis. The table includes per-query and whole-run token use, latency, and mean peak
  root context as distinct quantities. No structured arm is labelled as a KV budget.
- Qampari/Quest use answer coverage; the other subsets use subspan EM. Select a run
  for its configuration and prediction preview. All 233 runs with recovered
  checkpoints have CSV downloads with per-query tokens; the two vanilla reference
  rows have no usage measurement and display a dash.
- Notes indexes are complete but their answering results are pending. Historical
  structured-versus-agentic comparisons used different retrieval-query formulations
  and should not be treated as a clean pipeline comparison.

The explorer reads `data/structured_results.json`, with a flat export at
`data/structured_results.csv`. Its scripts and styles are in `assets/`. The original
26-source `DATA` payload remains intact for legacy tabs and archival consumers.

Update **only** this explorer safely from any checkout:

```bash
# Render template changes using committed data; standard-library Python only.
python3 update_structured_dashboard.py

# Import a freshly downloaded InfoLabs snapshot (requires PyYAML).
python3 update_structured_dashboard.py --snapshot /path/to/infolabs/raw

# Check sample counts, downloads, usage aggregates and exhaustive-read invariants.
python3 -m unittest test_structured_data.py
```

The import expects `qagnostic/results`, `structured_grid`, `loft128k_main`,
`structured_main`, `structured_main_r2`, `structured_ablation` and
`structured_ablation2` beneath the snapshot. It reads metrics/configs/checkpoints;
retries are deduplicated by example ID. Importing asserts that token sums match the
reported mean usage. Updating the explorer never invokes the older generator or
removes unrelated sources/downloads.

The **Graph order vs source order** campaign (BM25-graph vs source chunk order,
with vanilla, KVzip-only and base-RLM references on the same 55 test questions) is
added separately from an `export.json` written on the compute host:

```bash
python3 splice_graph_order.py /path/to/export.json
```

It replaces only the `graph-order` campaign, its rows in the flat CSV and
`downloads/structured/graph-order-*.csv`, and writes a summary to
`data/graph_order_loft128k.json`. A `--snapshot` import rebuilds the JSON from
scratch, so re-run the splice after one.

Direct link: [Structured RLM](https://rahul-chhabra-27.github.io/benchmark-dashboard/#structured).
Filter choices are retained in the URL for sharing.

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
