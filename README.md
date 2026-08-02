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
