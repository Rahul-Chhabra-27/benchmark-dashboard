# Benchmark dashboard

This standalone report visualizes completed LOFT, RULER32K, and Synthetic-KV
sweeps. LOFT
shows true no-compression baselines where available and omits the older KVzip
1%-pruned reference. It also shows the positional configuration used by each
result.

LOFT provenance is displayed directly in the dashboard:

- All canonical non-quantized LOFT 32K results use YaRN-4 and full
  context (`max_context_length: null`), including the no-compression
  baseline and 256, 512, 1024, 2048, and 4096 budget labels (MiB).
- All canonical non-quantized LOFT 128K results use YaRN-4.
- Synthetic-KV 64K includes the non-quantized BF16 true no-compression
  baseline and 512, 1024, 2048, 4096, and 8192 budgets. The matching
  Qwen3-8B-AWQ series is displayed only after its complete sweep is available.

Regenerate after evaluations finish:

```bash
python3 benchmark_dashboard/generate_dashboard.py
```

Then open `benchmark_dashboard/index.html` in a browser, or serve the repository
with any static HTTP server.
