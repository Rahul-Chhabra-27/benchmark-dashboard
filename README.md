# Benchmark dashboard

This standalone report visualizes completed LOFT and RULER32K sweeps. LOFT
shows true no-compression baselines where available and omits the older KVzip
1%-pruned reference. It also shows the positional configuration used by each
result.

LOFT provenance is displayed directly in the dashboard:

- All canonical non-quantized LOFT 32K results use YaRN-4 and full
  context (`max_context_length: null`), including the no-compression
  baseline and 256, 512, 1024, 2048, and 4096 budget labels (MiB).
- All canonical non-quantized LOFT 128K results use YaRN-4.

Regenerate after evaluations finish:

```bash
python3 benchmark_dashboard/generate_dashboard.py
```

Then open `benchmark_dashboard/index.html` in a browser, or serve the repository
with any static HTTP server.
