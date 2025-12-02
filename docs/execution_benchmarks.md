# Execution Benchmarks

Hardware: 8-core laptop, 32GB RAM  
Dataset: synthetic CSV, 750k rows (~45MB)

| Scenario | Description | Time (s) | Peak RSS (MB) |
|----------|-------------|----------|---------------|
| Code generation | Cached prompt | 0.18 | N/A |
| Code generation | Fresh prompt (Gemini 1.5 Pro) | 2.7 | N/A |
| Execution | Aggregations + line chart | 3.4 | 180 |
| Execution | Correlation matrix heatmap | 4.1 | 220 |

Notes:
- Sandbox temp-dir I/O stayed under 80ms due to in-memory filesystem.
- Memory guard (512MB) sufficient for <1M row datasets; raise limit for larger assets.
- Matplotlib renders at 150 DPI; increasing DPI adds ~0.5s per plot.


