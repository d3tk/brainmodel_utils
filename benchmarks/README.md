# Neural backend benchmarks

Run the deterministic CPU smoke workload from the repository root:

```bash
python -m benchmarks.benchmark_neural_backends --suite smoke --backend cpu
```

`synthetic` is a medium correlated workload. The bundled `production` suite is a production-shaped synthetic scaling proxy (39 stimuli, 27,118 sources, 200 bootstraps, and five folds), not a claim to reproduce an exact experiment. Real workloads observed elsewhere have about 27,118–40,467 source dimensions; their target and trial shapes vary and their data are not included here. Every shape and repetition count can be overridden, for example:

```bash
python -m benchmarks.benchmark_neural_backends \
  --suite synthetic \
  --backend cpu \
  --stimuli 512 \
  --sources 1024 \
  --targets 128 \
  --trials 20 \
  --bootstrap-iters 10 \
  --splits 5 \
  --workers 4 \
  --output benchmark.json
```

Use `--backend cuda --workers 1` for a single-GPU run. The harness rejects multiple consistency workers on CUDA to avoid launching competing CUDA processes on one device.

A real production fixture can be supplied as an `.npz` file with `source` and `target` arrays. `source` may be either `stimuli x sources` or `trials x stimuli x sources`; `target` must be `trials x stimuli x targets`. The harness uses seeded train/test splits and records the fixture's actual shapes.

Each JSON result contains mapper phase timings, isolated warmed H2D/D2H timings for CUDA, end-to-end consistency timing, input and output summaries, CPU/thread/library metadata, process-tree peak RSS, self-plus-terminated-children CPU time, and best-effort whole-device peak VRAM and utilization. Whole-device GPU readings can include memory or activity from unrelated processes.
