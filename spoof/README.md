# spoof/

Activation-spoofing and localized ("surgical") experiments on the sycophancy model organisms.

| Folder | What is in it | Where it runs |
|---|---|---|
| `pipeline/` | GPU jobs: data prep, spoof training, evaluation, scans, transfer, localized sweep. `common.py` holds shared paths and helpers; modules import each other flat. | pod |
| `pod/` | `pod_setup.sh` (repo + env bootstrap) and `run.sh` (single entry point for pipeline steps). | pod |
| `analysis/` | CPU analyses of synced results (statistics, localized-sweep analysis, reports). | laptop |
| `figures/` | pgfplots / matplotlib figure scripts; `pgf/style.tex` is the shared style. | laptop |

## On a pod

Copy this folder to `/workspace/spoof` and the data to `/workspace/spoof_data`, then:

```bash
bash /workspace/spoof/pod/pod_setup.sh
bash /workspace/spoof/pod/run.sh prep
bash /workspace/spoof/pod/run.sh train <name> <train args...>
bash /workspace/spoof/pod/run.sh eval <models>
bash /workspace/spoof/pod/run.sh surg --mode profile|heat|zone --model <m> [...]
```

`run.sh` sets `PYTHONPATH=/workspace/spoof/pipeline`. Per-pod job lists (`queue_*.sh`) are local only and gitignored.

## On the laptop

```bash
python spoof/analysis/analyze_surg.py <out_dir>
python spoof/figures/fig_surg.py
```

To run a pipeline module locally, put `spoof/pipeline` on `PYTHONPATH`.
