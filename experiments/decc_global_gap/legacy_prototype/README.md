# Legacy Prototype Flow

This folder contains the early prototype experiment flow that is no longer used
for the reported proof result.

Archived here:

- `run_experiment.py`
- `configs/default.yaml`
- non-v2 prototype modules under `src/`
- old prototype `results/`
- `CODEX_TASK.md`

The active reported proof experiment remains one directory up:

```bash
cd experiments/decc_global_gap
python run_proof_experiment.py --config configs/proof.yaml
```

The bandwidth proof comparison also remains one directory up:

```bash
python run_proof_experiment.py --config configs/proof_100mbps.yaml
python run_proof_experiment.py --config configs/proof_1000mbps.yaml
python plot_bandwidth_latency_comparison.py
```
