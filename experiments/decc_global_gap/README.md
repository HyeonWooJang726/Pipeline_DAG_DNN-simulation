# DECC-style Branch-wise Partitioning vs Pipeline-coupled Global Optimum

This experiment measures the **optimality gap** of DECC-style branch-wise partition selection under a pipeline-coupled DAG DNN execution model.

The goal is not to reproduce all DECC experiments. The goal is narrower:

> Use the same DECC-style branch decomposition, dependency handling, aggregation rule, candidate set, and pipeline recurrence, then compare only the partition selection rule.

## Compared methods

### 1. DECC-style branch-wise selection

For each branch `B_i`, independently select the candidate partition point minimizing local branch cost:

```text
p_i = argmin_q (d_i^q + t_i^q + s_i^q)
```

Then apply the DECC aggregation rule and evaluate the final pipeline latency.

### 2. Global pipeline optimum

Search over the same candidate set. For each candidate vector, apply the same DECC aggregation rule, then evaluate the final pipeline latency using:

```text
L_dev_i = L_dev_{i-1} + d_i
L_tx_i  = max(L_tx_{i-1}, L_dev_i) + t_i
L_srv_i = max(L_srv_{i-1}, L_tx_i) + s_i
```

The final latency is `L_srv_k`.

## Main metric

```text
Optimality gap = (L_DECC_style - L_Global) / L_DECC_style * 100
```

## Minimal scope

Included:

- Official torchvision models: VGG11, AlexNet, ResNet18, InceptionV3
- Real input images from CIFAR10 resized to model input size
- DECC-style DFS branch decomposition
- DECC-style dependency handling and aggregation hook
- Profiling-based cost table instead of LEM
- Fixed representative bandwidth per proof run
- Fixed representative mobile/server speed setting
- DECC-style vs global optimum comparison

Excluded:

- Full DECC reproduction
- Bandwidth sweep
- Device/server speed sweep
- DADS/Neurosurgeon reproduction
- Accuracy evaluation
- Energy evaluation
- Multi-device scheduling
- Dynamic bandwidth or queueing
- Full socket deployment for every split

## Installation

```bash
cd experiments/decc_global_gap
pip install -r requirements.txt
```

## Main Reported Proof Experiment

The reported proof experiment uses `run_proof_experiment.py` and the v2 proof
modules in `src/*_v2.py`.  It compares:

- Existing / DECC-style branch-wise selection: each branch independently chooses
  the candidate minimizing `d + t + s`.
- Global exhaustive selection: all partition candidate vectors are searched and
  the vector minimizing final pipeline latency is selected.

Run the bandwidth comparison proof from this directory:

```bash
cd experiments/decc_global_gap

python run_proof_experiment.py --config configs/proof_100mbps.yaml
python run_proof_experiment.py --config configs/proof_1000mbps.yaml
python plot_bandwidth_latency_comparison.py
```

The two bandwidth comparison configs evaluate InceptionV3 only, under fixed
100 Mbps and 1000 Mbps conditions.

## Expected outputs

The proof runs write per-bandwidth outputs to:

- `results_proof_100mbps/`
- `results_proof_1000mbps/`

The bandwidth comparison script writes:

- `results_bandwidth_comparison/figures/bandwidth_latency_comparison_ko.png`
- `results_bandwidth_comparison/figures/bandwidth_latency_summary_ko.csv`

Each proof results directory contains:

- `summary.csv`: block-level final pipeline latency, optimality gap, and `input_source`
- `selected_partitions.json`: selected candidate indices
- `gantt_events.json`: timing events for plotting
- `input_samples.png`: grid of the actual input tensors used by the proof loader

## Legacy Prototype Flow

`run_experiment.py` and the non-v2 modules are early prototype / legacy code.
They are kept for reference, but they are not used for the reported proof
experiment or the bandwidth comparison result above.

## Cost model note

This is an emulated mobile-edge experiment, not a socket deployment.  The
runner profiles FX-node execution on the available local machine, sums profiled
node costs into branch prefix/suffix costs, and applies fixed latency scales:

- `d_i^q`: profiled branch prefix latency multiplied by `mobile_latency_scale`
- `t_i^q`: profiled cut activation bytes divided by `bandwidth_mbps`
- `s_i^q`: profiled branch suffix latency multiplied by `server_latency_scale`

Set `download_cifar10: true` to allow the proof loader to download CIFAR10.
If CIFAR10 cannot be loaded and `require_real_cifar10: false`, the loader falls
back to deterministic random inputs and records `input_source=random_fallback`.
If `require_real_cifar10: true`, the same CIFAR10 failure raises an error.
