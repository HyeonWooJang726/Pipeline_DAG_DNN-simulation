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
- Fixed representative bandwidth
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

## Run

```bash
python run_experiment.py --config configs/default.yaml
```

## Expected outputs

Outputs are written to `results/`:

- `summary.csv`: model-level latency and optimality gap
- `selected_partitions.json`: selected candidate indices
- `gantt_events.json`: timing events for plotting
