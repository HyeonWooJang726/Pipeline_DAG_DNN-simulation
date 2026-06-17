# Codex Task: Implement DECC Global Optimality Gap Experiment

## Goal

Implement a minimal experimental pipeline for measuring the optimality gap of DECC-style branch-wise partitioning under pipeline-coupled DAG DNN inference.

## Hard constraints

Do **not** implement:

- full DECC reproduction
- bandwidth sweep
- device/server speed sweep
- DADS/Neurosurgeon reproduction
- accuracy evaluation
- energy evaluation
- multi-device scheduling
- dynamic bandwidth or queueing
- socket deployment for every split

The experiment must preserve DECC's:

1. DFS-based branch decomposition
2. branch execution order
3. dependency handling
4. aggregation rule
5. candidate partition set
6. pipeline recurrence

The only difference between the baseline and global method is the partition selection rule.

## Models

Use official torchvision models:

- VGG11
- AlexNet
- ResNet18
- InceptionV3

Use CIFAR10 test images resized as follows:

- VGG11 / AlexNet / ResNet18: 224 x 224
- InceptionV3: 299 x 299

Labels are not used.

## Cost model

Do **not** train an LEM.

Use profiling-based cost tables:

- `d_i^q`: profiled mobile-side prefix execution time
- `t_i^q`: activation tensor bytes / fixed bandwidth
- `s_i^q`: profiled server-side suffix execution time

If separate mobile/server hardware is unavailable, emulate:

- mobile: CPU with limited thread count or scaled latency
- server: CPU/GPU normal setting or lower latency scale

Document this as an emulated mobile-edge setting, not a real deployment.

## Baseline: DECC-style branch-wise selection

For each branch:

```text
p_i = argmin_q (d_i^q + t_i^q + s_i^q)
```

Then:

```text
raw partition vector
-> apply DECC aggregation rule
-> evaluate pipeline latency
```

## Global selection

Search over the same candidate set:

```text
for every candidate vector:
    apply DECC aggregation rule
    evaluate pipeline latency
return candidate vector with minimum final L_srv
```

Use brute force first. Add Pareto DP only when candidate count becomes too large.

## Pipeline recurrence

```text
L_dev_i = L_dev_{i-1} + d_i
L_tx_i  = max(L_tx_{i-1}, L_dev_i) + t_i
L_srv_i = max(L_srv_{i-1}, L_tx_i) + s_i
```

Final latency:

```text
L = L_srv_k
```

## Required outputs

For each model, report:

- model name
- structure type
- number of branches
- number of candidates per branch
- DECC-style latency
- global latency
- optimality gap
- runtime
- selected DECC-style partition vector
- selected global partition vector

Write outputs to:

- `results/summary.csv`
- `results/selected_partitions.json`
- `results/gantt_events.json`
