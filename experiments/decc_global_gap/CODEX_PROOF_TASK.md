# CODEX_PROOF_TASK.md

## Purpose

Revise the current `experiments/decc_global_gap` experiment into a smaller proof-oriented experiment.

This is **not** a full DECC reproduction and **not** a large-scale benchmark.  
The goal is to test one specific hypothesis:

> DECC-style branch-wise local partition selection can be suboptimal when the actual objective is the DECC pipeline completion time.

The comparison must be:

1. **DECC-style branch-wise selection**
   - For each branch, choose the candidate partition point minimizing local branch cost `d + t + s`.
   - Apply the same DECC-style aggregation rule.
   - Evaluate final latency using the DECC pipeline recurrence.

2. **Global exhaustive recurrence search**
   - Enumerate all candidate partition vectors for each DAG block.
   - For each vector, apply the same aggregation rule.
   - Evaluate final latency using the same pipeline recurrence.
   - Select the vector with minimum final latency.

The experiment must use **real DAG DNN blocks** from torchvision models:
- ResNet18 residual blocks
- InceptionV3 Mixed/Inception blocks

Do **not** treat each block as one atomic layer.  
Each selected block must be internally traced and handled as a DAG subgraph.

---

## Hard Constraints

1. Do **not** implement Pareto DP in this version.
2. Do **not** switch to DP when combinations are large.
3. Use brute-force exhaustive search only.
4. If the number of combinations is too large, skip that block and report `skipped_too_many_combinations`.
5. Preserve existing files where possible.
6. Add proof/v2-specific files instead of overwriting the current experiment.
7. ResNet/Inception blocks must be treated as actual DAG blocks.
8. Collect real intermediate block inputs using forward hooks.
9. Do not feed raw CIFAR10 images directly into residual or inception blocks.

---

## Key Technical Point: Block Inputs

A ResNet BasicBlock or Inception Mixed block does **not** receive the original image tensor directly.  
It receives an intermediate activation tensor from earlier layers of the full model.

Therefore:
- Load the full model.
- Run input images through the full model.
- Register forward pre-hooks on selected blocks.
- Collect the actual input tensor received by each block.
- Use those collected block inputs for block-level FX profiling.

Without this, the block experiment is invalid or may fail due to tensor shape mismatch.

---

## Models

Use actual torchvision models:

- `vgg11`: chain control
- `resnet18`: residual block-level DAG experiment
- `inception_v3`: Mixed/Inception block-level DAG experiment
- `alexnet`: optional weak-DAG/control model

For ResNet18, extract:

```text
layer1.0
layer1.1
layer2.0
layer2.1
layer3.0
layer3.1
layer4.0
layer4.1
```

For InceptionV3, extract available Mixed blocks:

```text
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
Mixed_7a
Mixed_7b
Mixed_7c
```

---

## Required Files to Add

Add the following files:

```text
configs/proof.yaml
src/block_extractor.py
src/block_input_collector.py
src/decc_branching_v2.py
src/block_profiler.py
src/cost_table_v2.py
src/aggregation_v2.py
src/exhaustive_search.py
src/latency_eval_v2.py
run_proof_experiment.py
```

---

## 1. `configs/proof.yaml`

Create:

```yaml
seed: 0
num_images: 8
batch_size: 1
bandwidth_mbps: 100.0
mobile_latency_scale: 4.0
server_latency_scale: 1.0
max_candidates_per_branch: 3
max_bruteforce_combinations_per_block: 50000
require_real_cifar10: false
results_dir: results_proof

models:
  - name: vgg11
    structure: chain
    input_size: 224

  - name: resnet18
    structure: residual_dag_blocks
    input_size: 224

  - name: inception_v3
    structure: inception_dag_blocks
    input_size: 299
```

---

## 2. `src/block_extractor.py`

Implement model-specific block extraction.

### ResNet18

Extract these BasicBlocks:

```text
layer1.0
layer1.1
layer2.0
layer2.1
layer3.0
layer3.1
layer4.0
layer4.1
```

Each BasicBlock is one DAG block.

Inside each block, the internal DAG should include:
- main path: conv/bn/relu/conv/bn
- skip path: identity or downsample
- merge: add/relu

### InceptionV3

Extract these blocks when present in torchvision:

```text
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
Mixed_7a
Mixed_7b
Mixed_7c
```

Each `Mixed_*` module is one DAG block.

Inside each block, trace internal branch paths and concat merge structure.

### VGG11

Use as a chain control. It can use a simple chain-style block or the existing chain behavior.

Return block metadata:

```python
{
    "model": model_name,
    "block_id": "...",
    "block_type": "residual" | "inception" | "chain",
    "module": module,
}
```

---

## 3. `src/block_input_collector.py`

Collect actual intermediate inputs for each selected block.

Requirements:
- Load the full torchvision model.
- Set `model.eval()`.
- Register forward pre-hooks on selected block modules.
- Run CIFAR10 or random fallback inputs through the full model.
- Save the actual input received by each block.
- If input is a tuple, preserve tuple format.
- Detach tensors and move to CPU if needed.
- Return `input_source = "cifar10"` or `"random_fallback"`.

Return format:

```python
{
    "block_id": block_id,
    "inputs": [input_tensor_or_tuple, ...],
    "input_source": "cifar10" | "random_fallback"
}
```

Do not profile residual or inception blocks using raw image tensors.

---

## 4. `src/decc_branching_v2.py`

Implement DECC-style DFS branch decomposition for a block FX graph.

Required behavior:
- Trace the block module using `torch.fx.symbolic_trace`.
- Remove `placeholder` and `output` nodes from candidate compute nodes.
- Build predecessor and successor lists.
- Use DFS to create the first branch.
- During DFS backtracking, create a new branch.
- If the new branch reaches a node already assigned to an earlier branch, move that node and the suffix after that node from the earlier branch to the current branch.
- Keep deterministic ordering using FX node order.
- Record branch movement and dependency events.

Return:

```python
{
    "branches": [...],
    "dependencies": [...],
    "branching_events": [...],
    "graph_kind": "fx_block"
}
```

Each branch:

```python
{
    "branch_id": int,
    "nodes": [node_name, ...],
    "candidates": [
        {"candidate_id": int, "cut_index": int, "cut_node": str}
    ]
}
```

Candidate rule:
- maximum 3 candidates per branch
- use start / middle / end unique cut indices
- activation-aware candidates are not required yet

---

## 5. `src/block_profiler.py`

Profile FX node latency and activation bytes for each block.

Use actual block input tensors from `block_input_collector.py`.

It is acceptable to use an FX-node-level measured cost proxy:
- node latency measured by FX Interpreter
- activation bytes computed from tensor outputs
- do not claim direct segment profiling

Return:

```python
{
    "latency_by_node": {...},
    "activation_bytes_by_node": {...},
    "input_bytes": int,
    "cost_model": "fx_node_measured_proxy"
}
```

---

## 6. `src/cost_table_v2.py`

For each branch and candidate, compute:

```python
d = prefix_node_latency_sum * mobile_latency_scale
t = activation_bytes_at_cut / bandwidth_Bps
s = suffix_node_latency_sum * server_latency_scale
```

Also store:

```text
prefix_latency_raw
suffix_latency_raw
activation_bytes
cut_node
cut_index
full_branch_server_cost
full_branch_mobile_cost
```

---

## 7. `src/aggregation_v2.py`

Replace the current vector-only repair with DECC-style aggregation over an effective execution plan.

Input:
- branches
- dependencies
- raw partition vector
- cost_table

Output:

```python
{
    "vector": repaired_vector,
    "effective_costs": [
        {"branch_id": i, "d": ..., "t": ..., "s": ...}
    ],
    "aggregation_events": [...]
}
```

Implement two cases.

### Case 1: full-cloud aggregation

If a branch requires input from an earlier branch node that is already assigned to cloud-side execution, convert the dependent branch to cloud-only:

```python
d = 0
t = 0
s = full_branch_server_cost
event_type = "case1_full_cloud"
```

### Case 2: suffix merge aggregation

If a later branch has cloud-side suffix nodes downstream of an earlier branch’s cloud-side partition output, move that suffix computation into the earlier branch’s server stage to avoid unnecessary transmission.

Use graph reachability or explicit dependency edges. Do not rely only on candidate index comparison.

Implementation idea:

```python
Bx.s += moved_suffix_server_cost
By.s -= moved_suffix_server_cost
mark moved nodes in aggregated_nodes to avoid duplicate counting
event_type = "case2_suffix_merge"
```

If there is ambiguity, choose deterministic behavior based on branch order and log the event.

Every aggregation decision must be saved in `aggregation_events`.

---

## 8. `src/exhaustive_search.py`

Implement exact brute-force global search only.

Functions:

```python
select_decc_style_v2(branches, cost_table, aggregation_fn)
select_global_exhaustive_v2(branches, cost_table, aggregation_fn, max_combinations)
```

### DECC-style

- choose per branch candidate minimizing `d + t + s`
- apply `aggregation_v2`
- evaluate pipeline recurrence

### Global exhaustive

- enumerate all candidate vectors
- for each vector:
  - apply `aggregation_v2`
  - evaluate pipeline recurrence
- select lowest latency
- if combinations > max_combinations:
  - return skipped result with reason
  - do not use Pareto DP

---

## 9. `src/latency_eval_v2.py`

Evaluate pipeline latency from effective costs:

```python
L_dev_i = L_dev_{i-1} + d_i
L_tx_i  = max(L_tx_{i-1}, L_dev_i) + t_i
L_srv_i = max(L_srv_{i-1}, L_tx_i) + s_i
```

Return:
- final latency
- gantt events
- device_busy
- tx_busy
- server_busy
- bottleneck_stage

---

## 10. `run_proof_experiment.py`

Run the proof experiment.

Flow:
1. Load model.
2. Load CIFAR10 or random fallback input.
3. Extract real DAG blocks.
4. Collect real intermediate block inputs using hooks.
5. For each block:
   - trace block graph
   - decompose branches
   - profile node costs using actual block input tensor
   - build candidate cost table
   - run DECC-style
   - run global exhaustive
   - compute optimality gap
6. Save result files.

Save:

```text
results_proof/summary.csv
results_proof/block_results.json
results_proof/selected_partitions.json
results_proof/gantt_events.json
results_proof/aggregation_events.json
```

Required `summary.csv` columns:

```text
model
block_id
block_type
branches
candidates_total
combination_count
global_solver
decc_style_latency_ms
global_latency_ms
optimality_gap_percent
device_busy_ms
tx_busy_ms
server_busy_ms
bottleneck_stage
cost_model
input_source
graph_kind
status
skip_reason
```

---

## Acceptance Criteria

1. No Pareto DP is used anywhere.
2. For each evaluated block, global search is brute-force exhaustive.
3. DECC-style and global use the same candidate set, same cost table, same aggregation rule, and same pipeline recurrence.
4. Aggregation events are saved and inspectable.
5. ResNet18 and InceptionV3 are evaluated at block level, not full FX-node model level.
6. Each residual or inception block is internally traced and treated as a DAG subgraph.
7. The whole block must not be treated as one atomic layer.
8. Real intermediate block inputs are collected using hooks and used for profiling.
9. The experiment runs with the small proof config in reasonable time.
10. The output clearly distinguishes evaluated blocks from skipped blocks.
