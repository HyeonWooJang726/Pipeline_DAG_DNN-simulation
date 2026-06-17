from itertools import islice
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def make_proof_loader(
    input_size: int,
    num_images: int,
    batch_size: int,
    seed: int,
    cifar10_root: str,
    require_real_cifar10: bool = False,
) -> Tuple[DataLoader, str]:
    """Load CIFAR10 if present, otherwise use deterministic random fallback."""
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    try:
        ds = datasets.CIFAR10(root=cifar10_root, train=False, download=False, transform=tf)
        subset = list(islice(ds, num_images))
        if len(subset) < num_images:
            raise RuntimeError(f"CIFAR10 only yielded {len(subset)} images")
        xs = torch.stack([x for x, _ in subset], dim=0)
        input_source = "cifar10"
    except Exception as exc:
        if require_real_cifar10:
            raise RuntimeError("Real CIFAR10 was required but could not be loaded") from exc
        generator = torch.Generator().manual_seed(seed)
        xs = torch.randn(num_images, 3, input_size, input_size, generator=generator)
        input_source = "random_fallback"

    ys = torch.zeros(len(xs), dtype=torch.long)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False)
    return loader, input_source


@torch.no_grad()
def collect_block_inputs(model, blocks: List[Dict], loader: Iterable, input_source: str) -> Dict[str, Dict]:
    """Collect actual forward pre-hook inputs for selected modules."""
    model.eval()
    collected = {
        block["block_id"]: {
            "block_id": block["block_id"],
            "inputs": [],
            "input_source": input_source,
        }
        for block in blocks
    }

    handles = []

    def make_hook(block_id: str):
        def hook(_module, args):
            collected[block_id]["inputs"].append(_detach_to_cpu(_normalize_hook_args(args)))
        return hook

    for block in blocks:
        handles.append(block["module"].register_forward_pre_hook(make_hook(block["block_id"])))

    try:
        for x, _ in loader:
            model(x)
    finally:
        for handle in handles:
            handle.remove()

    return collected


def _normalize_hook_args(args):
    if len(args) == 1:
        return args[0]
    return tuple(args)


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_detach_to_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    return value
