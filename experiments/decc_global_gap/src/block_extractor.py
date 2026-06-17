from typing import Dict, List


RESNET18_BLOCKS = [
    "layer1.0",
    "layer1.1",
    "layer2.0",
    "layer2.1",
    "layer3.0",
    "layer3.1",
    "layer4.0",
    "layer4.1",
]

INCEPTION_V3_BLOCKS = [
    "Mixed_5b",
    "Mixed_5c",
    "Mixed_5d",
    "Mixed_6a",
    "Mixed_6b",
    "Mixed_6c",
    "Mixed_6d",
    "Mixed_6e",
    "Mixed_7a",
    "Mixed_7b",
    "Mixed_7c",
]


def extract_model_blocks(model, model_name: str) -> List[Dict]:
    """Return proof blocks without treating residual/Inception blocks as atoms."""
    normalized = model_name.lower().replace("-", "_")
    modules = dict(model.named_modules())

    if normalized == "resnet18":
        return _extract_named_blocks(
            model_name=normalized,
            modules=modules,
            names=RESNET18_BLOCKS,
            block_type="residual",
        )

    if normalized in ("inception_v3", "inceptionv3"):
        return _extract_named_blocks(
            model_name="inception_v3",
            modules=modules,
            names=INCEPTION_V3_BLOCKS,
            block_type="inception",
            skip_missing=True,
        )

    if normalized == "vgg11":
        module = getattr(model, "features", model)
        return [{
            "model": normalized,
            "block_id": "features",
            "block_type": "chain",
            "module": module,
        }]

    return []


def _extract_named_blocks(
    model_name: str,
    modules: Dict[str, object],
    names: List[str],
    block_type: str,
    skip_missing: bool = False,
) -> List[Dict]:
    blocks = []
    missing = []
    for block_id in names:
        module = modules.get(block_id)
        if module is None:
            missing.append(block_id)
            continue
        blocks.append({
            "model": model_name,
            "block_id": block_id,
            "block_type": block_type,
            "module": module,
        })

    if missing and not skip_missing:
        raise KeyError(f"Missing expected {model_name} blocks: {missing}")
    return blocks
