from itertools import islice

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def make_loader(
    input_size: int,
    num_images: int,
    batch_size: int,
    seed: int = 0,
    cifar10_root: str = "./data",
    download_cifar10: bool = True,
):
    """Return a small real-image loader.

    Falls back to random tensors if CIFAR10 cannot be downloaded.
    Labels are unused.
    """
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    try:
        ds = datasets.CIFAR10(root=cifar10_root, train=False, download=download_cifar10, transform=tf)
        subset = list(islice(ds, num_images))
        xs = torch.stack([x for x, _ in subset], dim=0)
    except Exception:
        generator = torch.Generator().manual_seed(seed)
        xs = torch.randn(num_images, 3, input_size, input_size, generator=generator)

    ys = torch.zeros(len(xs), dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False)
