from torchvision import models


def load_model(name: str):
    name = name.lower()
    if name == "vgg11":
        model = models.vgg11(weights=None)
    elif name == "alexnet":
        model = models.alexnet(weights=None)
    elif name == "resnet18":
        model = models.resnet18(weights=None)
    elif name == "inception_v3":
        model = models.inception_v3(weights=None, aux_logits=False)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval()
    return model
