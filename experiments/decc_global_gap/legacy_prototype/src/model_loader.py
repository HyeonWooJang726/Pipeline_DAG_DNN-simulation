from torchvision import models


def load_model(name: str):
    name = name.lower().replace("-", "_")
    if name == "vgg11":
        model = models.vgg11(weights=None)
    elif name == "alexnet":
        model = models.alexnet(weights=None)
    elif name == "resnet18":
        model = models.resnet18(weights=None)
    elif name in ("inception_v3", "inceptionv3"):
        model = models.inception_v3(weights=None, aux_logits=False, init_weights=False)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval()
    return model
