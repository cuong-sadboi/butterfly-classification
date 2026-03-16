import timm
import torch.nn as nn


def build_model(model_name="resnet18", num_classes=5, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError("Unsupported model head for class replacement")

    return model
