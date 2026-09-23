import torch.nn as nn

from torchvision import models


def build_model(
    backbone,
    num_classes,
    pretrained,
):

    if backbone == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model


    if backbone == "densenet201":

        weights = (
            models.DenseNet201_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.densenet201(
            weights=weights
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )

        return model


    if backbone == "vit_b_16":

        weights = (
            models.ViT_B_16_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.vit_b_16(
            weights=weights
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )

        return model


    raise ValueError(
        f"Unknown backbone: "
        f"{backbone}"
    )