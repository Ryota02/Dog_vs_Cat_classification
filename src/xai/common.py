import numpy as np
import torch

from PIL import Image

from torchvision import transforms


from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from src.model import (
    build_model,
)


def load_classifier(
    backbone,
    num_classes,
    checkpoint_path,
    device,
):

    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )


    model = model.to(
        device
    )


    model.eval()


    return model


def image_to_tensor(
    image,
    device,
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


    tensor = transform(
        image
    ).unsqueeze(
        0
    )


    return tensor.to(
        device
    )


def numpy_batch_to_tensor(
    images,
    device,
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


    tensors = []


    for image in images:

        image = np.asarray(
            image
        )


        if image.dtype != np.uint8:

            if image.max() <= 1.0:

                image = (
                    image
                    * 255.0
                )


            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )


        pil_image = Image.fromarray(
            image
        ).convert(
            "RGB"
        )


        tensors.append(
            transform(
                pil_image
            )
        )


    return torch.stack(
        tensors,
        dim=0,
    ).to(
        device
    )


@torch.no_grad()
def predict(
    model,
    input_tensor,
):

    logits = model(
        input_tensor
    )


    probabilities = torch.softmax(
        logits,
        dim=1,
    )


    predicted_class = int(
        probabilities.argmax(
            dim=1
        ).item()
    )


    return (
        predicted_class,
        probabilities[
            0
        ].detach().cpu().numpy(),
    )