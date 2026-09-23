import numpy as np
import torch

from lime import lime_image
from torchvision import transforms

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def explain_lime(
    model,
    image,
    predicted_class,
    device,
    num_samples=500,
    num_features=10,
):

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


    def classifier_fn(
        images,
    ):

        tensors = []

        for image_array in images:

            image_array = np.asarray(
                image_array
            )

            tensor = normalize(
                image_array
            )

            tensors.append(
                tensor
            )


        batch = torch.stack(
            tensors
        ).to(
            device
        )


        with torch.no_grad():

            probabilities = (
                torch.softmax(
                    model(
                        batch
                    ),
                    dim=1,
                )
            )


        return (
            probabilities
            .cpu()
            .numpy()
        )


    image_array = np.asarray(
        image.convert(
            "RGB"
        )
    )


    explainer = (
        lime_image.LimeImageExplainer()
    )


    explanation = (
        explainer.explain_instance(
            image_array,
            classifier_fn,
            labels=[
                int(
                    predicted_class
                )
            ],
            top_labels=None,
            hide_color=0,
            num_samples=int(
                num_samples
            ),
        )
    )


    segments = (
        explanation.segments
    )


    contributions = (
        explanation.local_exp[
            int(
                predicted_class
            )
        ]
    )


    # Only regions supporting the predicted class
    positive = [
        (
            segment,
            weight,
        )
        for segment, weight
        in contributions
        if weight > 0
    ]


    positive.sort(
        key=lambda x: x[1],
        reverse=True,
    )


    positive = positive[
        :int(
            num_features
        )
    ]


    heatmap = np.zeros(
        segments.shape,
        dtype=np.float32,
    )


    for segment, weight in positive:

        heatmap[
            segments
            == segment
        ] = float(
            weight
        )


    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )


    return heatmap