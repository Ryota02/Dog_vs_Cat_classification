import numpy as np
import shap
import torch.nn as nn


class TargetModel(
    nn.Module
):

    def __init__(
        self,
        model,
        target_class,
    ):

        super().__init__()

        self.model = model

        self.target_class = int(
            target_class
        )


    def forward(
        self,
        x,
    ):

        logits = self.model(
            x
        )

        return logits[
            :,
            self.target_class:
            self.target_class + 1
        ]


def explain_shap(
    model,
    input_tensor,
    predicted_class,
    nsamples=64,
    background_size=4,
):

    wrapped = TargetModel(
        model=model,
        target_class=(
            predicted_class
        ),
    )


    background = (
        input_tensor
        .new_zeros(
            (
                int(
                    background_size
                ),
                *input_tensor.shape[
                    1:
                ],
            )
        )
    )


    explainer = (
        shap.GradientExplainer(
            wrapped,
            background,
        )
    )


    values = (
        explainer.shap_values(
            input_tensor,
            nsamples=int(
                nsamples
            ),
        )
    )


    if isinstance(
        values,
        list,
    ):

        values = values[
            0
        ]


    values = np.asarray(
        values
    )

    values = np.squeeze(
        values
    )


    if (
        values.ndim == 3
        and
        values.shape[
            0
        ] in {
            1,
            3,
        }
    ):

        # Positive contributions to predicted class
        values = np.maximum(
            values,
            0.0,
        )

        heatmap = values.mean(
            axis=0
        )


    elif (
        values.ndim == 3
        and
        values.shape[
            -1
        ] in {
            1,
            3,
        }
    ):

        values = np.maximum(
            values,
            0.0,
        )

        heatmap = values.mean(
            axis=-1
        )


    else:

        heatmap = np.maximum(
            values,
            0.0,
        )


    heatmap = heatmap.astype(
        np.float32
    )


    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )


    return heatmap