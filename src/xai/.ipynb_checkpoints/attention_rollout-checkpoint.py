import math

import numpy as np
import torch
import torch.nn.functional as F


def explain_attention_rollout(
    model,
    input_tensor,
):

    attentions = []

    handles = []


    def hook(
        module,
        inputs,
    ):

        x = inputs[
            0
        ]


        x_norm = module.ln_1(
            x
        )


        with torch.no_grad():

            _, weights = (
                module.self_attention(
                    x_norm,
                    x_norm,
                    x_norm,
                    need_weights=True,
                    average_attn_weights=False,
                )
            )


        attentions.append(
            weights.detach()
        )


    for block in (
        model.encoder.layers
    ):

        handles.append(
            block.register_forward_pre_hook(
                hook
            )
        )


    try:

        with torch.no_grad():

            model(
                input_tensor
            )

    finally:

        for handle in handles:

            handle.remove()


    joint = None


    for attention in attentions:

        attention = attention.mean(
            dim=1
        )


        n = attention.shape[
            -1
        ]


        identity = torch.eye(
            n,
            device=attention.device,
        ).unsqueeze(
            0
        )


        attention = (
            attention
            + identity
        )


        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True,
            )
        )


        if joint is None:

            joint = attention

        else:

            joint = torch.bmm(
                attention,
                joint,
            )


    cls_attention = (
        joint[
            0,
            0,
            1:
        ]
    )


    n_patches = int(
        cls_attention.numel()
    )


    grid = int(
        math.sqrt(
            n_patches
        )
    )


    heatmap = (
        cls_attention
        .reshape(
            1,
            1,
            grid,
            grid,
        )
    )


    heatmap = F.interpolate(
        heatmap,
        size=(
            input_tensor.shape[
                -2
            ],
            input_tensor.shape[
                -1
            ],
        ),
        mode="bilinear",
        align_corners=False,
    )


    heatmap = (
        heatmap[
            0,
            0
        ]
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    heatmap -= heatmap.min()


    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )


    return heatmap