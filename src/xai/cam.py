from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)


def explain_cam(
    model,
    input_tensor,
    predicted_class,
    backbone,
    method,
):

    if backbone != "resnet50":

        raise ValueError(
            "CAM is currently configured "
            "for ResNet50."
        )


    target_layer = (
        model.layer4[
            -1
        ]
    )


    if method == "gradcam":

        cam_class = GradCAM

    elif (
        method
        == "gradcam_plus_plus"
    ):

        cam_class = (
            GradCAMPlusPlus
        )

    else:

        raise ValueError(
            method
        )


    targets = [
        ClassifierOutputTarget(
            int(
                predicted_class
            )
        )
    ]


    with cam_class(
        model=model,
        target_layers=[
            target_layer
        ],
    ) as cam:

        result = cam(
            input_tensor=(
                input_tensor
            ),
            targets=targets,
        )


    return result[
        0
    ]