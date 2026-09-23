from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_xai_figure(
    image,
    heatmaps,
    predicted_class,
    probability,
    output_path,
):

    names = list(
        heatmaps.keys()
    )


    columns = (
        len(
            names
        )
        + 1
    )


    figure = plt.figure(
        figsize=(
            4 * columns,
            4,
        )
    )


    axis = figure.add_subplot(
        1,
        columns,
        1,
    )


    axis.imshow(
        image
    )

    axis.set_title(
        "Input"
    )

    axis.axis(
        "off"
    )


    for index, name in enumerate(
        names,
        start=2,
    ):

        axis = figure.add_subplot(
            1,
            columns,
            index,
        )


        axis.imshow(
            image
        )


        axis.imshow(
            heatmaps[
                name
            ],
            cmap="jet",
            alpha=0.45,
            vmin=0.0,
            vmax=1.0,
        )


        axis.set_title(
            name
        )

        axis.axis(
            "off"
        )


    figure.suptitle(
        (
            f"Predicted: "
            f"{predicted_class} | "
            f"p={probability:.4f}"
        )
    )


    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    figure.tight_layout()


    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )