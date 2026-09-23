import argparse
import csv
import random
import sys

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from PIL import ImageFilter

from torchvision import transforms


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(ROOT_DIR)
)


from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_display_transform,
)

from src.model import (
    build_model,
)

from src.utils import (
    get_device,
    load_config,
    set_seed,
)


def normalize_heatmap(
    heatmap,
):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )


    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


    heatmap -= (
        heatmap.min()
    )


    if heatmap.max() > 0:

        heatmap /= (
            heatmap.max()
        )


    return heatmap


def get_top_mask(
    heatmap,
    ratio,
):

    flat = (
        heatmap.reshape(
            -1
        )
    )


    num_pixels = max(
        1,
        int(
            flat.size
            * ratio
        ),
    )


    indices = np.argpartition(
        flat,
        -num_pixels,
    )[
        -num_pixels:
    ]


    mask = np.zeros(
        flat.size,
        dtype=bool,
    )


    mask[
        indices
    ] = True


    return mask.reshape(
        heatmap.shape
    )


def get_random_mask(
    shape,
    ratio,
    rng,
):

    total = (
        shape[
            0
        ]
        * shape[
            1
        ]
    )


    num_pixels = max(
        1,
        int(
            total
            * ratio
        ),
    )


    indices = rng.choice(
        total,
        size=num_pixels,
        replace=False,
    )


    mask = np.zeros(
        total,
        dtype=bool,
    )


    mask[
        indices
    ] = True


    return mask.reshape(
        shape
    )


def apply_mask(
    image,
    mask,
):

    original = np.asarray(
        image.convert(
            "RGB"
        )
    ).copy()


    blurred = np.asarray(
        image
        .convert(
            "RGB"
        )
        .filter(
            ImageFilter.GaussianBlur(
                radius=12
            )
        )
    )


    original[
        mask
    ] = blurred[
        mask
    ]


    return Image.fromarray(
        original
    )


def predict_probability(
    model,
    image,
    target_class,
    transform,
    device,
):

    tensor = (
        transform(
            image
        )
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )


    with torch.no_grad():

        probability = (
            torch.softmax(
                model(
                    tensor
                ),
                dim=1,
            )[
                0,
                target_class
            ]
            .item()
        )


    return float(
        probability
    )


def resize_heatmap(
    heatmap,
    size,
):

    heatmap = normalize_heatmap(
        heatmap
    )


    image = Image.fromarray(
        heatmap.astype(
            np.float32
        ),
        mode="F",
    )


    image = image.resize(
        size,
        Image.Resampling.BILINEAR,
    )


    return normalize_heatmap(
        np.asarray(
            image,
            dtype=np.float32,
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()


    cfg = load_config(
        args.config
    )


    seed = int(
        cfg[
            "seed"
        ]
    )


    set_seed(
        seed
    )


    rng = np.random.default_rng(
        seed
    )


    device = get_device(
        cfg[
            "device"
        ][
            "require_cuda"
        ]
    )


    checkpoint = torch.load(
        cfg[
            "output"
        ][
            "checkpoint"
        ],
        map_location=device,
        weights_only=False,
    )


    backbone = checkpoint[
        "backbone"
    ]


    classes = checkpoint[
        "classes"
    ]


    model = build_model(
        backbone=backbone,
        num_classes=len(
            classes
        ),
        pretrained=False,
    ).to(
        device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    display_transform = (
        build_display_transform(
            cfg
        )
    )


    tensor_transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),

                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD,
                ),
            ]
        )
    )


    test_root = (
        Path(
            cfg[
                "dataset"
            ][
                "root"
            ]
        )
        / cfg[
            "dataset"
        ][
            "test"
        ]
    )


    paths = sorted(
        [
            p
            for p
            in test_root.rglob(
                "*"
            )
            if p.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
            }
        ]
    )


    random.Random(
        seed
    ).shuffle(
        paths
    )


    paths = paths[
        :int(
            cfg[
                "xai"
            ][
                "num_images"
            ]
        )
    ]


    xai_root = Path(
        cfg[
            "output"
        ][
            "xai"
        ]
    )


    output_root = Path(
        cfg[
            "output"
        ][
            "faithfulness"
        ]
    )


    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    ratios = cfg[
        "faithfulness"
    ][
        "ratios"
    ]


    repeats = int(
        cfg[
            "faithfulness"
        ][
            "random_repeats"
        ]
    )


    rows = []


    for image_path in paths:

        image = display_transform(
            Image.open(
                image_path
            ).convert(
                "RGB"
            )
        )


        tensor = (
            tensor_transform(
                image
            )
            .unsqueeze(
                0
            )
            .to(
                device
            )
        )


        with torch.no_grad():

            probabilities = (
                torch.softmax(
                    model(
                        tensor
                    ),
                    dim=1,
                )
            )


        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )


        baseline_probability = float(
            probabilities[
                0,
                predicted_class
            ].item()
        )


        case_dir = (
            xai_root
            / backbone
            / image_path.parent.name
        )


        heatmap_files = sorted(
            case_dir.glob(
                f"{image_path.stem}__*.npy"
            )
        )


        for heatmap_path in heatmap_files:

            method = (
                heatmap_path.stem
                .split(
                    "__",
                    1,
                )[
                    1
                ]
            )


            heatmap = np.load(
                heatmap_path
            )


            heatmap = resize_heatmap(
                heatmap,
                image.size,
            )


            for ratio in ratios:

                ratio = float(
                    ratio
                )


                xai_mask = get_top_mask(
                    heatmap,
                    ratio,
                )


                xai_image = apply_mask(
                    image,
                    xai_mask,
                )


                xai_probability = (
                    predict_probability(
                        model,
                        xai_image,
                        predicted_class,
                        tensor_transform,
                        device,
                    )
                )


                xai_drop = (
                    baseline_probability
                    - xai_probability
                )


                random_probabilities = []


                for _ in range(
                    repeats
                ):

                    random_mask = (
                        get_random_mask(
                            heatmap.shape,
                            ratio,
                            rng,
                        )
                    )


                    random_image = apply_mask(
                        image,
                        random_mask,
                    )


                    random_probability = (
                        predict_probability(
                            model,
                            random_image,
                            predicted_class,
                            tensor_transform,
                            device,
                        )
                    )


                    random_probabilities.append(
                        random_probability
                    )


                random_probability_mean = float(
                    np.mean(
                        random_probabilities
                    )
                )


                random_drop = (
                    baseline_probability
                    - random_probability_mean
                )


                rows.append(
                    {
                        "image":
                            str(
                                image_path
                            ),

                        "predicted_class":
                            classes[
                                predicted_class
                            ],

                        "method":
                            method,

                        "ratio":
                            ratio,

                        "baseline_probability":
                            baseline_probability,

                        "xai_masked_probability":
                            xai_probability,

                        "xai_probability_drop":
                            xai_drop,

                        "random_masked_probability":
                            random_probability_mean,

                        "random_probability_drop":
                            random_drop,

                        "xai_minus_random_drop":
                            (
                                xai_drop
                                - random_drop
                            ),
                    }
                )


    csv_path = (
        output_root
        / "faithfulness_results.csv"
    )


    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print(
        "[INFO] Saved:",
        csv_path,
    )


if __name__ == "__main__":

    main()