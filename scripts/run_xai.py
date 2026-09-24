import argparse
import csv
import random
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(ROOT_DIR)
)


from src.data import (
    build_display_transform,
)

from src.utils import (
    get_device,
    load_config,
    set_seed,
)

from src.xai.common import (
    image_to_tensor,
    load_classifier,
    predict,
)

from src.xai.cam import (
    explain_cam,
)

from src.xai.lime_explainer import (
    explain_lime,
)

from src.xai.shap_explainer import (
    explain_shap,
)

from src.xai.attention_rollout import (
    explain_attention_rollout,
)

from src.xai.plotting import (
    save_xai_figure,
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


    device = get_device(
        cfg[
            "device"
        ][
            "require_cuda"
        ]
    )


    classes = list(
        cfg[
            "dataset"
        ][
            "classes"
        ]
    )


    num_classes = len(
        classes
    )


    # ========================================================
    # Test manifest
    # ========================================================

    manifest_path = (
        Path(
            cfg[
                "dataset"
            ][
                "manifest_dir"
            ]
        )
        / f"seed{seed}.csv"
    )


    manifest = pd.read_csv(
        manifest_path
    )


    test_df = manifest[
        manifest[
            "split"
        ]
        == "test"
    ].copy()


    # ========================================================
    # Same images for every backbone
    # ========================================================

    sample_seed = int(
        cfg[
            "xai"
        ].get(
            "sample_seed",
            seed,
        )
    )


    rng = random.Random(
        sample_seed
    )


    indices = list(
        test_df.index
    )


    rng.shuffle(
        indices
    )


    num_images = int(
        cfg[
            "xai"
        ][
            "num_images"
        ]
    )


    indices = indices[
        :min(
            num_images,
            len(
                indices
            ),
        )
    ]


    selected = (
        test_df.loc[
            indices
        ]
        .copy()
    )


    print(
        "[INFO] XAI images:",
        len(
            selected
        ),
    )


    for _, row in (
        selected.iterrows()
    ):

        print(
            "[XAI IMAGE]",
            row[
                "relative_path"
            ],
        )


    data_root = Path(
        cfg[
            "dataset"
        ][
            "root"
        ]
    )


    display_transform = (
        build_display_transform(
            cfg
        )
    )


    xai_cfg = (
        cfg[
            "xai"
        ]
    )


    methods_cfg = (
        xai_cfg[
            "methods"
        ]
    )


    output_root = Path(
        cfg[
            "output"
        ][
            "xai_root"
        ]
    )


    classification_root = Path(
        cfg[
            "output"
        ][
            "classification_root"
        ]
    )


    # ========================================================
    # Backbones
    # ========================================================

    for backbone in (
        xai_cfg[
            "backbones"
        ]
    ):

        checkpoint_path = (
            classification_root
            / backbone
            / f"seed{seed}"
            / "best_model.pth"
        )


        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )


        print(
            "\n"
            "========================================"
        )

        print(
            "[INFO] Backbone:",
            backbone,
        )

        print(
            "[INFO] Checkpoint:",
            checkpoint_path,
        )

        print(
            "========================================"
        )


        model = load_classifier(
            backbone=backbone,
            num_classes=num_classes,
            checkpoint_path=(
                checkpoint_path
            ),
            device=device,
        )


        run_dir = (
            output_root
            / backbone
            / f"seed{seed}"
        )


        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        metadata_rows = []


        for index, (
            _,
            row,
        ) in enumerate(
            selected.iterrows(),
            start=1,
        ):

            relative_path = str(
                row[
                    "relative_path"
                ]
            )


            source_class = str(
                row[
                    "class_name"
                ]
            )


            image_path = (
                data_root
                / relative_path
            )


            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            image = display_transform(
                image
            )


            input_tensor = (
                image_to_tensor(
                    image,
                    device,
                )
            )


            (predicted_class,probabilities) = predict(
                model,
                input_tensor,
            )

            probabilities = np.asarray(
                probabilities
            )
            
            if probabilities.ndim == 2:
                probabilities = probabilities[0]
            
            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )
            
            predicted_probability = float(
                probabilities[
                    predicted_index
                ]
            )


            predicted_name = classes[predicted_class]

            # =================================================
            # IMPORTANT
            #
            # XAI target is ALWAYS the predicted class.
            # Ground Truth is never used as XAI target.
            # =================================================

            print(
                f"[{index}/"
                f"{len(selected)}] "
                f"{relative_path}"
            )

            print(
                "  Prediction:",
                predicted_name,
                f"{predicted_probability:.4f}",
            )


            heatmaps = {}


            # =================================================
            # Grad-CAM
            # =================================================

            if (
                methods_cfg[
                    "gradcam"
                ][
                    "enabled"
                ]
                and
                backbone
                in {
                    "resnet50",
                    "densenet201",
                }
            ):

                heatmaps[
                    "Grad-CAM"
                ] = explain_cam(
                    model=model,
                    input_tensor=input_tensor,
                    predicted_class=predicted_class,
                    backbone=backbone,
                    method="gradcam",
                )


            # =================================================
            # Grad-CAM++
            # =================================================

            if (
                methods_cfg[
                    "gradcam_plus_plus"
                ][
                    "enabled"
                ]
                and
                backbone
                in {
                    "resnet50",
                    "densenet201",
                }
            ):

                heatmaps[
                    "Grad-CAM++"
                ] = explain_cam(
                    model=model,
                    input_tensor=input_tensor,
                    predicted_class=predicted_class,
                    backbone=backbone,
                    method="gradcam_plus_plus",
                )


            # =================================================
            # LIME
            # =================================================

            if (
                methods_cfg[
                    "lime"
                ][
                    "enabled"
                ]
            ):

                heatmaps[
                    "LIME"
                ] = explain_lime(
                    model=model,
                    image=image,
                    predicted_class=predicted_class,
                    device=device,
                    num_samples=int(
                        xai_cfg[
                            "lime"
                        ][
                            "num_samples"
                        ]
                    ),
                    num_features=int(
                        xai_cfg["lime"]["num_features"]
                    ),
                )


            # =================================================
            # SHAP
            # =================================================

            if methods_cfg["shap"]["enabled"]:

                heatmaps[
                    "SHAP"
                ] = explain_shap(
                    model=model,
                    input_tensor=input_tensor,
                    predicted_class=predicted_class,
                    nsamples=int(xai_cfg["shap"]["nsamples"]),
                    background_size=int(xai_cfg["shap"]["background_size"]
                    ),
                )

            # =================================================
            # Attention Rollout
            #
            # ViT only
            # class-agnostic
            # =================================================

            if (methods_cfg["attention_rollout"]["enabled"]
                and
                backbone
                == "vit_b_16"
            ):

                heatmaps[
                    "Attention Rollout "
                    "(class-agnostic)"
                ] = (
                    explain_attention_rollout(
                        model=model,
                        input_tensor=(
                            input_tensor
                        ),
                    )
                )

            # =================================================
            # Unique case ID
            # =================================================

            relative_object = Path(relative_path)

            case_id = (
                "__".join(
                    relative_object
                    .with_suffix("")
                    .parts
                )
            )

            # =================================================
            # Save raw heatmaps
            # =================================================

            name_map = {

                "Grad-CAM":
                    "gradcam",

                "Grad-CAM++":
                    "gradcam_plus_plus",

                "LIME":
                    "lime",

                "SHAP":
                    "shap",

                "Attention Rollout (class-agnostic)":
                    "attention_rollout",
            }


            for (
                display_name,
                heatmap,
            ) in heatmaps.items():

                method_name = (
                    name_map[
                        display_name
                    ]
                )


                heatmap_dir = (
                    run_dir
                    / "heatmaps"
                    / method_name
                )


                heatmap_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                np.save(
                    heatmap_dir
                    / f"{case_id}.npy",
                    np.asarray(
                        heatmap,
                        dtype=np.float32,
                    ),
                )


            # =================================================
            # Save combined figure
            # =================================================

            figure_dir = (
                run_dir
                / "figures"
            )


            figure_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            figure_path = (
                figure_dir
                / f"{case_id}_xai.png"
            )

            save_xai_figure(
                image=image,
                heatmaps=heatmaps,
                predicted_class=predicted_class,
                probability=predicted_probability,
                output_path=figure_path,
            )


            # =================================================
            # Metadata
            #
            # source_class is stored only for checking.
            # It is NOT used as XAI target.
            # =================================================

            metadata_rows.append(
                {
                    "case_id":
                        case_id,

                    "relative_path":
                        relative_path,

                    "source_class":
                        source_class,

                    "backbone":
                        backbone,

                    "predicted_class_index":
                        predicted_class,

                    "predicted_class":
                        predicted_name,

                    "predicted_probability":
                        predicted_probability,

                    "xai_target_class":
                        predicted_name,

                    "xai_target_source":
                        "predicted_class",

                    "figure_path":
                        str(
                            figure_path
                        ),
                }
            )


        # ====================================================
        # Metadata CSV
        # ====================================================

        metadata_path = (
            run_dir
            / "xai_metadata.csv"
        )


        if metadata_rows:

            with metadata_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        metadata_rows[
                            0
                        ].keys()
                    ),
                )


                writer.writeheader()

                writer.writerows(
                    metadata_rows
                )


        print(
            "[INFO] Metadata:",
            metadata_path,
        )


        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


if __name__ == "__main__":

    main()