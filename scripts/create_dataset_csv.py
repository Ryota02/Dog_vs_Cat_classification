import argparse
import sys

from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(ROOT_DIR)
)


from src.utils import (
    load_config,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_images(
    directory,
):

    directory = Path(
        directory
    )

    return sorted(
        path
        for path
        in directory.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
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


    dataset_cfg = (
        cfg[
            "dataset"
        ]
    )


    data_root = Path(
        dataset_cfg[
            "root"
        ]
    )


    classes = list(
        dataset_cfg[
            "classes"
        ]
    )


    split_cfg = (
        dataset_cfg[
            "split"
        ]
    )


    train_ratio = float(
        split_cfg[
            "train"
        ]
    )

    val_ratio = float(
        split_cfg[
            "val"
        ]
    )

    test_ratio = float(
        split_cfg[
            "test"
        ]
    )


    total_ratio = (
        train_ratio
        + val_ratio
        + test_ratio
    )


    if abs(
        total_ratio - 1.0
    ) > 1e-8:

        raise ValueError(
            "train + val + test "
            "must equal 1.0."
        )


    # ========================================================
    # Collect images
    # ========================================================

    rows = []


    for label, class_name in enumerate(
        classes
    ):

        class_dir = (
            data_root
            / class_name
        )


        if not class_dir.exists():

            raise FileNotFoundError(
                f"Class directory not found: "
                f"{class_dir}"
            )


        image_paths = find_images(
            class_dir
        )


        print(
            f"[INFO] {class_name}: "
            f"{len(image_paths)} images"
        )


        if len(
            image_paths
        ) < 10:

            raise RuntimeError(
                f"Too few images in "
                f"{class_name}: "
                f"{len(image_paths)}"
            )


        for image_path in image_paths:

            rows.append(
                {
                    "relative_path":
                        str(
                            image_path.relative_to(
                                data_root
                            )
                        ),

                    "label":
                        label,

                    "class_name":
                        class_name,
                }
            )


    dataframe = pd.DataFrame(
        rows
    )


    # ========================================================
    # 80% train / 20% temporary
    # ========================================================

    train_df, temp_df = (
        train_test_split(
            dataframe,
            test_size=(
                val_ratio
                + test_ratio
            ),
            random_state=seed,
            stratify=(
                dataframe[
                    "label"
                ]
            ),
        )
    )


    # ========================================================
    # Temporary -> val / test
    #
    # 10 : 10
    # ========================================================

    test_fraction = (
        test_ratio
        / (
            val_ratio
            + test_ratio
        )
    )


    val_df, test_df = (
        train_test_split(
            temp_df,
            test_size=(
                test_fraction
            ),
            random_state=seed,
            stratify=(
                temp_df[
                    "label"
                ]
            ),
        )
    )


    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()


    train_df[
        "split"
    ] = "train"

    val_df[
        "split"
    ] = "val"

    test_df[
        "split"
    ] = "test"


    output_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )


    output_df = output_df[
        [
            "split",
            "relative_path",
            "label",
            "class_name",
        ]
    ]


    output_df = output_df.sort_values(
        by=[
            "split",
            "label",
            "relative_path",
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # Save
    # ========================================================

    manifest_dir = Path(
        dataset_cfg[
            "manifest_dir"
        ]
    )


    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        manifest_dir
        / f"seed{seed}.csv"
    )


    output_df.to_csv(
        output_path,
        index=False,
    )


    print(
        "\n[INFO] Saved:",
        output_path,
    )


    print(
        "\n[INFO] Split counts:"
    )


    print(
        output_df.groupby(
            [
                "split",
                "class_name",
            ]
        ).size()
    )


if __name__ == "__main__":

    main()