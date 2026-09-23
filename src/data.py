from pathlib import Path

import pandas as pd

from PIL import Image

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


class DogCatDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        data_root,
        transform,
    ):

        self.dataframe = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.data_root = Path(
            data_root
        )

        self.transform = transform


    def __len__(
        self,
    ):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index,
    ):

        row = (
            self.dataframe
            .iloc[
                index
            ]
        )


        image_path = (
            self.data_root
            / row[
                "relative_path"
            ]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        if self.transform is not None:

            image = self.transform(
                image
            )


        label = int(
            row[
                "label"
            ]
        )


        return (
            image,
            label,
        )


def build_transform(
    cfg,
    training,
):

    resize_size = int(
        cfg[
            "image"
        ][
            "resize_size"
        ]
    )

    image_size = int(
        cfg[
            "image"
        ][
            "image_size"
        ]
    )


    if training:

        return transforms.Compose(
            [
                transforms.Resize(
                    (
                        resize_size,
                        resize_size,
                    )
                ),

                transforms.RandomCrop(
                    image_size
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    IMAGENET_MEAN,
                    IMAGENET_STD,
                ),
            ]
        )


    return transforms.Compose(
        [
            transforms.Resize(
                (
                    resize_size,
                    resize_size,
                )
            ),

            transforms.CenterCrop(
                image_size
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def build_display_transform(
    cfg,
):

    resize_size = int(
        cfg[
            "image"
        ][
            "resize_size"
        ]
    )

    image_size = int(
        cfg[
            "image"
        ][
            "image_size"
        ]
    )


    return transforms.Compose(
        [
            transforms.Resize(
                (
                    resize_size,
                    resize_size,
                )
            ),

            transforms.CenterCrop(
                image_size
            ),
        ]
    )


def build_datasets(
    cfg,
):

    seed = int(
        cfg[
            "seed"
        ]
    )


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


    dataframe = pd.read_csv(
        manifest_path
    )


    data_root = Path(
        cfg[
            "dataset"
        ][
            "root"
        ]
    )


    train_df = dataframe[
        dataframe[
            "split"
        ]
        == "train"
    ].copy()


    val_df = dataframe[
        dataframe[
            "split"
        ]
        == "val"
    ].copy()


    test_df = dataframe[
        dataframe[
            "split"
        ]
        == "test"
    ].copy()


    train_dataset = DogCatDataset(
        dataframe=train_df,
        data_root=data_root,
        transform=build_transform(
            cfg,
            training=True,
        ),
    )


    val_dataset = DogCatDataset(
        dataframe=val_df,
        data_root=data_root,
        transform=build_transform(
            cfg,
            training=False,
        ),
    )


    test_dataset = DogCatDataset(
        dataframe=test_df,
        data_root=data_root,
        transform=build_transform(
            cfg,
            training=False,
        ),
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def build_loaders(
    cfg,
    batch_size,
):

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_datasets(
        cfg
    )


    num_workers = int(
        cfg[
            "train"
        ][
            "num_workers"
        ]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


    return (
        train_loader,
        val_loader,
        test_loader,
    )