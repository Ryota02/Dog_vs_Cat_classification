import argparse
import sys

from pathlib import Path

import torch
import torch.nn as nn


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.append(
    str(ROOT_DIR)
)


from src.data import (
    build_loaders,
)

from src.model import (
    build_model,
)

from src.utils import (
    get_device,
    load_config,
    set_seed,
)


def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels,
            )


            total_loss += (
                loss.item()
                * images.size(
                    0
                )
            )


            predictions = logits.argmax(
                dim=1
            )


            correct += (
                predictions
                == labels
            ).sum().item()


            total += images.size(
                0
            )


    return (
        total_loss
        / total,
        correct
        / total,
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


    output_root = Path(
        cfg[
            "output"
        ][
            "classification_root"
        ]
    )


    criterion = (
        nn.CrossEntropyLoss()
    )


    for backbone, model_cfg in (
        cfg[
            "models"
        ].items()
    ):

        if not model_cfg.get(
            "enabled",
            False,
        ):

            continue


        set_seed(
            seed
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
            "========================================"
        )


        batch_size = int(
            model_cfg[
                "batch_size"
            ]
        )


        (
            train_loader,
            val_loader,
            test_loader,
        ) = build_loaders(
            cfg,
            batch_size=batch_size,
        )


        print(
            "[INFO] Train:",
            len(
                train_loader.dataset
            ),
        )

        print(
            "[INFO] Val:",
            len(
                val_loader.dataset
            ),
        )

        print(
            "[INFO] Test:",
            len(
                test_loader.dataset
            ),
        )


        model = build_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=bool(
                model_cfg[
                    "pretrained"
                ]
            ),
        ).to(
            device
        )


        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(
                model_cfg[
                    "lr"
                ]
            ),
            weight_decay=float(
                cfg[
                    "train"
                ][
                    "weight_decay"
                ]
            ),
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


        checkpoint_path = (
            run_dir
            / "best_model.pth"
        )


        best_val_accuracy = -1.0


        for epoch in range(
            1,
            int(
                cfg[
                    "train"
                ][
                    "epochs"
                ]
            )
            + 1,
        ):

            model.train()

            running_loss = 0.0
            total = 0
            correct = 0


            for images, labels in (
                train_loader
            ):

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )


                optimizer.zero_grad()


                logits = model(
                    images
                )


                loss = criterion(
                    logits,
                    labels,
                )


                loss.backward()

                optimizer.step()


                running_loss += (
                    loss.item()
                    * images.size(
                        0
                    )
                )


                predictions = (
                    logits.argmax(
                        dim=1
                    )
                )


                correct += (
                    predictions
                    == labels
                ).sum().item()


                total += images.size(
                    0
                )


            train_loss = (
                running_loss
                / total
            )


            train_accuracy = (
                correct
                / total
            )


            (
                val_loss,
                val_accuracy,
            ) = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )


            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Train Acc: "
                f"{train_accuracy:.4f} | "
                f"Val Loss: "
                f"{val_loss:.4f} | "
                f"Val Acc: "
                f"{val_accuracy:.4f}"
            )


            if (
                val_accuracy
                > best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )


                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "backbone":
                            backbone,

                        "classes":
                            classes,

                        "seed":
                            seed,

                        "best_val_accuracy":
                            best_val_accuracy,
                    },
                    checkpoint_path,
                )


        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        (
            test_loss,
            test_accuracy,
        ) = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )


        print(
            f"\n[RESULT] "
            f"{backbone}"
        )

        print(
            f"Best Val Acc: "
            f"{best_val_accuracy:.4f}"
        )

        print(
            f"Test Acc: "
            f"{test_accuracy:.4f}"
        )


        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


if __name__ == "__main__":

    main()