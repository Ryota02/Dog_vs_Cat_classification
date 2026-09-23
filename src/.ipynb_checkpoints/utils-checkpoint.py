import random

import numpy as np
import torch
import yaml


def load_config(path,):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(
            f
        )


def set_seed(
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(
    require_cuda=True,
):

    if require_cuda:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required "
                "but is not available."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )