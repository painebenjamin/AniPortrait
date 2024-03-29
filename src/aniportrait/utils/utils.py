from __future__ import annotations
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision

from einops import rearrange
from PIL import Image

from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "seed_everything",
    "get_data_dir",
    "reiterator",
    "latent_friendly_image",
]

def latent_friendly_image(
    image: Union[List[Image.Image], Image.Image],
    nearest: int=8
) -> Union[List[Image.Image], Image.Image]:
    """
    Resize an image or list of images to be friendly to latent space optimization
    """
    if isinstance(image, list):
        return [latent_friendly_image(img, nearest) for img in image]
    width, height = image.size
    new_width = (width // nearest) * nearest
    new_height = (height // nearest) * nearest
    image = image.resize((new_width, new_height), Image.NEAREST)
    return image

def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def get_data_dir() -> str:
    return osp.realpath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "data"))

class reiterator:
    """
    Transparently memoize any iterator
    """
    memoized: List[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable
        self.memoized = []
        self.started = False
        self.finished = False

    def __iter__(self) -> Iterable[Any]:
        if not self.started:
            self.started = True
            last_index: Optional[int] = None
            for i, value in enumerate(self.iterable):
                yield value
                self.memoized.append(value)
                last_index = i
                if self.finished:
                    # Completed somewhere else
                    break
            if self.finished:
                if last_index is None:
                    last_index = 0
                for value in self.memoized[last_index+1:]:
                    yield value
            self.finished = True
            del self.iterable
        elif not self.finished:
            # Complete iterator
            self.memoized += [item for item in self.iterable]
            self.finished = True
            del self.iterable
            for item in self.memoized:
                yield item
        else:
            for item in self.memoized:
                yield item
