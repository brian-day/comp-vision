"""
See for jax typing options: https://docs.kidger.site/jaxtyping/api/array/

NOTE: Neither pytorch not jaxtyping can enforce dtype hints or shape as a static lint, only during
runtime. Will at least guarantee that operations take in / return tensors, though this is only true
for the inputs / outputs of our own defined functions, not return type of torch operations.
"""

import jaxtyping as jt
import torch
from beartype import beartype
from typing import Callable, NotRequired, Tuple, TypedDict
from PIL import Image
from torchvision import tv_tensors


# NOTE: Tried (unsuccessfully) rewriting these with typing's NewType to yield stricter annotations.
# Pylance will treat all of these types as simply torch.Tensor.
type float32Tensor = jt.Float32[torch.Tensor, "..."]
type float64Tensor = jt.Float64[torch.Tensor, "..."]
type int32Tensor = jt.Int32[torch.Tensor, "..."]
type int64Tensor = jt.Int64[torch.Tensor, "..."]
type boolTensor = jt.Bool[torch.Tensor, "..."]


class TargetDict(TypedDict):
    image_id: int
    boxes: tv_tensors.BoundingBoxes
    labels: int32Tensor
    # original_size will be added by transforms if included.
    original_size: NotRequired[int32Tensor]


class TargetDictPureTensor(TypedDict):
    image_id: int
    boxes: float64Tensor
    labels: int32Tensor
    # original_size will be added by transforms if included.
    original_size: NotRequired[int32Tensor]


type ImageTransform = Callable[[Image.Image], torch.Tensor]
type BoxedImageTransform = Callable[
    [tv_tensors.Image, TargetDict], Tuple[torch.Tensor, TargetDictPureTensor]
]


# Example: pylance does not warn of incorrect return type since float32Tensor and int64Tensor are
# both treated as just torch.Tensor
@jt.jaxtyped(typechecker=beartype)
def typing_test() -> int64Tensor:
    t1: float32Tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    t2: float64Tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
    t3: int64Tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

    # NOTE: torch.matmul will fail becasue t1 and t3 are different types...
    res: float32Tensor = torch.matmul(t1, t3)

    print(res, res.dtype)

    return res


if __name__ == "__main__":
    typing_test()
