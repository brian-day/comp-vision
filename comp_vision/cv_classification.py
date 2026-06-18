from pathlib import Path
from typing import Tuple

import torch
import torchvision
from torchvision.transforms import v2 as T

from .cv_training import get_resnet50_model
from .cv_typing import TargetDictPureTensor


def apply_nms(
    boxes, scores, labels, *, iou_thresh=0.5, score_thresh=0.05, max_detections=100, batched=True
):
    """Applies Non-Maximal Supression (NMS) to the detected bounding boxes for a given image.
    Two options for multi-class detection are (1) Standard NMS, which eliminates overlapping
    bounding boxes regardless of label, and (2) Batched NMS which only eliminates overlapping
    bounding boxes when they are of the same class. The maximal allowed overlapping area and score
    threshold can be set as parameters. By default, a batched filter is applied.

    Note that, in the present implementation, we do not allow for rotated bounding boxes, so params
    should be set with this limitation in mind. For well plate detection, it is safe to assume
    very little overlap in the detected boudning boxes.

    Args:
        boxes (_type_): _description_
        scores (_type_): _description_
        labels (_type_): _description_
        iou_thresh (float, optional): Maximum allowed intersection-over-union value. Anything boxes
            exceeding this value will be eliminated, retaining only the one with the highest score.
            This value is bounded between 0 and 1. Defaults to 0.5.
        score_thresh (float, optional): Minimum expected certainty of the predicted score for a
            given bounding box / label. Any predicted boxes with a score below this value are
            removed. Defaults to 0.05.
        max_detections (int, optional): Maximum number of predicted bounding boxes. If, after NMS
            filtering, the number of predicted boxes exceed max_detections, boxes with the lowest
            are removed. Defaults to 100.
        batched (bool, optional): Indicated if a batched (class aware) or non-batched (class
            agnostic) nms filter is used.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: _description_
    """
    # boxes: (N,4) XYXY float tensor, scores: (N,), labels: (N,) int
    keep_mask = scores > score_thresh
    boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = (
        torchvision.ops.batched_nms(boxes, scores, labels, iou_thresh)
        if batched
        else torchvision.ops.nms(boxes, scores, iou_thresh)
    )
    keep = keep[:max_detections]

    return boxes[keep], scores[keep], labels[keep]


# TODO: Remove Dataset, index, and Model, and take in image and model as params
def single_image_classify(
    dataset, state_dict_file: str | Path, idx: int = 0
) -> Tuple[torch.Tensor, TargetDictPureTensor]:
    # NOTE: Index parameter is left for convenience if wanting to look at specific images in a large
    # dataset, but the intended use of this function is single image datasets, hence the default.

    model = get_resnet50_model(dataset)
    state_dict = torch.load(state_dict_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    img, _ = dataset[idx]
    img_tensor = T.ToPureTensor()(img)
    img_batch = img_tensor.unsqueeze(0)

    device = "cpu"
    model.to(device)
    img_batch = img_batch.to(device)

    with torch.no_grad():
        target_pred = model(img_batch)[0]

    boxes = target_pred["boxes"]  # (M,4)
    scores = target_pred["scores"]
    labels = target_pred["labels"]

    score_thresh = 0.35  # 0.3-0.5 recommended
    # NOTE: Overlap filter only applies to bounding boxes of the same class unless batched=False
    boxes, scores, labels = apply_nms(
        boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh
    )
    boxes, scores, labels = apply_nms(
        boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh, batched=False
    )

    target_pred["boxes"] = boxes
    target_pred["scores"] = scores
    target_pred["labels"] = labels 

    return img, target_pred