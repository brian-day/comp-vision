"""New tutorial for bounding boxes using torchvision resnet 50.
Loosely based on the tutorial for the PennFudan dataset (tv_object_detection.py), but without
segmentation masks or superfluous modules. Data is in YOLO format.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, NotRequired, Tuple, Optional, Union, TypedDict
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection import MeanAveragePrecision

from comp_vision.cv_typing import TargetDict, TargetDictPureTensor, ImageTransform, BoxedImageTransform


class BoundingBoxDataset(torch.utils.data.Dataset):
    """The dataset should inherit from the standard torch.utils.data.Dataset class, and implement
    __len__ and __getitem__. The only specificity that we require is that the dataset __getitem__
    should return a tuple of:
        image: torchvision.tv_tensors.Image
        target: a dict containing (at least) the following fields
            boxes: torchvision.tv_tensor.BoundingBoxes of shape [N, 4], which is the coordinates of
                   N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
            labels: integer torch.Tensor of shape [N] of the label corresponding to each bounding box;
                    Note that background is always label 0.

    Root data directory is expected to contain the following:
        `images` directory, containing only png images
        `labels` directory, containing text files of normalized bounding boxed
        `classes.txt` file listing, in order, all possible classes.
    """

    root: Path
    transforms: BoxedImageTransform
    img_files: List[Path]
    box_files: List[Path]
    classes: List[str]  # List of all classes, excluding background
    num_classes: int  # Number of classes, including background, which is necessary for the model

    def __init__(self, root: Path | str, transforms: Optional[BoxedImageTransform],
                 img_extension: str = ".jpg"):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        # Use minimimal set of transforms if None provided for type safety.
        self.transforms = transforms if transforms else generate_transform()
        self.img_files = sorted((self.root / "images").glob(f"*{img_extension}"))
        self.box_files = sorted((self.root / "labels").glob("*.txt"))

        # TODO: Add check that all images contain a corresponding box file.
        self.classes = read_classes_file(self.root / "classes.txt")
        self.num_classes = len(self.classes) + 1  # Add 1 for background class.

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, TargetDictPureTensor]:
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        img = read_image(str(img_path))  # later versions of torchvision accept Paths directly
        img = tv_tensors.Image(img)

        # Remove alpha channel if present. TODO: Handle this more elegantly...
        if img.shape[0] == 4:
            img = tv_tensors.Image(img[:3, :, :])
        labels, boxes = read_box_file(box_path)

        # NOTE: We expect unnormalized bounding boxes, so apply that transform here.
        # Applying tensor math to a BoundingBoxes object will convert it back to a standard tensor.
        # TODO: Move this logic to a helper function and hide behind flag so that it's easy to
        # control whether boxes need normalization.
        h, w = F.get_size(img)
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = tv_tensors.BoundingBoxes(boxes, format="CXCYWH", canvas_size=F.get_size(img))  # type: ignore
        converter = T.ConvertBoundingBoxFormat("XYXY")
        boxes = converter(boxes)
        # boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))  # type: ignore

        target_dict: TargetDict = {"image_id": idx, "labels": labels, "boxes": boxes}

        img_tens, target_dict_pure_tens = self.transforms(img, target_dict)

        return img_tens, target_dict_pure_tens

    def __len__(self) -> int:
        return len(self.img_files)

    def get_non_transformed_item(self, idx):
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        img = read_image(str(img_path))  # later versions of torchvision accept Paths directly
        img = tv_tensors.Image(img)

        # Remove alpha channel if present. TODO: Handle this more elegantly...
        if img.shape[0] == 4:
            img = tv_tensors.Image(img[:3, :, :])
        labels, boxes = read_box_file(box_path)

        labels, boxes = read_box_file(box_path)

        # NOTE: We expect unnormalized bounding boxes, so apply that transform here.
        # Applying tensor math to a BoundingBoxes object will convert it back to a standard tensor.
        # TODO: Move this logic to a helper function and hide behind flag so that it's easy to
        # control whether boxes need normalization.
        h, w = F.get_size(img)
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = tv_tensors.BoundingBoxes(boxes, format="CXCYWH", canvas_size=F.get_size(img))  # type: ignore
        converter = T.ConvertBoundingBoxFormat("XYXY")
        boxes = converter(boxes)

        target_dict: TargetDict = {"image_id": idx, "labels": labels, "boxes": boxes}

        return img, target_dict


# Tried annotating with exact tensor type (torch.IntTensor and torch.FloatTensor) but pylance does
# not recognize the datatypes of the actual tensors, and assumes type mismatch.
# NOTE: We mostly work with bounding boxes in tv_tensors.BoundingBoxes format, but depending on use
# case, we may unnormalize the box data, which is image dependent. This function returns a plain tensor,
# and any additional transforms are up to the developer.
def read_box_file(
    file: Path, delimiter: str = " ", increment_labels: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    lines = file.read_text(encoding="utf-8").splitlines()

    labels = []
    boxes = []
    for line in lines:
        vals = line.split(delimiter)
        labels.append(int(vals[0]))
        boxes.append([float(v) for v in vals[1:]])

    labels = torch.as_tensor(labels, dtype=torch.int64)
    labels = labels + 1 if increment_labels else labels
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    # Swap the 2nd and 3rd columns of each individual box tensor to convert from
    # XXYY format to XYXY format.
    # boxes[:, [1, 2]] = boxes[:, [2, 1]]

    return labels, boxes


def read_classes_file(file: Path) -> List[str]:
    return file.read_text(encoding="utf-8").splitlines()


class RecordOriginalImageSize(torch.nn.Module):
    """Store the original image dimensions before resizing.

    This transform can be inserted before a resize operation so downstream code can
    recover original coordinates for bounding boxes later.
    """

    def forward(self, image: tv_tensors.Image, target: TargetDict):
        height, width = F.get_size(image)
        target["original_size"] = torch.tensor([height, width], dtype=torch.int32)
        return image, target
    

# TODO: Add brief description of each transform.
# TODO: Research better set of image transforms for training.
def generate_transform(train: bool = False) -> BoxedImageTransform:
    transforms = []
    transforms.append(RecordOriginalImageSize())
    # NOTE: Model has resizing baked in. Do not add separate transform.
    # transforms.append(T.Resize(size=(640, 640)))
    if train:
        transforms.append(T.RandomHorizontalFlip(1.0))
        transforms.append(T.RandomVerticalFlip(0.1))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    # Ensures no invalid bounding boxes (post-augmenation) are passed into the model
    transforms.append(T.SanitizeBoundingBoxes())

    # Convert images / boxes to pure tensors for model compatibility.
    # Careful with when special tpyes are needed vs. pure tensors.
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def generate_class_balanced_weights(dataset: BoundingBoxDataset):
    # To account for class imbalance in the training data, we can modify the loss function of the
    # model to more harshly penalize rare classes. This prevents the model from over prediciting the
    # common class, which is would otherwise learn to do during training.
    # Note: Classes assumed to have index 0 = background, so must include a small non-zero weight
    # for the background at the beginning of the weight tensor.
    class_count = {i: 0 for i in range(dataset.num_classes)}
    for _, target in dataset:
        for label in target["labels"]:
            class_count[label.item()] += 1  #type: ignore - labels is an integer tensor
    total = sum(class_count.values())

    # Inverse weights so that misclassifying more common objects are less penalized.
    # This effect is offset by frequent appearences.
    weights = [total/(v*dataset.num_classes) if v != 0 else 0 for v in list(class_count.values())]
    weights[0] = 0.1  # Manually set the background weight.
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    return weights_tensor


def get_resnet50_model(dataset):
    # Load the backbone with state-of-the-art weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        # NOTE: Model handles resizing, no need for a separate transform.
        # Adjusting min/max sizes can help improve performance depending on feature size relative
        # to image size. Also note, aspect ratio is preserved during resizing.
        min_size=600,
        max_size=1000
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.num_classes)

    return model


def collate_fn(batch):
    # Used to avoid shape issues.
    return tuple(zip(*batch))


def _test_forward_method(dataset: BoundingBoxDataset):
    model = get_resnet50_model(dataset.num_classes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,  # Figure out what this does...
    )

    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    print("-----")
    print(images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(images)  # Returns predictions
    print(predictions[0])

    output_image = draw_bounding_boxes(images[0], predictions[0]["boxes"], colors="red", width=5)
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()  # type: ignore - torch vision abstracts stuff weirdly here. 
        optimizer.step()

        print(f"Batch #{batch} Loss: {losses}")


def train_model(dataset: BoundingBoxDataset, epochs: int = 5,
                state_dict_file: Optional[str | Path] = None,
                model_file_out: str = "model.pth", device="cpu"):
    print(f"Evaluating on {device}")

    # Create dataloader from dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model and set device
    model = get_resnet50_model(dataset)
    if state_dict_file:
        state_dict = torch.load(state_dict_file, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)

    # Construct an optimizer
    # Only get parameters from non-frozen layers (i.e. requires grad)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Construct a learning rate scheduler
    # NOTE: Not strictly necessary, but useful for efficient training
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    for epoch in range(epochs):
        print(f"----- Epoch #{epoch} -----")
        train_one_epoch(model, optimizer, data_loader, device)
        # Update the learning rate
        lr_scheduler.step()

    # TODO: Add datetime to model if no name provided to provent accidnetal model overwrite.
    torch.save(model.state_dict(), model_file_out)


def validate_model(dataset: BoundingBoxDataset, state_dict_file: str|Path, device="cpu"):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_resnet50_model(dataset)
    state_dict = torch.load(state_dict_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # Redundant to with torch.no_grad()?

    metric = MeanAveragePrecision(class_metrics=True)
    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        preds = model(images)

        metric.update(preds, target)
    
    results = metric.compute()

    print("Overall mAP:", results["map"])
    print("mAP at 0.50 IoU:", results["map_50"])
    print("mAP per Class:", results["map_per_class"])


def apply_nms(boxes, scores, labels, *, iou_thresh=0.5, score_thresh=0.05, max_detections=100,
              batched=True):
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
    keep = (torchvision.ops.batched_nms(boxes, scores, labels, iou_thresh) 
        if batched
        else torchvision.ops.nms(boxes, scores, iou_thresh))
    keep = keep[:max_detections]
    return boxes[keep], scores[keep], labels[keep]


# TODO: Fix type annotations.
# TODO: Rename function for clarity
def recover_original_image_dimensions(
    boxes: torch.Tensor,
    current_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Scale resized bounding boxes back to the original image dimensions."""
    orig_height, orig_width = original_size
    resized_height, resized_width = current_size
    scale = torch.tensor(
        [orig_width / resized_width, orig_height / resized_height,
         orig_width / resized_width, orig_height / resized_height],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    return boxes * scale
