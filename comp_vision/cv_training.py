"""New tutorial for bounding boxes using torchvision resnet 50.
Loosely based on the tutorial for the PennFudan dataset (tv_object_detection.py), but without
segmentation masks or superfluous modules. Data is in YOLO format.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchmetrics.detection import MeanAveragePrecision

from .cv_typing import (
    TargetDict,
    TargetDictPureTensor,
    BoxedImageTransform,
)


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
        `images` directory, containing images of the same file extension (.jpg or .png)
        `labels` directory, containing text files of normalized bounding boxed
        `classes.txt` file listing, in order, all possible classes.
    """

    root: Path
    transforms: BoxedImageTransform
    img_files: List[Path]
    box_files: List[Path]
    classes: List[str]  # List of all classes, excluding background
    num_classes: int  # Number of classes, including background, which is necessary for the model

    def __init__(
        self,
        root: Path | str,
        transforms: Optional[BoxedImageTransform],
        img_extension: str = ".jpg",
    ):
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
        img = remove_alpha_channel(img)

        labels, boxes = read_box_file(box_path)
        boxes = reformat_bounding_boxes(img, boxes)

        target_dict: TargetDict = {"image_id": idx, "labels": labels, "boxes": boxes}

        img_tens, target_dict_pure_tens = self.transforms(img, target_dict)

        return img_tens, target_dict_pure_tens

    def __len__(self) -> int:
        return len(self.img_files)

    def get_non_transformed_item(self, idx: int) -> Tuple[tv_tensors.Image, TargetDict]:
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        img = read_image(str(img_path))  # later versions of torchvision accept Paths directly
        img = tv_tensors.Image(img)
        img = remove_alpha_channel(img)

        labels, boxes = read_box_file(box_path)
        boxes = reformat_bounding_boxes(img, boxes)

        target_dict: TargetDict = {"image_id": idx, "labels": labels, "boxes": boxes}

        return img, target_dict


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
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.1))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    # Ensures no invalid bounding boxes (post-augmenation) are passed into the model
    transforms.append(T.SanitizeBoundingBoxes())

    # Convert images / boxes to pure tensors for model compatibility.
    # Careful with when special types are needed vs. pure tensors.
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    return T.Compose(transforms)


def remove_alpha_channel(img: tv_tensors.Image) -> tv_tensors.Image:
    # Remove alpha channel if present.
    if img.shape[0] == 4:
        img = tv_tensors.Image(img[:3, :, :])
    
    return img


def reformat_bounding_boxes(img, boxes) -> tv_tensors.BoundingBoxes:
    # NOTE: We expect unnormalized bounding boxes, so apply that transform here.
    # Applying tensor math to a BoundingBoxes object will convert it back to a standard tensor.
    h, w = F.get_size(img)
    boxes_tens = boxes * torch.tensor([w, h, w, h])
    boxes = tv_tensors.BoundingBoxes(boxes_tens, format="CXCYWH", canvas_size=F.get_size(img))  # type: ignore
    converter = T.ConvertBoundingBoxFormat("XYXY")
    boxes = converter(boxes)

    return boxes


def get_resnet50_model(dataset: BoundingBoxDataset) -> FasterRCNN:
    # Load the backbone with state-of-the-art weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        # NOTE: Model handles resizing, no need for a separate transform.
        # Adjusting min/max sizes can help improve performance depending on feature size relative
        # to image size. Also note, aspect ratio is preserved during resizing.
        min_size=600,
        max_size=1000,
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.num_classes)

    return model


def collate_fn(batch):
    # Used to avoid shape issues.
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()  # type: ignore - torch vision abstracts stuff weirdly here.
        optimizer.step()

        print(f"Batch #{batch} Loss: {losses}")


def train_model(
    dataset: BoundingBoxDataset,
    epochs: int = 5,
    state_dict_file: Optional[str | Path] = None,
    model_file_out: str = "model.pth",
    device="cpu",
):
    print(f"Evaluating on {device}")

    # Create dataloader from dataset
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
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
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Construct a learning rate scheduler
    # NOTE: Not strictly necessary, but useful for efficient training
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        print(f"----- Epoch #{epoch} -----")
        train_one_epoch(model, optimizer, data_loader, device)
        # Update the learning rate
        lr_scheduler.step()

    # TODO: Add datetime to model if no name provided to provent accidnetal model overwrite.
    torch.save(model.state_dict(), model_file_out)


def validate_model(dataset: BoundingBoxDataset, state_dict_file: str | Path, device="cpu"):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    model = get_resnet50_model(dataset)
    state_dict = torch.load(state_dict_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # Redundant to with torch.no_grad()?

    metric = MeanAveragePrecision(class_metrics=True)
    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        target = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        preds = model(images)

        metric.update(preds, target)

    results = metric.compute()

    print("Overall mAP:", results["map"])
    print("mAP at 0.50 IoU:", results["map_50"])
    print("mAP per Class:", results["map_per_class"])

