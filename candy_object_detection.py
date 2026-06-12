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
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as nnF


DATASET_PATH = "datasets/candy_data_14DEC24/"
IMG_FILE_EXT = ".jpg"
MODEL_FILE = "models/finetuned_candy_model.pth"

# DATASET_PATH = "datasets/coin_data_12DEC30/"
# IMG_FILE_EXT = ".JPG"
# MODEL_FILE = "models/finetuned_coin_model.pth"

# DATASET_PATH = "datasets/wellplate_data/"
# IMG_FILE_EXT = ".png"
# MODEL_FILE = "models/finetuned_wellplate_model.pth"

# Add addtioanl type which ahndles transofrms of iamges with bounding boxes.
type ImageTransform = Callable[[Image.Image], torch.Tensor]


class TargetDict(TypedDict):
    image_id: int
    boxes: tv_tensors.BoundingBoxes
    labels: torch.Tensor  # int32
    # original_size will be added by transforms if included.
    original_size: NotRequired[torch.Tensor]  # int32


class TargetDictPureTensor(TypedDict):
    image_id: int
    boxes: torch.Tensor  # float64
    labels: torch.Tensor  # int32
    # original_size will be added by transforms if included.
    original_size: NotRequired[torch.Tensor]  # int32


class RecordOriginalImageSize(torch.nn.Module):
    """Store the original image dimensions before resizing.

    This transform can be inserted before a resize operation so downstream code can
    recover original coordinates for bounding boxes later.
    """

    def forward(self, image: tv_tensors.Image, target: TargetDict):
        height, width = F.get_size(image)
        target["original_size"] = torch.tensor([height, width], dtype=torch.int32)
        return image, target


# TODO: Fix type annotations.
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


type BoxedImageTransform = Callable[
    [tv_tensors.Image, TargetDict], Tuple[torch.Tensor, TargetDictPureTensor]
]


# NOTE: YOLO models implictly handle object / background distinguishing, and thus classes are
# 0-indexed. FasterRNN however needs an explicit background class (always 0), and thus classes are
# 1-indexed. If bounding box label data is in YOLO format, we must increment to match FasterRNN
# format. The following code is written assuming FasterRNN model architecture, but the sample data
# is in YOLO format. Model architecture will assumed to be fixed, but data format flexible, thus
# any conversions specific to the input data format will be behind a flag.


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

    def __init__(self, root: Path | str, transforms: Optional[BoxedImageTransform], img_extension: str = IMG_FILE_EXT):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        # Use minimimal set of transforms if None provided for type safety.
        self.transforms = transforms if transforms else generate_transform()
        self.img_files = sorted((self.root / "images").glob(f"*{img_extension}"))
        self.box_files = sorted((self.root / "labels").glob("*.txt"))

        # Limit to 8 images and train to check configuration...
        # self.img_files = self.img_files[0:64]
        # self.box_files = self.box_files[0:64]

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


# TODO: Add brief description of each transform.
# TODO: Research better set of image transforms for training.
def generate_transform(train: bool = False) -> BoxedImageTransform:
    transforms = []
    # ResNet50 Model has rescaling baked in... Allow it to handle this...
    # transforms.append(RecordOriginalImageSize())
    # transforms.append(T.Resize(size=(640, 640)))
    if train:
        transforms.append(T.RandomHorizontalFlip(1.0))
        transforms.append(T.RandomVerticalFlip(0.1))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    # Ensures no invalid bounding boxes (post augmenation) are passed into the model
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


# weights and weights backbone not the same...
def get_resnet50_model(dataset):
    # -----------
    # from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    # # 1. Load a ResNet-101 FPN backbone pre-trained on ImageNet
    # backbone = resnet_fpn_backbone('resnet101', weights='DEFAULT', pretrained=True)
    
    # # 2. Construct the Faster R-CNN model using the custom backbone
    # model = FasterRCNN(backbone=backbone, num_classes=dataset.num_classes)

    # -----------
    # 1. Load the backbone with state-of-the-art weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=600,
        max_size=1000
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore

    # (Option 1, Non Weighted) Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.num_classes)

    # -----------
    # (Option 1, Weighted) Replace the box_predictor with your weighted version
    # class_weights = generate_class_balanced_weights(dataset)
    # model.roi_heads.box_predictor = WeightedFastRCNNPredictor(
    #     in_features, 
    #     dataset.num_classes, 
    #     weights=class_weights
    # )

    return model


def collate_fn(batch):
    # Used to avoid shape issues.
    return tuple(zip(*batch))


def _test_forward_method():
    model = get_resnet50_model(12)
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))
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


def plot_training_data(idx):
    # NOTE: This includes applies transforms to both iamge and bounding boxes.
    # NOTE: Candy Dataset appears to include to boxes which need to be rotated relative to raw image.
    candy_data = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))
    image, target_dict = candy_data[idx]

    # NOTE: draw_bounding_boxes explicitly expect that boxes use XYXY format.
    converter = T.ConvertBoundingBoxFormat("XYXY")
    new_boxes = converter(target_dict["boxes"])

    # Given the code below, image and box can be pure tensors, which is what would be returned by
    # the model. Be careful with when an image/box must be in specialized format vs pure tensor format.
    image = T.ToPureTensor()(image)
    new_boxes = T.ToPureTensor()(new_boxes)
    class_names_true = [candy_data.classes[i-1] for i in target_dict["labels"]]
    output_image = draw_bounding_boxes(
        image, target_dict["boxes"], class_names_true,
        colors="blue", width=5, font="Arial", font_size=48)

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    # plt.show()
    plt.savefig(str(Path(f"~/Desktop/train-data-plot/{idx}.png").expanduser()))
    plt.close()


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


def train_model(epochs: int = 5, state_dict_file: Optional[Path] = None):
    device = "cpu"
    print(f"Evaluating on {device}")

    # Create dataset and dataloader
    # TODO: Add training / test / validation split
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model and set device
    # weights = ResNet50_Weights.IMAGENET1K_V2
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

    torch.save(model.state_dict(), MODEL_FILE)


def validate_model(device="cpu"):
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform())
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_resnet50_model(dataset)
    state_dict = torch.load(MODEL_FILE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # Redundant to with torch.no_grad()?

    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision()
    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        preds = model(images)

        metric.update(preds, target)
    
    results = metric.compute()

    print("Overall mAP:", results["map"])
    print("mAP at 0.50 IoU:", results["map_50"])


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


def get_luminance(color_tuple):
    """Calculates relative luminance for an RGB tuple (R, G, B) with values 0-255."""
    r, g, b = [c / 255.0 for c in color_tuple]
    
    # Apply gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    
    return 0.2126 * r + 0.0722 * g + 0.0722 * b # Wait, let's look at the correct formula below


def get_contrast_ratio(color1, color2):
    """Calculates the contrast ratio between two RGB tuples."""
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    
    # Ensure L1 is the lighter color and L2 is the darker color
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (lighter + 0.055) / (darker + 0.055)


def generate_label_colormap(dataset):
    colors = cm.get_cmap('turbo', 12)
    colors = colors(np.linspace(0,1,dataset.num_classes-1))
    # Draw bounding boxes expects tuple of three ints, ranging from 0 to 255.
    colors = np.round(colors * 255)
    colors = [tuple(int(v) for v in color[0:3]) for color in colors]
    colors_dict = {label: colors[i] for (i, label) in enumerate(dataset.classes)}

    return colors_dict


def generate_label_text_colormap(label_color_map):
    white = (255, 255, 255)
    black = (0, 0, 0)

    label_text_color = {}
    for label, color in label_color_map.items():
        # lighter color must go first, hence the ordering of inputs
        white_contrast = get_contrast_ratio(white, color)
        black_contrast = get_contrast_ratio(color, black)

        if black_contrast > white_contrast:
            label_text_color[label] = "black"
        else:
            label_text_color[label] = "white"

    return label_text_color


# TODO: Factor out plotting portion of code...
def single_image_classify(idx, plot_transformed=False):
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform())

    # weights = ResNet50_Weights.IMAGENET1K_V2
    model = get_resnet50_model(dataset)
    state_dict = torch.load("models/finetuned_wellplate_model.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # Redundant to with torch.no_grad()?

    img, target_true = dataset[idx]
    img_tensor = T.ToPureTensor()(img)
    img_batch = img_tensor.unsqueeze(0)
    device = "cpu"
    model.to(device)
    img_batch = img_batch.to(device)

    with torch.no_grad():
        target_pred = model(img_batch)[0]
    
    boxes = target_pred['boxes']   # (M,4)
    scores = target_pred['scores']
    labels = target_pred['labels']
    score_thresh = 0.35  # 0.3-0.5 recommended
    # NOTE: Overlap filter only applies to bounding boxes of the same class unless batched=False
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh)
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh,
                                      batched=False)

    if not plot_transformed:
        img_raw, _ = dataset.get_non_transformed_item(idx)
        boxes_rescaled = recover_original_image_dimensions(boxes, F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        target_true["boxes"] = recover_original_image_dimensions(target_true["boxes"], F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        img = img_raw
        boxes = boxes_rescaled

    label_cmap = generate_label_colormap(dataset)
    label_text_cmap = generate_label_text_colormap(label_cmap)

    output_image = img
    # class_names_true = [dataset.classes[i-1] for i in target_true["labels"]]
    # output_image = draw_bounding_boxes(
    #     img, target_true["boxes"],
    #     colors="blue", width=1, font="Arial", font_size=12)
    class_names_pred = [dataset.classes[i-1] for i in labels]
    class_names_w_score = [f"{class_pred}: {score:.3f}" for (class_pred, score) in zip(class_names_pred, scores)]
    output_image = draw_bounding_boxes(
        output_image, boxes, labels=class_names_w_score, 
        colors=[label_cmap[label] for label in class_names_pred], width=3, font="Arial", font_size=20,
        fill_labels=True, label_colors=[label_text_cmap[label] for label in class_names_pred])

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


def rename_images_and_labels(root):
    if isinstance(root, str):
        root = Path(root)
    img_files = sorted((root / "images").glob("*.jpg"))
    box_files = sorted((root / "labels").glob("*.txt"))

    for i, (img, label) in enumerate(zip(img_files, box_files)):
        new_img_name = root / "images" / f"{i}.jpg"
        new_label_name = root / "labels" / f"{i}.txt"
        os.rename(img, str(new_img_name))
        os.rename(label, str(new_label_name))


# TODO: Implement IoU with handling for closest guess (Hungarian Algorithm?)
# Implement mAP, precision, recall, F1 score
# Hungarian Algorithm asses the overlap of all predicted boxes with true boxes and class labels 
# to assign map predections with truth and assess False Positives and False Negatives.

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torchvision.ops as ops
from torchmetrics.classification import MulticlassAveragePrecision
# https://blog.roboflow.com/object-detection-metrics/
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Matches GT and Pred boxes under missing/extra constraints using the Hungarian Algorithm.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    # Step 1: Calculate the N x M IoU Matrix
    iou_matrix = ops.box_iou(gt_boxes, pred_boxes).numpy()
    
    # Step 2: Convert to cost matrix for minimization (1 - IoU)
    cost_matrix = 1.0 - iou_matrix
    
    # Step 3: Find optimal 1-to-1 matching indices
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    matched_pairs = []
    unmatched_gt = set(range(len(gt_boxes)))
    unmatched_pred = set(range(len(pred_boxes)))
    
    # Step 4: Filter assignments by IoU threshold
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:
            matched_pairs.append({'gt_idx': gt_idx, 'pred_idx': pred_idx, 'iou': iou})
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)
            
    return matched_pairs, list(unmatched_gt), list(unmatched_pred)


def hung_alg_test():
    # --- EXAMPLE USAGE ---
    # 3 Ground Truth Boxes
    gt = torch.tensor([
        [10, 10, 50, 50],
        [60, 60, 100, 100],
        [120, 120, 150, 150]  # This one will be MISSING (no match)
    ], dtype=torch.float32)

    # 4 Predicted Boxes (1 extra/false positive)
    pred = torch.tensor([
        [12, 12, 48, 48],    # Matches GT 0
        [58, 62, 102, 98],   # Matches GT 1
        [200, 200, 250, 250], # EXTRA box (No GT close by)
        [11, 11, 49, 49]     # EXTRA prediction overlapping GT 0 (Filtered out by Hungarian)
    ], dtype=torch.float32)


    matches, missing_gt, extra_pred = match_boxes(gt, pred, iou_threshold=0.5)

    print("✅ MATCHED PAIRS:")
    for m in matches:
        print(f"   GT Box {m['gt_idx']} <--> Pred Box {m['pred_idx']} (IoU: {m['iou']:.2f})")

    print(f"\n❌ MISSING BOXES (False Negatives): {missing_gt}")
    print(f"⚠️ EXTRA BOXES (False Positives): {extra_pred}")


# TODO: Figure out how to handle the weighting. Per image? Per object? 
# I *think* per object, so batch images and flatten results.
# Yes, this is based on object occurence. Individual images needed for separatring true postitives
# from false positives / negatives.
# TODO: This should also be calcualted per class and overall.
def calculate_precision(true_pos, false_pos):
    return true_pos / (true_pos +  false_pos) 


def calculate_recall(true_pos, false_neg):
    return true_pos / (true_pos +  false_neg) 


def calculate_mean_average_precision():
    # Use torchmetrics.detection implementation
    pass


def calculate_f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)



def main():
    candy_data = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))


if __name__ == "__main__":
    # main()
    # _test_forward_method()
    # rename_images_and_labels(DATASET_PATH)
    # for i in range(0, 500):
    #     plot_training_data(i)
    # train_model()
    # train_model(state_dict_file=Path(MODEL_FILE))
    # for i in range(0,8):
    #     single_image_classify(i)

    # stats_tests()
    # hung_alg_test()
    # stats_test2()
    validate_model()
