"""New tutorial for bounding boxes using torchvision resnet 50.
Loosely based on the tutorial for the PennFudan dataset (tv_object_detection.py), but without
segmentation masks or superfluous modules. Data is in YOLO format.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, NotRequired, Tuple, Optional, Union, TypedDict
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as nnF


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

    def __init__(self, root: Path | str, transforms: Optional[BoxedImageTransform]):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        # Use minimimal set of transforms if None provided for type safety.
        self.transforms = transforms if transforms else generate_transform()
        self.img_files = sorted((self.root / "images").glob("*.jpg"))
        self.box_files = sorted((self.root / "labels").glob("*.txt"))
        # TODO: Add check that all images contain a corresponding box file.
        self.classes = read_classes_file(self.root / "classes.txt")
        self.num_classes = len(self.classes) + 1  # Add 1 for background class.

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, TargetDictPureTensor]:
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        img = read_image(str(img_path))  # later versions of torchvision accept Paths directly
        img = tv_tensors.Image(img)
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

        img_tens, target_dict_pure_tens = self.transforms(img, target_dict)

        return img_tens, target_dict_pure_tens

    def __len__(self) -> int:
        return len(self.img_files)

    def get_non_transformed_item(self, idx):
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        img = read_image(str(img_path))  # later versions of torchvision accept Paths directly
        img = tv_tensors.Image(img)
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
    transforms.append(RecordOriginalImageSize())
    transforms.append(T.Resize(size=(640, 640)))
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
    # Note: Classes assumed 0 = background, so must include a 0 weight item at the beginning of the
    # list (I think...? Check this...)

    class_count = {i: 0 for i in range(dataset.num_classes)}
    for _, target in dataset:
        for label in target["labels"]:
            class_count[label.item()] += 1  #type: ignore - labels is an integer tensor
    total = sum(class_count.values())

    # inverse weights so that misclassify more common objects are less penalized.
    # this effect is offset by frequent appearences.
    return torch.Tensor([total/v if v != 0 else 0 for v in list(class_count.values())])


# Create a custom box predictor (inheriting from the standard torchvision FastRCNNPredictor)
class WeightedFastRCNNPredictor(FastRCNNPredictor):
    def __init__(self, in_channels, num_classes, weights):
        super(WeightedFastRCNNPredictor, self).__init__(in_channels, num_classes)
        # Replace the default CrossEntropyLoss with a weighted one
        self.cls_loss_func = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


def get_resnet50_model(dataset):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore

    # (Option 1, Non Weighted) Replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # (Option 1, Weighted) Replace the box_predictor with your weighted version
    class_weights = generate_class_balanced_weights(dataset)
    model.roi_heads.box_predictor = WeightedFastRCNNPredictor(
        in_features, 
        dataset.num_classes, 
        weights=class_weights
    )

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def _test_forward_method():
    model = get_resnet50_model(12)
    dataset = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))
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
    candy_data = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))
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


# Unused, Incomplete
def train_model(epochs: int = 30, state_dict_file: Optional[Path] = None):
    device = "cpu"
    print(f"Evaluating on {device}")

    # Create dataset and dataloader
    # TODO: Add training / test / validation split
    dataset = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
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

    class_weights = generate_class_balanced_weights(dataset)

    for epoch in range(epochs):
        print(f"----- Epoch #{epoch} -----")
        # Train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, data_loader, device)
        train_one_epoch(model, optimizer, data_loader, device)
        # Update the learning rate
        lr_scheduler.step()

    torch.save(model.state_dict(), 'finetuned_candy_model.pth')


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


def single_image_classify(idx, plot_transformed=False):
    dataset = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform())

    model = get_resnet50_model(dataset.num_classes)
    state_dict = torch.load("finetuned_candy_model.pth", weights_only=True)
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
    score_thresh = scores[10]
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh)
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=0.5, score_thresh=score_thresh,
                                      batched=False)
    # NOTE: OVerlap filter only applies to bounding boxes of the same class!

    if not plot_transformed:
        img_raw, _ = dataset.get_non_transformed_item(idx)
        boxes_rescaled = recover_original_image_dimensions(boxes, F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        target_true["boxes"] = recover_original_image_dimensions(target_true["boxes"], F.get_size(img), F.get_size(img_raw)) #type: ignore - list[int] vs tuple(int, int)
        img = img_raw
        boxes = boxes_rescaled

    # class_names_true = [dataset.classes[i-1] for i in target_true["labels"]]
    output_image = draw_bounding_boxes(
        img, target_true["boxes"],
        colors="blue", width=1, font="Arial", font_size=12)
    class_names_pred = [dataset.classes[i-1] for i in labels]
    output_image = draw_bounding_boxes(
        output_image, boxes, class_names_pred, 
        colors="red", width=1, font="Arial", font_size=12)

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


def main():
    candy_data = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))
    dataset = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))


if __name__ == "__main__":
    # main()
    # _test_forward_method()
    # rename_images_and_labels("datasets/candy_data_14DEC24/")
    # for i in range(0, 500):
    #     plot_training_data(i)
    train_model()
    # train_model(state_dict_file=Path("finetuned_candy_model.pth"))
    # for i in range(0,1):
    #     single_image_classify(i)
