from pathlib import Path
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import torch
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes

from .cv_training import BoundingBoxDataset
from .cv_typing import TargetDictPureTensor


type LabelColorMap = Dict[str, Tuple[int, int, int]]
type LabelTextColorMap = Dict[str, str]


def get_luminance(color_tuple) -> float:
    """Calculates relative luminance for an RGB tuple (R, G, B) with values 0-255."""
    r, g, b = [c / 255.0 for c in color_tuple]

    # Apply gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

    return 0.2126 * r + 0.0722 * g + 0.0722 * b  # Wait, let's look at the correct formula below


def get_contrast_ratio(color1, color2) -> float:
    """Calculates the contrast ratio between two RGB tuples."""
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)

    # Ensure L1 is the lighter color and L2 is the darker color
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    return (lighter + 0.055) / (darker + 0.055)


def generate_label_colormap(dataset: BoundingBoxDataset) -> LabelColorMap:
    colors = cm.get_cmap("turbo", dataset.num_classes)
    colors = colors(np.linspace(0, 1, dataset.num_classes - 1))
    # Draw bounding boxes expects tuple of three ints, ranging from 0 to 255.
    colors = np.round(colors * 255)
    colors_tuple: list[Tuple[int, int, int]] = [
        tuple(int(v) for v in color[0:3]) for color in colors
    ]  # type: ignore
    colors_dict = {label: colors_tuple[i] for (i, label) in enumerate(dataset.classes)}

    return colors_dict


def generate_label_text_colormap(label_color_map: LabelColorMap) -> LabelTextColorMap:
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


def plot_bounding_boxes_from_dataset(
    dataset: BoundingBoxDataset,
    idx: int,
    *,
    target_dict: Optional[TargetDictPureTensor] = None,
    image_out: Optional[str | Path] = None,
    interactive: bool = False,
):
    # NOTE: This applies transforms to both image and bounding boxes.
    image, target_true = dataset[idx]
    image = T.ToPureTensor()(image)
    # Plot training data boxes if none provided
    # NOTE: If target_dict is provided, be careful that none of the transforms applied to the image
    # impact the boxes (e.g. random flip) since we fetch the image again here. If they should, use 
    # the plot from image function and provide all additional data.
    if not target_dict:
        target_dict = target_true 

    label_cmap = generate_label_colormap(dataset)
    label_text_cmap = generate_label_text_colormap(label_cmap)

    plot_bounding_boxes_from_image(
        image,
        target_dict,
        classes = dataset.classes,
        label_cmap = label_cmap,
        label_text_cmap = label_text_cmap,
        interactive = interactive,
        image_out = image_out
    )


def plot_bounding_boxes_from_image(
    image: torch.Tensor,
    target_dict,
    *,
    classes: list[str] = [],
    label_cmap: str | LabelColorMap = "blue",
    label_text_cmap: str | LabelTextColorMap = "white",
    interactive: bool = False,
    image_out: Optional[str | Path] = None,
):
    boxes = target_dict["boxes"]
    scores = target_dict["scores"] if "scores" in target_dict else None
    labels = target_dict["labels"]

    # Convert label indicies to class names if provided
    class_names = [classes[i - 1] for i in labels] if classes else labels
    # Add scores to bbox labels if included (i.e. Non-training data boxes)
    bbox_labels = [
        f"{class_pred}: {score:.3f}" for (class_pred, score) in zip(class_names, scores)
    ] if scores is not None else class_names

    # Format colors and label colors accoring to provided color_map
    colors = (
        [label_cmap[label] for label in class_names]
        if isinstance(label_cmap, dict)
        else label_cmap
    )
    label_colors = (
        [label_text_cmap[label] for label in class_names]
        if isinstance(label_cmap, dict)
        else label_cmap
    )

    output_image = draw_bounding_boxes(
        image,
        boxes,
        labels=bbox_labels,
        colors=colors,  # type: ignore
        label_colors=label_colors,  # type: ignore
        width=3,
        font="Arial",
        font_size=20,
        fill_labels=True,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(output_image.permute(1, 2, 0))
    if interactive:
        plt.show()
    if image_out:
        plt.savefig(image_out)
    plt.close()
