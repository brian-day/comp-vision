from pathlib import Path
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes

from cv_training import BoundingBoxDataset, generate_transform

def get_luminance(color_tuple) -> float:
    """Calculates relative luminance for an RGB tuple (R, G, B) with values 0-255."""
    r, g, b = [c / 255.0 for c in color_tuple]
    
    # Apply gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    
    return 0.2126 * r + 0.0722 * g + 0.0722 * b # Wait, let's look at the correct formula below


def get_contrast_ratio(color1, color2) -> float:
    """Calculates the contrast ratio between two RGB tuples."""
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    
    # Ensure L1 is the lighter color and L2 is the darker color
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (lighter + 0.055) / (darker + 0.055)


def generate_label_colormap(dataset) -> Dict[str, Tuple[int, int, int]]:
    colors = cm.get_cmap('turbo', 12)
    colors = colors(np.linspace(0,1,dataset.num_classes-1))
    # Draw bounding boxes expects tuple of three ints, ranging from 0 to 255.
    colors = np.round(colors * 255)
    colors = [tuple(int(v) for v in color[0:3]) for color in colors]
    colors_dict = {label: colors[i] for (i, label) in enumerate(dataset.classes)}

    return colors_dict


def generate_label_text_colormap(label_color_map) -> Dict[str, str]:
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