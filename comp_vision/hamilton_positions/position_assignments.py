from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from typing import Tuple, Literal

from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
import torch
from torchvision import tv_tensors


type PositionClass = Literal[
    "Pipette Tips",
    "Nested Pipette Tips",
    "SBS Plate",
    "SBS Heater/Shaker",
    "SBS Heater/Cooler",
    "Lid Stack",
    "MALDI",
    "2mL Tube",
    "50 mL Tube",
]
type Position = Tuple[PositionClass, int, Tuple[float, float]]


def boxes_to_centers(img, boxes):
    # Use bounding box converter to recast in CXCYWH format
    boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))  # type: ignore
    converter = T.ConvertBoundingBoxFormat("CXCYWH")
    boxes = converter(boxes)

    # Convert back to pure tensor and return just CXCY values
    boxes_tens = T.ToPureTensor()(boxes)

    return boxes_tens[:, :2]


def get_deck_positions_and_types(img, positions_file):
    # img = Image.open(img).convert("RGB")  # If plan is to use a basis img... just adjust the 
    # # sizes in the positions file? If resolution can change, should use the target image instead.
    # # Alternatively, could report normalized bbox centers.
    width, height = F.get_size(img)
    positions_df = pd.read_csv(positions_file)
    xs = torch.tensor(positions_df["center_x"]) * width
    ys = torch.tensor(positions_df["center_y"]) * height

    position_types = list(positions_df["class"])
    centers = torch.stack([xs, ys], dim=1).to(torch.float32)

    return centers, position_types


def get_nearest_deck_position(centers_items, centers_plates):
    # NOTE: Tensor dypes must be identical for torch.cdist
    dists = torch.cdist(centers_items, centers_plates)  # [N,2], [M,2] -> [N,M]
    # TODO: Consider adding a distance check in case the detected object is not near enough to any
    # valid plate positions. That said, all detected objects should likely be assigned something.
    min_values, min_indicies = torch.min(dists, dim=1)

    return min_values, min_indicies


def boxes_to_deck_positions(img, boxes, positions_file):
    box_centers = boxes_to_centers(img, boxes)
    position_centers, position_types = get_deck_positions_and_types(img, positions_file)
    _, nearest_position_indicies = get_nearest_deck_position(box_centers, position_centers)
    detected_position_types = [position_types[i] for i in nearest_position_indicies]

    return detected_position_types

    
def plot_annotated_hamilton_deck(img, positions):
    if isinstance(img, str):
        img = Path(img)
    img = Image.open(img).convert("RGB")
    width, height = img.size

    positions_df = pd.read_csv(positions)
    xs = np.array(positions_df["center_x"]) * width
    ys = np.array(positions_df["center_y"]) * height

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, c="red", s=50, edgecolor="white", linewidth=1)
    plt.imshow(img)
    plt.show()
    plt.close()


def plot_box_centers(img, positions, point_labels):
    xs = np.array(positions[:,0])
    ys = np.array(positions[:,1])

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, c="red", s=50, edgecolor="white", linewidth=1)
    # t = img.as_subclass(torch.Tensor)  # [C,H,W]
    arr = img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
    if arr.dtype == np.float32 or arr.max() <= 1.0:
        plt.imshow(np.clip(arr, 0.0, 1.0))
    else:
        plt.imshow(np.clip(arr, 0, 255).astype(np.uint8))

    for i, txt in enumerate(point_labels):
        plt.annotate(txt, (xs[i], ys[i]))

    plt.show()


def main():
    script_dir = Path(__file__).parent

    hamilton_left = script_dir / "hamilton_left.png"
    positions_left = script_dir / "positions_basis_left.csv"
    plot_annotated_hamilton_deck(hamilton_left, positions_left)

    hamilton_right = script_dir / "hamilton_right.png"
    positions_right = script_dir / "positions_basis_right.csv"
    plot_annotated_hamilton_deck(hamilton_right, positions_right)

    plate_types, centers = get_deck_positions_and_types(hamilton_left, positions_left)


if __name__ == "__main__":
    main()
