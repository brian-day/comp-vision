from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from typing import Tuple, Literal

# TODO: Move basis positions into csv file
type PositionClass = Literal[
    "Pipette Tips",
    "Nested Pipette Tips",
    "SBS Plate",
    "SBS Heater/Shaker",
    "SBS Heater/Cooler",
    "Lid Stack",
    "MALDI",
    "2mL Tube",
    "50 mL Tube"
]
type Position = Tuple[PositionClass, int, Tuple[float, float]]


def plot_annotated_hamilton_deck(img, positions):
    if isinstance(img, str):
        img = Path(img)
    img = Image.open(img).convert("RGB")
    width, height = img.size

    positions_df = pd.read_csv(positions)
    xs = np.array(positions_df["center_x"])*width
    ys = np.array(positions_df["center_y"])*height

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, c="red", s=50, edgecolor="white", linewidth=1)
    plt.imshow(img)
    plt.show()


def main():
    # hamilton_left = Path("hamilton_left.png")
    # positions_left = Path("positions_basis_left.csv")
    # plot_annotated_hamilton_deck(hamilton_left, positions_left)

    hamilton_right = Path("hamilton_right.png")
    positions_right = Path("positions_basis_right.csv")
    plot_annotated_hamilton_deck(hamilton_right, positions_right)


if __name__ == "__main__":
    main()