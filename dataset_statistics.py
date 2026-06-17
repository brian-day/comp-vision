import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, Dict, List, NotRequired, Tuple, Optional, Union, TypedDict
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes

from cv_training import BoundingBoxDataset, generate_transform


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


def get_bounding_box_dataset_stats(dataset: BoundingBoxDataset):
    classes = dataset.classes
    class_count = {label: 0 for label in classes}
    sizes = []
    num_boxes = []
    for _, target in dataset:
        sizes.append(target["original_size"])  #type: ignore - original_size transform applied
        num_boxes.append(len(target["boxes"]))

        for label in target["labels"]:
            class_count[classes[label.item()-1]] += 1  #type: ignore - labels is an integer tensor
    
    # Plot Results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sizes_x= [s[0] for s in sizes]
    sizes_y= [s[1] for s in sizes]
    axs[0].plot(sizes_x, sizes_y, 'o')   # Possibly plot as histogram is sets of common sizes
    axs[0].set_title("Image Sizes")

    min_count, max_count = min(num_boxes), max(num_boxes)
    box_count = {i: 0 for i in range(min_count, max_count+1)}  # range is exclusive on upper bound
    for v in num_boxes:
        box_count[v] += 1
    axs[1].bar(box_count.keys(), box_count.values())
    axs[1].set_xticks(list(box_count.keys()))
    axs[1].set_title("Avg. Number of Boxes")

    axs[2].bar(class_count.keys(), class_count.values())
    axs[2].set_xticklabels(class_count.keys(), rotation=45, ha="right", rotation_mode="anchor")
    axs[2].set_title("Label Frequency")

    plt.tight_layout()
    plt.show()


def main():
    dataset = BoundingBoxDataset("datasets/candy_data_14DEC24/", generate_transform(train=True))
    get_bounding_box_dataset_stats(dataset)


if __name__ == "__main__":
    main()