import requests  # Review Module
import zipfile
from pathlib import Path  # Review Module
import os
import random
from PIL import Image  # Review Module

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from typing import List, Tuple, Dict, Union, Optional


def download_food_data(data_path: Path, image_path: Path) -> None:
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)


def main1():
    data_path = Path("datasets/")
    image_path = data_path / "pizza_steak_sushi"

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # download_food_data(data_path, image_path)
    # walk_through_dir(image_path)
    # print_random_image(image_path)
    train_transform, test_transform = generate_data_transforms()
    image_path_list = list(image_path.glob("*/*.png"))
    # plot_transformed_images(image_path_list, transform=train_transform, n=3)

    # # TorchVision Style: Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(
        root=train_dir,  # target folder of images
        transform=train_transform,  # transforms to perform on data (images)
        target_transform=None,
    )  # transforms to perform on labels (if necessary)

    # test_data = datasets.ImageFolder(root=test_dir,
    #                                  transform=test_transform)

    print(f"Train data:\n{train_data}")
    # print(f"Test data:\n{test_data}")

    class_names = train_data.classes
    class_dict = train_data.class_to_idx

    img, label = train_data[0][0], train_data[0][1]
    # print(f"Image tensor:\n{img}")
    # print(f"Image shape: {img.shape}")
    # print(f"Image datatype: {img.dtype}")
    # print(f"Image label: {label}")
    # print(f"Label datatype: {type(label)}")

    # Rearrange the order of dimensions
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    # Matplotlib expects differenet ordering of data, see print statements above
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(class_names[label], fontsize=14)
    # Custom Torch Dataset: Custom class subtyping generic dataset class
    # See Class above
    # train_data = ImageFolderCustom(targ_dir=train_dir, transform=train_transform)
    # test_data = ImageFolderCustom(targ_dir=test_dir, transform=test_transform)

    display_random_images(train_data, train_data.classes)

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=1,  # how many samples per batch?
        num_workers=1,  # how many subprocesses to use for data loading? (higher = more)
        shuffle=True,
    )  # shuffle the data?

    # test_dataloader = DataLoader(dataset=test_data,
    #                             batch_size=1,
    #                             num_workers=1,
    #                             shuffle=False) # don't usually need to shuffle testing data

    img, label = next(iter(train_dataloader))

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    # What if a pre-built Dataset creator like torchvision.datasets.ImageFolder() didn't exist?
    # Or one for your specific problem didn't exist?
    # Well, you could build your own.
    # To see this in action, let's work towards replicating torchvision.datasets.ImageFolder()
    # by subclassing torch.utils.data.Dataset (the base class for all Dataset's in PyTorch).
