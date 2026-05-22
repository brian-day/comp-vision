from typing import Callable, List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import textwrap
from timeit import default_timer as timer

import torch
from torchvision import datasets, transforms
from torchinfo import summary

from util import train_model, setup_logger, multiclass_stats, find_classes

import random

import numpy as np
import matplotlib.pyplot as plt


# Custom Types
type ImageTransform = Callable[[Image.Image], torch.Tensor]


# Not stricly necessary, can use built in torch class ImageFolder
# TODO: Add default to_tensor transform, Add better file extension handling
class ImageFolderCustom(torch.utils.data.Dataset):
    def __init__(self, targ_dir: Union[str, Path], transform: ImageTransform) -> None:
        if isinstance(targ_dir, str):
            targ_dir = Path(targ_dir)

        self.paths = list(targ_dir.glob("*/*.png"))
        # note: you'd have to update this if you've got .png's or .jpeg's
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

        self.color_mode: str = "RGB"

    # Overwrite the __len__() method
    # (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # Overwrite the __getitem__() method
    # (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.png
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        # if self.transform:
        img = self.transform(img)

        return img, class_idx

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path).convert(self.color_mode)

    def get_random_item(self) -> Tuple[Path, str]:
        idx = np.random.randint(len(self.paths))

        return self.paths[idx], self.paths[idx].parent.name


def generate_data_transforms(*, img_size: int = 64) -> Tuple[ImageTransform, ImageTransform]:
    """Ref for other transfrom types: https://docs.pytorch.org/vision/stable/transforms.html

    Returns:
        Object containing all the transforms to be performed.
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )

    return train_transforms, test_transforms


# Baseline Model (without data augmentation), TinyVGG Architecture
# Convolutional Neural Network (CNN)
class BaselineModel(torch.nn.Module):
    """Model architecture that replicates the TinyVGG model from CNN explainer website.
    https://poloclub.github.io/cnn-explainer/

    CNNs are composed of blocks, which are just groups of layers.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_units * 64 * 64, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


# TODO: Add more gpu device flexibility (cuda, mps, and cpu)
def single_image_classify(
    img_path: Path,
    transforms: ImageTransform,
    model: torch.nn.Module,
    classes: List[str],
    device="cpu",
    color_mode="RGB",
) -> str:
    img = Image.open(img_path).convert(color_mode)
    img_tensor = transforms(img)
    # 4. Add the batch dimension: [3, x, y] -> [1, 3, x, y]
    img_batch = img_tensor.unsqueeze(0)

    model.to(device)
    img_batch = img_batch.to(device)

    # 6. Predict
    with torch.no_grad():
        output = model(img_batch)

    # 7. Get the predicted class index
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_idx = int(torch.argmax(probabilities).item())  # Should not need to call int?

    return classes[predicted_idx]


# TODO: Make this a decorator
def model_timing(start: float, end: float) -> float:
    # print(f"Train Time on {device}: {total_time:.3f} seconds")
    return end - start


# ----- Plotting Functions for Development Only -----
def plot_single_image_classify(
    img_path: Path, true_class: str, pred_class: str, color_mode: str = "RGB"
) -> None:
    with Image.open(img_path).convert(color_mode) as f:
        plt.figure(figsize=(12, 4))
        plt.imshow(f)
        plt.axis("off")
        plt.title(f"True: {true_class}, Predicted: {pred_class}")
        plt.show()


def plot_random_images(
    dataset: ImageFolderCustom,
    classes: Optional[List[str]] = None,
    n: int = 5,
    display_shape: bool = False,
    seed: Optional[int] = None,
) -> None:

    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(
            f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(12, 4))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {list(targ_image_adjust.shape)}"
            plt.title(title, size=12)

    plt.show()


def plot_transformed_images(
    image_paths: List[Path],
    transform: ImageTransform,
    n: int = 3,
    seed: int = 42,
    color_mode: str = "RGB",
):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path).convert(color_mode) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

            plt.show()


# ----- Run as Main -----
def main():
    BATCH_SIZE: int = 32
    RANDOM_SEED: int = 99
    IMAGE_DIM: int = 256
    LEARNING_RATE: float = 0.01
    EPOCHS: int = 21

    logger = setup_logger()
    logger.info(
        textwrap.dedent(f"""\
        Training model with the following parameters:
            BATCH_SIZE: {BATCH_SIZE}
            RANDOM_SEED: {RANDOM_SEED}
        """)
    )

    # device = 'mps' if torch.mps.is_available() else 'cpu'
    device = "cpu"
    logger.info(f"Running on device: {device}")
    torch.device(device)
    torch.manual_seed(RANDOM_SEED)  # type: ignore

    train_transform, test_transform = generate_data_transforms(img_size=IMAGE_DIM)
    data_path = Path("datasets/")
    image_path = data_path / "Hamilton"
    train_dir = image_path
    train_data_simple = ImageFolderCustom(targ_dir=train_dir, transform=train_transform)

    class_names = train_data_simple.classes
    print(f"Classes: {class_names}")
    baseline_model = BaselineModel(input_shape=3, hidden_units=10, output_shape=len(class_names))

    # plot_random_images(train_data_simple, class_names)
    plot_transformed_images(train_data_simple.paths, train_transform)

    logger.info(summary(baseline_model, input_size=[1, 3, IMAGE_DIM, IMAGE_DIM]))

    start = timer()
    train_model(
        baseline_model,
        train_data_simple,
        print_iter=5,
        loss_fxn=torch.nn.CrossEntropyLoss(),
        epochs=EPOCHS,
        print_stats=True,
        stats_fxn=multiclass_stats,
    )

    end = timer()
    total_time = model_timing(start, end)
    logger.info(f"Train Time on {device}: {total_time:.3f} seconds")

    MODEL_PATH = Path("models")
    MODEL_NAME = "05_cnn_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(baseline_model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saving model to: {MODEL_SAVE_PATH}")

    print("-------")


def random_img_classify():
    IMAGE_DIM: int = 256

    # device = 'mps' if torch.mps.is_available() else 'cpu'
    device = "cpu"
    torch.device(device)
    # torch.manual_seed(RANDOM_SEED)  # type: ignore

    train_transform, test_transform = generate_data_transforms(img_size=IMAGE_DIM)
    data_path = Path("datasets/")
    image_path = data_path / "Hamilton"
    train_dir = image_path
    train_data_simple = ImageFolderCustom(targ_dir=train_dir, transform=train_transform)
    class_names = train_data_simple.classes

    model = BaselineModel(input_shape=3, hidden_units=10, output_shape=len(class_names))
    state_dict = torch.load("models/05_cnn_model.pth", weights_only=True)
    model.load_state_dict(state_dict)

    print("Classifying Random Image...")
    img_path, true_class = train_data_simple.get_random_item()
    pred_class = single_image_classify(img_path, test_transform, model, class_names)

    print(f"Classified {true_class} as {pred_class}.")
    plot_single_image_classify(img_path, true_class, pred_class)


if __name__ == "__main__":
    # main()
    random_img_classify()
