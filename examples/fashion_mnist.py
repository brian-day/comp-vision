import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from timeit import default_timer as timer
from tqdm.auto import tqdm

from pathlib import Path

from util import train_model, eval_model, multiclass_stats

""" Computer Vision Libraries:
- torchvision * 
- torchvision.datasets
- torchvision.model (pretrained models)
- torchvision.transforms
- torch.utils.data.Dataset
- torch.utils.data.DataLoader

* many other torch libraries, ex. torchtext
"""

# Root Logger
import logging
import textwrap

def set_up_logger(log_file='log.txt', level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    logger.addHandler(file_handler)

    return logger

logger = set_up_logger()


def get_fashion_mnist_dataset():
    train_data = datasets.FashionMNIST('datasets/',
                                       train=True,                      # False grabs test dataset
                                       transform=transforms.ToTensor(), # Transform the Data
                                       target_transform=None,           # Transform the Labels
                                       download=True)
    test_data = datasets.FashionMNIST('datasets/',
                                      train=False,
                                      transform=transforms.ToTensor(),
                                      target_transform=None,
                                      download=True)
    return train_data, test_data


# Simple Linear Model
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
            out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
            out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Simple Non-Linear Model
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
            out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
            out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)

# Convolutional Neural Network (CNN)
class FashionMNISTModelV2(nn.Module):
    """Model architecture that replicates teh TinyVGG model from CNN explainer website.
    https://poloclub.github.io/cnn-explainer/

    CNNs are composed of blocks, which are just groups of layers.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                    out_features=output_shape)
        )

    def forward(self, x):
        # x = self.conv_block_1(x)
        # print(f"Output shape of conv_block_1: {x.shape}")
        # x = self.conv_block_2(x)
        # print(f"Output shape of conv_block_2: {x.shape}")
        # x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")

        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def model_timing(start, end, device=None):
    # TODO: Make this a decorator
    # print(f"Train Time on {device}: {total_time:.3f} seconds")
    return end - start


def main():
    BATCH_SIZE = 32
    RANDOM_SEED = 99

    logger.info(textwrap.dedent(
        f"""\
        Training model with the following parameters:
            BATCH_SIZE: {BATCH_SIZE}
            RANDOM_SEED: {RANDOM_SEED}
        """))

    # Set Device & Seed
    # device = 'mps' if torch.mps.is_available() else 'cpu'
    device='cpu'
    logger.info(f"Running on device: {device}")
    # mps is currently slower. Assuming this is because the base model is quite fast, and we
    # are using very few epochs, so the bottleneck is in copying the tensors to the device rather than evaluating
    # the model. I guess epochs don't matter here since we need to copy those tensors to device each batch/epoch.
    # So it's just (lack of) model complexity...
    torch.device(device)
    torch.manual_seed(RANDOM_SEED)

    # Generate Datasets, Convert to Loaders, and Set Device
    train_data, test_data = get_fashion_mnist_dataset()
    class_names = train_data.classes

    # Generate Model and Set Device
    # model_0 = FashionMNISTModelV0(
    # input_shape=28*28,
    # hidden_units=128,
    # output_shape=len(class_names)
    # )

    model_0 = FashionMNISTModelV2(
        input_shape=1,
        hidden_units=10,
        output_shape=len(class_names)
    )
    logger.info(model_0)
    # Train the Model - Update training functions with batching
    # Using tqdm for progress bar.
    start = timer()
    LEARNING_RATE = 0.01
    EPOCHS = 5

    train_model(model_0, train_data,
                loss_fxn=nn.CrossEntropyLoss(),
                optim_type=torch.optim.SGD,
                learning_rate=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                print_iter=1,
                # print_stats=True,
                # stats_fxn=multiclass_stats,
                print_test=False,
                test_dataset=test_data,
                device=device)

    end = timer()
    total_time = model_timing(start, end, device)
    logger.info(f"Train Time on {device}: {total_time:.3f} seconds")

    MODEL_PATH = Path("models")
    MODEL_NAME = "05_cnn_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

    torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Saving model to: {MODEL_SAVE_PATH}")


def format_float_dict(input):
    strout = []
    for key, val in input.items():
        strout.append(f"{key}: {val:.4f}")
        
    return '{' + ", ".join(strout) + '}'


def plot_predicitons(model, test_samples, test_labels, class_names):

    pred_logits = model(test_samples)
    pred_classes = pred_logits.argmax(dim=1)
    print(pred_classes)

    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="grey")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c='g')
        else:
            plt.title(title_text, fontsize=10, c='r')
        plt.xticks([])
        plt.yticks([])
        
    plt.show()


def make_confusion_matrix(model, test_data):
    """Leveraging torchmetrics. Requires pred and truth labels, as before.
    """
    test_data_loader = DataLoader(dataset=test_data,
                                    batch_size=32,
                                    shuffle=False)
    class_names = test_data.classes
    num_classes = len(class_names)

    pred_labels = []
    for data, labels in test_data_loader:
        pred_logits = model(data)
        pred_classes = pred_logits.argmax(dim=1)
        pred_labels.append(pred_classes)

    pred_labels_tensor = torch.cat(pred_labels)

    confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes)
    confmat_tens = confmat(preds=pred_labels_tensor,
                            target=test_data.targets)
    # print(confmat_tens)
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tens.numpy(),
        class_names=class_names,
        figsize=(10,7)
    )
    plt.show()


def testModel():
    BATCH_SIZE = 32
    # RANDOM_SEED = 99
    # torch.manual_seed(RANDOM_SEED)

    train_data, test_data = get_fashion_mnist_dataset()
    class_names = train_data.classes

    loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                        hidden_units=10,
                                        output_shape=len(class_names))

    MODEL_PATH = Path("models")
    MODEL_NAME = "05_cnn_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    # loss, stats = eval_model(
    # loaded_model_2,
    # test_data,
    # loss_fxn = nn.CrossEntropyLoss(),
    # batch_size = BATCH_SIZE,
    # stats_fxn = multiclass_stats)
    # print(f"Loss: {loss.item():.4f}, Stats: {format_float_dict(stats)}")

    # import random
    # rand_ints = [random.randint(0, len(test_data)) for i in range(9)]
    # # print(rand_ints)
    # test_samples = torch.stack([test_data[i][0] for i in rand_ints])
    # test_labels = [test_data[i][1] for i in rand_ints]
    # # print(test_labels)
    # plot_predicitons(loaded_model_2, test_samples, test_labels, class_names)

    make_confusion_matrix(loaded_model_2, test_data)

if __name__ == '__main__':
    # main()
    testModel()
