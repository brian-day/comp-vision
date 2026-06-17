from comp_vision.cv_training import BoundingBoxDataset, generate_transform, train_model
from comp_vision.cv_classification import single_image_classify
from comp_vision.cv_plotting import plot_training_data


"""
Candy and Coin Data from YouTube video for YOLO Bounding Boxes:
    https://www.youtube.com/watch?v=r0RspiLG260
And correspinding google colab notebook: 
    https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb

Downloading the candy and coin datasets:
    !wget -O /content/data.zip https://s3.us-west-1.amazonaws.com/evanjuras.com/resources/candy_data_06JAN25.zip # Candy dataset
    !wget -O /content/data.zip https://s3.us-west-1.amazonaws.com/evanjuras.com/resources/YOLO_coin_data_12DEC30.zip # Coin dataset
"""

DATASET_PATH = "datasets/candy_data/"
IMG_FILE_EXT = ".jpg"
MODEL_FILE = "models/finetuned_candy_model.pth"

# DATASET_PATH = "datasets/coin_data/"
# IMG_FILE_EXT = ".JPG"
# MODEL_FILE = "models/finetuned_coin_model.pth"

# DATASET_PATH = "datasets/wellplate_data/"
# IMG_FILE_EXT = ".png"
# MODEL_FILE = "models/finetuned_wellplate_model.pth"


def main():
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True), img_extension=IMG_FILE_EXT)

    # Display some training images
    for i in range(0, 5):
        plot_training_data(dataset, i, interactive=True)

    # Build and train a model from scratch
    train_model(dataset, model_file_out=MODEL_FILE, epochs=5)

    # Classify some images with the model
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(), img_extension=IMG_FILE_EXT)
    for i in range(0,5):
        single_image_classify(dataset, MODEL_FILE, i)


if __name__ == "__main__":
    main()
