from cv_training import BoundingBoxDataset, generate_transform, train_model
from cv_classification import single_image_classify
from cv_plotting import plot_training_data

DATASET_PATH = "datasets/candy_data_14DEC24/"
IMG_FILE_EXT = ".jpg"
MODEL_FILE = "models/finetuned_candy_model.pth"

# DATASET_PATH = "datasets/coin_data_12DEC30/"
# IMG_FILE_EXT = ".JPG"
# MODEL_FILE = "models/finetuned_coin_model.pth"

# DATASET_PATH = "datasets/wellplate_data/"
# IMG_FILE_EXT = ".png"
# MODEL_FILE = "models/finetuned_wellplate_model.pth"


def main():
    dataset = BoundingBoxDataset(DATASET_PATH, generate_transform(train=True))

    # Display some training images
    for i in range(0, 5):
        plot_training_data(i)

    # Build and train a model from scratch
    train_model()

    # Classify some images with the model
    for i in range(0,5):
        single_image_classify(i)


if __name__ == "__main__":
    main()
