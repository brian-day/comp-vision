## Overview

This module assists with the development of object detection models for various tasks in the lab. Presently, the applications for this work do not require real time detection, and as such, we use models which sacrifice speed for accuracy (though are still quite fast, just not instantaneous). The model employed here is the resnet50 model / FasterRCNN which is a two-stage model (object detection, then classification). We finetune these models with our own training data.

## Installation

The environment for this project is managed with uv. After cloning the project, run `uv sync` which will create an environment and install all necessary packages.

## Examples

This project contains three example datasets: candy data, coin data, and well plate data.

In order to run the example script, which will build and train a model, then detect objects and plot the results for 5 images, run the script as a module using `uv run python -m examples.candy_object_detection`. Running as a script rather than a module will lead to import errors.