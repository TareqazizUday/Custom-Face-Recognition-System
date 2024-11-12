# Custom Face Recognition System

This project implements a face recognition system using the EfficientNet model with TensorFlow. It identifies known faces from a dataset, classifies images, and dynamically adds new faces through the webcam. The project includes scripts for model training, predictions on new images, and real-time recognition with a webcam.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
  - [Prediction](#prediction)
  - [Real-time Face Recognition](#real-time-face-recognition)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running Predictions](#running-predictions)
  - [Real-time Face Recognition](#real-time-face-recognition)
- [Adding New Faces](#adding-new-faces)
- [Model Architecture](#model-architecture)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

## Project Structure

- **training.py**: Script to train the EfficientNet model on a custom dataset.
- **prediction.py**: Script to predict the class of faces in images, with confidence thresholds for "unknown" faces.
- **main.py**: Main script to perform real-time face recognition with a webcam, allowing for dynamic addition of new faces.

## Requirements

Install the required libraries before running the scripts:

```bash
pip install -r requirements.txt
```

## Setup

### Dataset Preparation
1. **Download the Dataset**:
   - Download the celebrity face dataset from [Kaggle](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset).
   - Organize your training and validation images into subdirectories under `Dataset/train` and `Dataset/valid`, where each folder represents a class (a person’s name).

2. **Training the Model**:
   - Run `training.py` to train the model on your custom dataset.
   - Adjust `PARENT_DATA_DIR` in `training.py` to point to your dataset location.
   - Model architecture and weights are saved upon training completion.

3. **Prediction**:
   - Use `prediction.py` to classify images in a specified folder.
   - Update the `dataset_classes` list with names corresponding to the trained classes.
   - Modify the paths for model architecture and weights in `prediction.py` to match your setup.

4. **Real-time Face Recognition**:
   - Run `main.py` to start real-time recognition using a webcam.
   - Press `n` to add a new face and `q` to quit.
   - Faces are saved in `Dataset/train` for future training.

## Usage

### Training the Model

To train the model, use the following command:

```bash
python training.py
```

This script:
- Loads the dataset from the specified directories.
- Builds and trains the model using EfficientNet B0.
- Plots training history and saves the model architecture and weights.

### Running Predictions

To classify images, run:

```bash
python prediction.py
```

This script:
- Loads the trained model and processes each image in the input folder.
- Classifies faces based on the trained dataset and identifies "unknown" faces with low confidence scores.

### Real-time Face Recognition

To use the webcam for face recognition, run:

```bash
python main.py
```

The script:
- Starts the webcam feed.
- Detects faces in real-time and classifies them based on trained classes.
- Allows you to add new faces dynamically by pressing `n` and inputting a name.

## Adding New Faces

When running `main.py`, press `n` to add a new face. You will be prompted to enter a name, and the face will be saved in the dataset for future recognition.

**Note**: Adding a new face only saves the image. To recognize the new face, retrain the model using `training.py`.

## Model Architecture

This project uses EfficientNet B0, a deep learning model that achieves high accuracy with efficient computation. The model’s output layer is customized to classify faces based on the provided dataset.

## Future Improvements

- **Automated Retraining**: Implement code to automatically retrain the model after new faces are added.
- **Confidence Thresholding**: Adjust the `confidence_threshold` to improve "unknown" detection accuracy.
- **Data Augmentation**: Apply transformations to improve model robustness on diverse face images.

## License

This project is licensed under the MIT License.

## Contact

For questions, contact Tareq Aziz Uday at tareqazizuday20@gmail.com.
