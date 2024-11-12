import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import efficientnet.tfkeras as efn

# Load the model architecture from JSON
with open("F:/Pb_project/Face_recognation/fac/model/model_architecture.json", "r") as json_file:
    model_json = json_file.read()

# Create the model from the loaded JSON
model = tf.keras.models.model_from_json(model_json)

# Load the model weights
model.load_weights("F:/Pb_project/Face_recognation/fac/model/model_weights.weights.h5")

# Compile the model (necessary before prediction)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

IMG_SIZE = 64
dataset_classes = ['Leonardo DiCaprio', 'Nicole Kidman', 'Angelina Jolie', 'Tom Cruise']  # Replace with actual classes
confidence_threshold = 0.6  # Set a confidence threshold for unknown detection

# Input folder path for multiple images
input_folder = 'F:/Pb_project/Face_recognation/fac/Test_Image'  # Folder containing input images

def preprocess_image(image_path):
    """Resize and normalize the image for prediction."""
    img = Image.open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32').reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    return img_array

def predict_image(image_path):
    """Predict the class and confidence, return 'unknown' if below threshold."""
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    
    if confidence < confidence_threshold:
        predicted_class = "unknown"
    else:
        predicted_class = dataset_classes[predicted_class_index]
    
    return predicted_class, confidence

# Process each image in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    
    # Skip non-image files
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Predict class and confidence
    predicted_class, confidence = predict_image(img_path)
    
    # Print results for each image
    print(f"Image: {img_name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}/n")
