import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow import keras
import efficientnet.tfkeras as efn

# Paths for saving new faces and loading models
NEW_FACES_FOLDER = 'F:/Pb_project/Face_recognation/Custom_Face_Recognition_System/Dataset/train'
MODEL_ARCHITECTURE_PATH = "F:/Pb_project/Face_recognation/Custom_Face_Recognition_System/model/model_architecture.json"
MODEL_WEIGHTS_PATH = "F:/Pb_project/Face_recognation/Custom_Face_Recognition_System/model/model_weights.weights.h5"

# Load the model
with open(MODEL_ARCHITECTURE_PATH, "r") as json_file:
    model_json = json_file.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(MODEL_WEIGHTS_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

IMG_SIZE = 64
dataset_classes = ['Leonardo DiCaprio', 'Nicole Kidman', 'Angelina Jolie', 'Tom Cruise']  # List of known classes
confidence_threshold = 0.98

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    img = Image.fromarray(image)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32').reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    return img_array

def predict_image(image, confidence_threshold=0.98):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    
    if confidence < confidence_threshold:
        predicted_class = "unknown"
    else:
        predicted_class = dataset_classes[predicted_class_index]
    
    return predicted_class, confidence

def add_new_face(face_image, name):
    """
    Save a new face to the dataset and update model to recognize it.
    """
    global dataset_classes
    person_folder = os.path.join(NEW_FACES_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)
    img_count = len(os.listdir(person_folder))
    img_path = os.path.join(person_folder, f"{name}_{img_count + 1}.jpg")
    face_image_pil = Image.fromarray(face_image)
    face_image_pil.save(img_path)
    
    # Add new class label
    if name not in dataset_classes:
        dataset_classes.append(name)
        print(f"Added new class '{name}' to the model.")
    
    # Placeholder: To retrain or fine-tune the model in a real scenario,
    # you would need to reload and train with the new data here.

if __name__ == "__main__":
    # Create folder to store new faces if it doesn't exist
    os.makedirs(NEW_FACES_FOLDER, exist_ok=True)
    
    # Start the webcam
    cap = cv2.VideoCapture(0)
    adding_new_face = False
    new_face_name = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from the camera.")
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if adding_new_face:
                add_new_face(face_img, new_face_name)
                adding_new_face = False  # Turn off adding mode after one capture
                print(f"New face added for {new_face_name}")
                continue

            predicted_class, confidence = predict_image(face_img)
            label = f"{predicted_class}: {confidence:.2f}" if predicted_class != "unknown" else "unknown"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Webcam - Press 'q' to quit, 'n' to add a new face", frame)
        
        # Handle user input for adding new faces
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('n'):  # Add a new face
            adding_new_face = True
            new_face_name = input("Enter name for the new face: ")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
