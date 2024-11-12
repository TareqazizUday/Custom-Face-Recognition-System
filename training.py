import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(1000)

# Set paths for dataset
PARENT_DATA_DIR = 'F:/Pb_project/Face_recognation/fac/Dataset'
TRAIN_DATA_DIR = os.path.join(PARENT_DATA_DIR, 'train')
VALID_DATA_DIR = os.path.join(PARENT_DATA_DIR, 'valid')
# TEST_DATA_DIR = os.path.join(PARENT_DATA_DIR, 'test')

IMG_SIZE = 64

# Load class names
dataset_classes = [cls for cls in os.listdir(TRAIN_DATA_DIR)]
number_of_classes = len(dataset_classes)
print(f"Classes: {dataset_classes}")
print(f"Number of classes: {number_of_classes}")

def load_data(data_dir, img_size=IMG_SIZE):
    """Loads images from a directory and resizes them."""
    data = []
    for cls in dataset_classes:
        path = os.path.join(data_dir, cls)
        class_num = dataset_classes.index(cls)
        for img in tqdm(os.listdir(path), desc=f"Loading {cls}"):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_resized = cv2.resize(img_array, (img_size, img_size))
                data.append([img_resized, class_num])
            except Exception as e:
                pass
    return data

# Load and preprocess data
training_data = load_data(TRAIN_DATA_DIR)
validation_data = load_data(VALID_DATA_DIR)
# test_data = load_data(TEST_DATA_DIR)

# Separate features and labels
def prepare_data(data):
    X, Y = [], []
    for img, label in data:
        X.append(img)
        Y.append(label)
    X = np.array(X).astype('float32').reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    Y = np.array(Y)
    return X, Y

X_train, Y_train = prepare_data(training_data)
X_valid, Y_valid = prepare_data(validation_data)
# X_test, Y_test = prepare_data(test_data)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, Y_valid shape: {Y_valid.shape}")
# print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Build the EfficientNet B0 model
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=number_of_classes):
    enet = efn.EfficientNetb0(input_shape=input_shape, weights='imagenet', include_top=False)
    x = layers.GlobalMaxPooling2D()(enet.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    y = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=enet.input, outputs=y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

# Training the model
history = model.fit(
    x=X_train, y=Y_train,
    epochs=1,
    validation_data=(X_valid, Y_valid),
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Plotting training history
def plot_history(history):
    sns.set_theme()

    # Accuracy plot
    plt.figure(figsize=[8, 4])
    plt.plot(history.history['accuracy'], color="blue")
    plt.plot(history.history['val_accuracy'], color="magenta")
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

    # Loss plot
    plt.figure(figsize=[8, 4])
    plt.plot(history.history['loss'], color="blue")
    plt.plot(history.history['val_loss'], color="magenta")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

plot_history(history)

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train, Y_train)
valid_loss, valid_acc = model.evaluate(X_valid, Y_valid)
print(f'\nTrain Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}')
print(f'Validation Accuracy: {valid_acc:.2f}, Validation Loss: {valid_loss:.2f}')

# Classification report and confusion matrix
y_pred = np.argmax(model.predict(X_valid), axis=1)
print("\nClassification Report:")
print(classification_report(Y_valid, y_pred, digits=3, target_names=dataset_classes))

cm = confusion_matrix(Y_valid, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset_classes, yticklabels=dataset_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save the model architecture and weights
model_json = model.to_json()
with open("/content/drive/MyDrive/face_rec/model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("/content/drive/MyDrive/face_rec/model_weights.h5")
print("Saved model architecture and weights to disk")
