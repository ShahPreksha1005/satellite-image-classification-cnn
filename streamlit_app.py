import pandas as pd
import os
import shap
import numpy as np
import random
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title('Satellite Image Classification with CNN')

# Load dataset and display basic information
def load_dataset():
    data = pd.DataFrame(columns=['image_path', 'label'])
    
    labels = {
        '/path/to/cloudy': 'Cloudy',
        '/path/to/desert': 'Desert',
        '/path/to/green_area': 'Green_Area',
        '/path/to/water': 'Water'
    }

    for folder in labels:
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = labels[folder]
                data = pd.concat(
                    [data, pd.DataFrame({'image_path': [image_path], 'label': [label]})],
                    ignore_index=True
                )
    return data

data = load_dataset()
st.write("Dataset Loaded with", len(data), "images.")
st.write("Dataset Information:", data.info())

# Visualize class distribution
st.subheader("Class Distribution")
sns.countplot(y='label', data=data, palette='viridis')
plt.title("Class Distribution")
st.pyplot(plt)

# Display sample images from each class
st.subheader("Sample Images")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
classes = data['label'].unique()
for i, label in enumerate(classes):
    sample_path = data[data['label'] == label].iloc[0]['image_path']
    image = Image.open(sample_path)
    axes[i].imshow(image)
    axes[i].set_title(label)
    axes[i].axis('off')
plt.tight_layout()
st.pyplot(plt)

# Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data['label'], random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_data, x_col='image_path', y_col='label', 
                                                    target_size=(256, 256), class_mode='categorical', batch_size=32)
val_generator = val_test_datagen.flow_from_dataframe(val_data, x_col='image_path', y_col='label', 
                                                     target_size=(256, 256), class_mode='categorical', batch_size=32)
test_generator = val_test_datagen.flow_from_dataframe(test_data, x_col='image_path', y_col='label', 
                                                      target_size=(256, 256), class_mode='categorical', batch_size=32)

# Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st.write(model.summary())

# Train Model
st.subheader("Training the Model")
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate Model
st.subheader("Evaluate the Model")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(test_generator.class_indices.keys()))
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix")
st.pyplot(plt)

# Plot accuracy and loss curves
st.subheader("Training and Validation Accuracy/Loss")
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axs[0].plot(history.history['accuracy'], label='Training Accuracy')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title("Accuracy")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

# Loss plot
axs[1].plot(history.history['loss'], label='Training Loss')
axs[1].plot(history.history['val_loss'], label='Validation Loss')
axs[1].set_title("Loss")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")
axs[1].legend()

st.pyplot(plt)

# Save the model
model.save("custom_cnn_model.h5")
st.write("Model saved successfully.")

# Allow loading model for inference
st.subheader("Load and Use Model for Inference")
uploaded_file = st.file_uploader("Choose an image for prediction", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted class:", np.argmax(prediction))

# End of the Streamlit app
