import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Load the saved model
MODEL_PATH = "custom_cnn_model.h5"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Title and Description
st.title("Satellite Image Classification")
st.markdown("""
This app classifies satellite images into four categories:
1. Cloudy
2. Desert
3. Green Area
4. Water
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option:", ("Home", "Upload and Classify"))

# Home Section
if option == "Home":
    st.header("About the Project")
    st.markdown("""
    - This project uses a custom Convolutional Neural Network (CNN) to classify satellite images.
    - The model was trained on a dataset of images representing four categories: Cloudy, Desert, Green Area, and Water.
    """)

# Upload and Classify Section
elif option == "Upload and Classify":
    st.header("Upload and Classify")
    
    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image for the model
        image = image.resize((256, 256))
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by [Your Name]")
