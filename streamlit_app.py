import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from keras.models import load_model
# Load the model
model = load_model('keras_model.h5', compile = False)

# Load the labels
labels = ["Paper", "Glass"]

# Function to preprocess the uploaded image for prediction
def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize the image
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Material Classification App")
st.write("Upload an image, and the model will predict if it's Paper or Glass.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = labels[np.argmax(prediction)]

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
