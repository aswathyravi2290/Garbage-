import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

# Load labels
with open('labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

# Streamlit app layout
st.title("Rice Type Classifier")
st.write("Upload an image of rice grains to classify its type.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((224, 224)))  # Resize to model's expected input size
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]

    # Display the result
    st.write(f"Predicted Type: **{predicted_label}**")
