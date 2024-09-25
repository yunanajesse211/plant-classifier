from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the DNN model for image classification
used_model = tf.keras.models.load_model("densenet_model.h5")

# Function to classify the image and return species class name (A, B, or C)
def image_classifier(image_selected):
    img_array = np.array(image_selected.resize((224, 224))) / 255.0  # Resize and normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    prediction = used_model.predict(img_array)
    prediction_probability = prediction[0]
    specie_class = np.argmax(prediction_probability)

    # Map the numeric class to class names A, B, C
    class_names = {0: "A", 1: "B", 2: "C"}
    return class_names[specie_class]

# Main App
st.title("PLANT SPECIE CLASSIFIER APP")
selected_file = st.file_uploader("Select image to upload", type=["png", "jpg", "dng", "jpeg"])

if selected_file is not None:
    image_holder = st.empty()
    selected_image = Image.open(selected_file)
    image_holder.image(selected_image, caption="Uploaded selected_image", use_column_width=True)

    st_writer = st.empty()
    st_writer.write("App is classifying the plant. Please wait...")

    # Call the classification function
    pred_class = image_classifier(selected_image)

    # Display the predicted species class (A, B, or C)
    st_writer.markdown(f"<h3 style='font-weight:bold; font-size: 30px;'>Predicted Class: {pred_class}</h3>", unsafe_allow_html=True)
