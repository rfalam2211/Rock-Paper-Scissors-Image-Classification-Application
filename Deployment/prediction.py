import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- Optimize Model Loading with Caching ---
# @st.cache_resource ensures the model is only loaded once when the app starts.
# This makes the app much faster during user interaction.
@st.cache_resource
def load_keras_model():
    """
    Function to load the Keras model. Attempts to load the improved model first.
    """
    # Attempt to load the new improved model
    base_path = os.path.dirname(__file__)
    model_path_improved = os.path.join(base_path, 'src', 'rock_paper_scissors_improved_model.h5')
    model_path_basic = os.path.join(base_path, 'src', 'rock_paper_scissors_model.h5')
    
    if os.path.exists(model_path_improved):
        model = load_model(model_path_improved)
        return model, 224 # EfficientNet size
    elif os.path.exists(model_path_basic):
        model = load_model(model_path_basic)
        return model, 150 # Basic CNN size
    else:
        return None, None


# --- Main Application Function ---
def run():
    # --- Title and Header ---
    st.header("Rock-Paper-Scissors Image Prediction")
    st.write("""
    Upload an image of your hand forming a rock, paper, or scissors gesture.
    The model will attempt to classify the gesture.
    """)

    # --- Load Model ---
    model, target_size = load_keras_model()

    # Check if model loaded successfully
    if model is None:
        st.error("Model could not be loaded. Ensure the model file (.h5) exists in the './src/' directory.")
        return


    # --- Image Upload Widget ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

    # --- Prediction Logic ---
    if uploaded_file is not None:
        # Display the image uploaded by user
        image_to_show = Image.open(uploaded_file)
        st.image(image_to_show, caption="Uploaded Image", use_container_width=True)
        st.write("") 

        # Button to trigger prediction
        if st.button("Perform Prediction"):
            with st.spinner("Model is thinking..."):
                # 1. Image Preprocessing
                # The image needs to be converted to the same format as during training.
                image_for_pred = image_to_show.resize((target_size, target_size)) # Adjust size dynamically

                img_array = np.array(image_for_pred)              # Convert to numpy array
                img_array = np.expand_dims(img_array, axis=0)     # Add batch dimension
                
                # Rescale only for the basic model. EfficientNet expects [0, 255]
                if target_size == 150:
                    img_array = img_array / 255.0

                # 2. Perform Prediction
                prediction = model.predict(img_array)
                score = tf.nn.softmax(prediction[0])

                # 3. Display Results
                class_names = ['Paper', 'Rock', 'Scissors']
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"Prediction Result: **{predicted_class}**")
                st.info(f"Confidence Score: **{confidence:.2f}%**")

if __name__ == "__main__":
    run()