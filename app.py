import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import time


# Load Pretrained Model (Fix: Avoid compiling to prevent 'lr' error)
@st.cache_resource
def load_colorization_model(model_path):
    return load_model(model_path, compile=False)


# Function to preprocess frame before feeding into the model
def preprocess_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame_gray = cv2.resize(frame_gray, (512, 512))  # Resize for model
    frame_gray = frame_gray.astype("float32") / 255.0  # Normalize
    frame_gray = np.expand_dims(frame_gray, axis=0)  # Add batch dimension
    frame_gray = np.expand_dims(frame_gray, axis=-1)  # Add channel dimension
    return frame_gray


# Function to colorize a single frame
def colorize_frame(model, frame):
    processed_frame = preprocess_frame(frame)
    colorized_frame = model.predict(processed_frame)[0]  # Predict
    colorized_frame = np.clip(colorized_frame * 255, 0, 255).astype(np.uint8)  # Convert to image
    return cv2.resize(colorized_frame, (frame.shape[1], frame.shape[0]))  # Resize to original


# Streamlit GUI
st.title("Real-Time Video Colorization")
st.sidebar.title("Settings")

# Upload Model
uploaded_model = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"])

if "stop" not in st.session_state:
    st.session_state.stop = False  # Track if user stopped video

if uploaded_model:
    model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())
    model = load_colorization_model(model_path)

    # Video Source Selection
    video_source = st.sidebar.radio("Choose Input", ("Live Webcam", "Upload Video"))

    if "cap" not in st.session_state:
        st.session_state.cap = None  # Store video capture across re-runs

    if video_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload a Grayscale Video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            st.session_state.cap = cv2.VideoCapture(temp_file.name)  # Read from uploaded file
    else:
        st.session_state.cap = cv2.VideoCapture(0)  # Use webcam

    # Stop button
    if st.sidebar.button("Stop Video", key="stop_button"):
        st.session_state.stop = True

    # Display Video
    frame_placeholder = st.empty()

    # Ensure cap is available before using it
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        while not st.session_state.stop:
            ret, frame = st.session_state.cap.read()
            if not ret:
                break

            # Apply colorization
            colorized_frame = colorize_frame(model, frame)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2RGB)

            # Display the colorized frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Introduce a small delay to avoid excessive re-runs
            time.sleep(0.03)

        st.session_state.cap.release()
        st.session_state.cap = None  # Reset capture after stopping

else:
    st.warning("Please upload a model (.h5) to continue.")
