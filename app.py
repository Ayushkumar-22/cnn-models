import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model("cnn_cifar10.h5")

# CIFAR-10 class labels
classes = ["✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat", "🦌 Deer",
           "🐶 Dog", "🐸 Frog", "🐴 Horse", "🚢 Ship", "🚚 Truck"]

# Streamlit UI - Title & Sidebar
st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

st.title("CIFAR-10 Image Classification")
st.markdown("### Upload an image and let the AI predict what it is!")

# Sidebar Info
st.sidebar.header("📌 How to Use:")
st.sidebar.write("1️⃣ Upload an image (JPG, PNG, JPEG).")
st.sidebar.write("2️⃣ The AI model will analyze the image.")
st.sidebar.write("3️⃣ You'll see the **top 3 predictions** with probabilities.")

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    top_3_indices = np.argsort(prediction)[-3:][::-1]  # Get top 3 predictions
    return [(classes[i], prediction[i]) for i in top_3_indices]

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🤖 AI is analyzing the image..."):
            predictions = predict(image)
        
        # Display Predictions
        st.success(f"🏷 **Top Prediction:** {predictions[0][0]} ({predictions[0][1]*100:.2f}%)")
        
        # Show top 3 predictions with a progress bar
        st.write("### 🔥 Top 3 Predictions:")
        for label, prob in predictions:
            st.write(f"🔹 **{label}**: {prob*100:.2f}%")
            st.progress(float(prob))

        # Display probabilities in a chart
        labels, probs = zip(*predictions)
        st.bar_chart({labels[i]: probs[i] for i in range(3)})
