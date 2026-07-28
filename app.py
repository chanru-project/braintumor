import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "brain_tumor_cnn.keras"
DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    # A couple of fallback strategies, since .keras files saved with a different
    # TF/Keras version than what's installed can hit deserialization errors.
    last_error = None
    for kwargs in ({"compile": False, "safe_mode": False}, {"compile": False}, {}):
        try:
            return tf.keras.models.load_model(MODEL_PATH, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise last_error


def detect_input_size(model, fallback=(180, 180)):
    """Read the real (H, W) the model expects straight from its own config,
    instead of trusting a hardcoded constant that can silently go stale."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape and len(shape) == 4 and shape[1] and shape[2]:
            return int(shape[1]), int(shape[2])
    except Exception:  # noqa: BLE001
        pass
    return fallback


def model_has_builtin_rescaling(model):
    """If the model already contains a Rescaling(1./255)-style layer, don't
    divide by 255 again in preprocessing — that double-normalizes and produces
    garbage predictions."""
    try:
        for layer in model.layers:
            if "rescaling" in layer.__class__.__name__.lower():
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def load_class_names():
    if os.path.exists("class_names.json"):
        try:
            with open("class_names.json", "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


def load_metrics():
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"test_accuracy": "Not available"}


def preprocess_image(image, img_size, normalize):
    image = image.convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image, dtype=np.float32)
    if normalize:
        img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image, class_names, img_size, normalize):
    img_array = preprocess_image(image, img_size, normalize)
    prediction = model.predict(img_array, verbose=0)

    if len(prediction.shape) == 2 and prediction.shape[1] == 1:
        confidence_raw = float(prediction[0][0])
        predicted_class = "tumor" if confidence_raw >= 0.5 else "no_tumor"
        confidence = round(confidence_raw * 100, 2) if confidence_raw >= 0.5 else round((1 - confidence_raw) * 100, 2)
        labels = ["no_tumor", "tumor"]
        probs = [
            round((1 - confidence_raw) * 100, 2),
            round(confidence_raw * 100, 2)
        ]
    else:
        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = class_names[predicted_index]
        confidence = round(float(np.max(prediction[0])) * 100, 2)
        labels = class_names
        probs = [round(float(p) * 100, 2) for p in prediction[0]]

    return predicted_class, confidence, labels, probs


st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to classify it with a local brain tumor model file.")

model = load_model()
class_names = load_class_names()
metrics = load_metrics()

if model is None:
    st.error("brain_tumor_cnn.keras file is not available locally.")
    st.info("Add your trained model file next to app.py to enable predictions.")
    st.stop()

# Auto-detect the correct input size and normalization from the model itself,
# instead of relying on a hardcoded constant that can mismatch the actual model.
IMG_SIZE = detect_input_size(model)
NORMALIZE = not model_has_builtin_rescaling(model)
st.caption(f"Model expects **{IMG_SIZE[0]}×{IMG_SIZE[1]}** input "
           f"({'normalizing 0-255 → 0-1 in the app' if NORMALIZE else 'model rescales internally, app sends raw pixels'}).")

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except Exception as e:  # noqa: BLE001
        st.error(f"The uploaded file could not be read as an image: {e}")
        st.stop()

    st.subheader("Uploaded Image")
    st.image(image, caption="Selected MRI image", use_container_width=True)

    try:
        with st.spinner("Predicting tumor class..."):
            predicted_class, confidence, labels, probs = predict_image(
                model, image, class_names, IMG_SIZE, NORMALIZE
            )

        st.success("Prediction completed successfully.")

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence}%")

        test_accuracy = metrics.get("test_accuracy", "Not available")
        if test_accuracy != "Not available":
            st.write(f"**Model Test Accuracy:** {test_accuracy}%")
        else:
            st.write("**Model Test Accuracy:** Not available")

        df = pd.DataFrame({
            "Class": labels,
            "Probability (%)": probs
        })

        st.subheader("Class Probabilities")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.bar_chart(df.set_index("Class"), use_container_width=True)

        st.info("Confidence is for this uploaded image. Accuracy is the overall model performance on a test dataset.")

    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.caption(
            "If this still complains about a shape mismatch, the model file itself may be "
            "corrupted or from an incompatible Keras version. Re-save it with "
            "model.save('brain_tumor_cnn.keras') using the same TensorFlow/Keras version "
            "installed here."
        )
else:
    st.info("Upload a PNG, JPG, or JPEG brain MRI image to get a prediction.")