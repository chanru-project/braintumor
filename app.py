import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

ARCHIVE_PATH = "archive"
CLASS_DIRECTORIES = {
    "glioma": "glioma",
    "healthy": "healthly",
}
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FEATURE_SIZE = (64, 64)


@st.cache_resource
def load_reference_images():
    reference_vectors = []
    reference_labels = []
    class_counts = {}

    for label, directory_name in CLASS_DIRECTORIES.items():
        folder_path = os.path.join(ARCHIVE_PATH, directory_name)
        if not os.path.isdir(folder_path):
            class_counts[label] = 0
            continue

        image_names = sorted(
            name for name in os.listdir(folder_path)
            if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
        )
        class_counts[label] = len(image_names)

        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            try:
                with Image.open(image_path) as image:
                    reference_vectors.append(preprocess_image(image))
                reference_labels.append(label)
            except Exception:  # noqa: BLE001
                continue

    if not reference_vectors:
        return None, None, class_counts

    return np.vstack(reference_vectors), np.array(reference_labels), class_counts


def preprocess_image(image, img_size=FEATURE_SIZE):
    image = image.convert("L")
    image = image.resize(img_size)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return img_array.reshape(-1)


def predict_image(image, reference_vectors, reference_labels):
    query_vector = preprocess_image(image)
    distances = np.linalg.norm(reference_vectors - query_vector, axis=1)

    labels = list(CLASS_DIRECTORIES.keys())
    class_scores = {}
    for label in labels:
        label_distances = distances[reference_labels == label]
        if len(label_distances) == 0:
            class_scores[label] = 0.0
            continue
        class_scores[label] = 1.0 / (float(np.min(label_distances)) + 1e-6)

    total_score = sum(class_scores.values()) or 1.0
    probs = [round((class_scores[label] / total_score) * 100, 2) for label in labels]
    predicted_index = int(np.argmax(probs))
    predicted_class = labels[predicted_index]
    confidence = probs[predicted_index]
    return predicted_class, confidence, labels, probs


st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to classify it using the local glioma and healthy archive images.")

reference_vectors, reference_labels, class_counts = load_reference_images()

if reference_vectors is None or reference_labels is None:
    st.error("No reference images were found in the archive folder.")
    st.info("Add images inside archive/glioma and archive/healthly to enable predictions.")
    st.stop()

st.caption(
    f"Using {class_counts.get('glioma', 0)} glioma and {class_counts.get('healthy', 0)} healthy "
    f"reference images from archive/ at {FEATURE_SIZE[0]}x{FEATURE_SIZE[1]} grayscale resolution."
)

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
        with st.spinner("Comparing with archive images..."):
            predicted_class, confidence, labels, probs = predict_image(
                image, reference_vectors, reference_labels
            )

        st.success("Prediction completed successfully.")

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence}%")

        df = pd.DataFrame({
            "Class": labels,
            "Probability (%)": probs
        })

        st.subheader("Class Probabilities")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.bar_chart(df.set_index("Class"), use_container_width=True)

        st.info("This result is based on similarity to the small local archive image set, not on any large Keras model.")

    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.caption("Check that archive/glioma and archive/healthly contain readable MRI images.")
else:
    st.info("Upload a PNG, JPG, or JPEG brain MRI image to get a glioma or healthy prediction.")