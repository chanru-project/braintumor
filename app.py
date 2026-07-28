
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Explorer", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root { --ink: #182B3A; --accent: #2E6F9E; --bg: #F4F8FB; }
    .stApp { background: linear-gradient(180deg, #F4F8FB 0%, #EAF2F8 100%); }
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] li, [data-testid="stAppViewContainer"] label,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
    [data-testid="stWidgetLabel"] p { color: var(--ink) !important; }
    h1, h2, h3, h4 { color: var(--ink) !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #182B3A 0%, #24425A 100%); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #F4F8FB !important;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #D7E4EE; border-left: 6px solid var(--accent);
        padding: 14px 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(24,43,58,0.08);
    }
    div[data-testid="stAlert"] { background-color: #EAF4FF !important; border: 1px solid #9CC5E8; border-radius: 8px; }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span, div[data-testid="stAlert"] div { color: #182B3A !important; }
    .stTabs [data-baseweb="tab"] { background-color: #FFFFFF; border-radius: 10px 10px 0 0; padding: 10px 18px; border: 1px solid #D7E4EE; }
    .stTabs [data-baseweb="tab"] p { color: var(--ink) !important; }
    .stTabs [aria-selected="true"] { background-color: var(--accent) !important; }
    .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }
    .stButton>button { background-color: var(--accent); color: #FFFFFF !important; border-radius: 8px; border: none; padding: 0.5em 1.2em; font-weight: 600; }
    .stButton>button p { color: #FFFFFF !important; }
    .stButton>button:hover { background-color: #1F4E6E; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_COLORS = {"glioma": "#E4572E", "meningioma": "#F3A712", "pituitary": "#2E6F9E", "healthy": "#3BA55D"}

# --------------------------------------------------------------------------------------
# SIDEBAR: dataset location
# --------------------------------------------------------------------------------------
st.sidebar.header("🧠 Dataset")
_expected_classes = {"glioma", "healthy", "meningioma", "pituitary"}
_cwd_subfolders = {p.name for p in Path(".").iterdir() if p.is_dir()} if Path(".").exists() else set()
default_dir = "." if _expected_classes.issubset(_cwd_subfolders) else "archive"
data_dir = st.sidebar.text_input(
    "Dataset folder (contains glioma/ healthy/ meningioma/ pituitary/)",
    value=default_dir,
)
data_path = Path(data_dir)


@st.cache_data(show_spinner="Scanning dataset...")
def scan_dataset(root: str):
    root_path = Path(root)
    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()]) if root_path.exists() else []
    records = []
    for c in classes:
        files = sorted((root_path / c).glob("*.jpg")) + sorted((root_path / c).glob("*.png"))
        for f in files:
            records.append({"class": c, "path": str(f)})
    return pd.DataFrame(records), classes


if not data_path.exists():
    st.warning(
        f"Folder `{data_dir}` was not found next to the app. Update the path in the sidebar "
        "to point at the folder that contains `glioma/`, `healthy/`, `meningioma/`, `pituitary/`."
    )
    st.stop()

df, classes = scan_dataset(str(data_path))
if df.empty:
    st.warning("No images found in that folder. Check the path and try again.")
    st.stop()

st.sidebar.markdown("---")
class_filter = st.sidebar.multiselect("Filter classes", classes, default=classes)
sample_size = st.sidebar.slider("Images to sample for pixel-statistics plots", 50, 500, 150, step=50)
df_f = df[df["class"].isin(class_filter)]

st.title("🧠 Brain Tumor MRI — Explorer & Classifier")
st.caption(
    "Exploratory analysis and a trainable CNN classifier (with accuracy reporting) for brain MRI scans "
    "across **glioma**, **meningioma**, **pituitary** tumors, and **healthy** scans."
)

tab_overview, tab_samples, tab_stats, tab_train, tab_predict = st.tabs(
    ["📊 Overview", "🖼️ Sample Images", "📈 Pixel Statistics", "🤖 Train & Accuracy", "🔮 Predict"]
)

# --------------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------------------------------------------
with tab_overview:
    counts = df_f["class"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total images", f"{len(df_f):,}")
    c2.metric("Classes", f"{df_f['class'].nunique()}")
    c3.metric("Largest class", counts.index[0] if len(counts) else "-")
    c4.metric("Smallest class", counts.index[-1] if len(counts) else "-")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Class distribution")
        fig = px.bar(counts, x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map=CLASS_COLORS, text=counts.values,
                     labels={"x": "Class", "y": "Number of images"})
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown("#### Class balance")
        fig = px.pie(counts, names=counts.index, values=counts.values, hole=0.5,
                     color=counts.index, color_discrete_map=CLASS_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The **healthy** class images are stored at a different resolution (225×225) than the "
        "tumor classes (512×512) in the raw files. The training pipeline resizes everything to a "
        "common size, so this doesn't bias the model, but it's worth knowing when inspecting raw files."
    )

# --------------------------------------------------------------------------------------
# TAB 2: SAMPLE IMAGES
# --------------------------------------------------------------------------------------
with tab_samples:
    st.markdown("### Sample scans per class")
    n_per_class = st.slider("Images per class", 2, 8, 4)
    for c in class_filter:
        st.markdown(f"**{c.title()}**")
        subset = df_f[df_f["class"] == c]
        sample_paths = subset["path"].sample(min(n_per_class, len(subset)), random_state=42).tolist()
        cols = st.columns(len(sample_paths))
        for col, p in zip(cols, sample_paths):
            with col:
                st.image(Image.open(p), use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB 3: PIXEL STATISTICS
# --------------------------------------------------------------------------------------
with tab_stats:
    st.markdown("### Pixel-intensity statistics (sampled)")
    st.caption(
        f"Computed from a random sample of up to {sample_size} images per class to keep this fast."
    )

    @st.cache_data(show_spinner="Computing pixel statistics...")
    def compute_stats(paths_by_class: dict, n: int):
        rows = []
        for cls, paths in paths_by_class.items():
            sample = random.sample(paths, min(n, len(paths)))
            for p in sample:
                img = Image.open(p).convert("L").resize((128, 128))
                arr = np.asarray(img, dtype=np.float32)
                rows.append({
                    "class": cls,
                    "mean_intensity": arr.mean(),
                    "std_intensity": arr.std(),
                    "brightness_p90": np.percentile(arr, 90),
                })
        return pd.DataFrame(rows)

    paths_by_class = {c: df_f[df_f["class"] == c]["path"].tolist() for c in class_filter}
    stats_df = compute_stats(paths_by_class, sample_size)

    colA, colB = st.columns(2)
    with colA:
        fig = px.box(stats_df, x="class", y="mean_intensity", color="class",
                     color_discrete_map=CLASS_COLORS, points=False)
        fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_title="Mean pixel intensity")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.box(stats_df, x="class", y="std_intensity", color="class",
                     color_discrete_map=CLASS_COLORS, points=False)
        fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_title="Intensity std. dev. (contrast)")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(stats_df, x="mean_intensity", color="class", barmode="overlay",
                        color_discrete_map=CLASS_COLORS, nbins=40, opacity=0.6)
    fig.update_layout(plot_bgcolor="white", title="Distribution of mean brightness by class")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB 4: TRAIN CLASSIFIER + ACCURACY
# --------------------------------------------------------------------------------------
with tab_train:
    st.markdown("## 🤖 Train a CNN Classifier")
    st.write(
        "Trains an image classifier (transfer learning on **MobileNetV2**) on your dataset folder "
        "and reports test accuracy, per-class accuracy, and a confusion matrix. Needs `tensorflow` "
        "installed (see `requirements.txt`). A GPU speeds this up a lot but isn't required for a few epochs."
    )

    st.markdown("### Load an already-trained model instead")
    st.caption("If you already have a saved `.keras` model in this folder, load it instantly instead of retraining.")
    existing_models = sorted(str(p) for p in Path(".").glob("*.keras"))
    if existing_models:
        model_to_load = st.selectbox("Saved model file", existing_models)
        load_img_size = st.number_input("Image size the saved model was trained at", min_value=32, max_value=512,
                                         value=160, step=8)
        if st.button("📂 Load saved model"):
            try:
                import tensorflow as tf
            except ImportError:
                st.error("TensorFlow isn't installed. Run `pip install tensorflow` and try again.")
                st.stop()

            loaded_model = None
            last_error = None
            # Try a few loading strategies, since .keras files saved with a different
            # TF/Keras version than what's installed locally can hit deserialization bugs.
            for kwargs in ({"compile": False, "safe_mode": False}, {"compile": False}, {}):
                try:
                    loaded_model = tf.keras.models.load_model(model_to_load, **kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e

            if loaded_model is None:
                st.error(
                    "Couldn't load this `.keras` file. This usually means it was saved with a "
                    "different TensorFlow/Keras version than the one installed here "
                    f"(`tensorflow=={tf.__version__}`).\n\n"
                    f"Underlying error: `{type(last_error).__name__}: {last_error}`\n\n"
                    "**Fixes to try:**\n"
                    "- Match TensorFlow/Keras versions between the machine that trained the model "
                    "and this one.\n"
                    "- Re-save the model on the training machine with "
                    "`model.save('model.keras')` using the *same* Keras version installed here.\n"
                    "- Or just train a fresh model using the section below instead."
                )
            else:
                st.session_state["mri_model"] = loaded_model
                st.session_state["mri_class_names"] = classes
                st.session_state["mri_img_size"] = load_img_size
                st.success(f"Loaded `{model_to_load}`. Head to the **Predict** tab to try it, or train "
                           "fresh below to also get accuracy metrics.")
    else:
        st.caption("No `.keras` files found in this folder.")

    st.markdown("---")
    st.markdown("### ...or train a new one from scratch")
    img_size = st.select_slider("Image size (px, square)", options=[96, 128, 160, 224], value=160)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    epochs = st.slider("Epochs", 1, 30, 8)
    val_split = st.slider("Validation split", 0.1, 0.4, 0.2, step=0.05)
    fine_tune = st.checkbox("Fine-tune base model after warmup (slower, usually more accurate)", value=False)

    if st.button("🚀 Train model"):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
        except ImportError:
            st.error(
                "TensorFlow isn't installed in this environment. Run "
                "`pip install tensorflow` (see requirements.txt) and try again."
            )
            st.stop()

        with st.spinner("Loading dataset..."):
            train_ds = tf.keras.utils.image_dataset_from_directory(
                str(data_path), validation_split=val_split, subset="training", seed=42,
                image_size=(img_size, img_size), batch_size=batch_size,
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(data_path), validation_split=val_split, subset="validation", seed=42,
                image_size=(img_size, img_size), batch_size=batch_size,
            )
            class_names = train_ds.class_names
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds_p = train_ds.prefetch(AUTOTUNE)
            val_ds_cached = val_ds.cache().prefetch(AUTOTUNE)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(class_names), activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        progress_bar = st.progress(0.0, text="Training warmup head...")

        class StProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs} — "
                                       f"acc={logs.get('accuracy', 0):.2f}, val_acc={logs.get('val_accuracy', 0):.2f}")

        history = model.fit(train_ds_p, validation_data=val_ds_cached, epochs=epochs,
                             callbacks=[StProgress()], verbose=0)

        if fine_tune:
            st.write("Fine-tuning base model...")
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            fine_epochs = max(3, epochs // 2)
            ft_progress = st.progress(0.0, text="Fine-tuning...")

            class FtProgress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    ft_progress.progress((epoch + 1) / fine_epochs,
                                          text=f"Fine-tune epoch {epoch + 1}/{fine_epochs}")

            history_ft = model.fit(train_ds_p, validation_data=val_ds_cached, epochs=fine_epochs,
                                    callbacks=[FtProgress()], verbose=0)
            for k in history.history:
                history.history[k] += history_ft.history[k]

        st.session_state["mri_model"] = model
        st.session_state["mri_class_names"] = class_names
        st.session_state["mri_img_size"] = img_size

        hist_df = pd.DataFrame(history.history)
        hist_df["epoch"] = range(1, len(hist_df) + 1)

        colA, colB = st.columns(2)
        with colA:
            fig = px.line(hist_df, x="epoch", y=["accuracy", "val_accuracy"], markers=True,
                          labels={"value": "Accuracy", "variable": ""})
            fig.update_layout(plot_bgcolor="white", title="Accuracy over training")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.line(hist_df, x="epoch", y=["loss", "val_loss"], markers=True,
                          labels={"value": "Loss", "variable": ""})
            fig.update_layout(plot_bgcolor="white", title="Loss over training")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Accuracy reporting ----
        with st.spinner("Evaluating on validation set..."):
            val_loss, val_acc = model.evaluate(val_ds_cached, verbose=0)

            y_true, y_pred = [], []
            for images, labels in val_ds:
                preds = model.predict(images, verbose=0)
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(np.argmax(preds, axis=1).tolist())

            from sklearn.metrics import confusion_matrix, classification_report

            cm = confusion_matrix(y_true, y_pred)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        st.success(f"✅ Overall validation accuracy: **{val_acc*100:.2f}%**  (loss: {val_loss:.3f})")

        acc_cols = st.columns(len(class_names))
        for col, cname, acc in zip(acc_cols, class_names, per_class_acc):
            col.metric(f"{cname} accuracy", f"{acc*100:.1f}%")

        st.markdown("#### Confusion matrix")
        fig = px.imshow(cm, text_auto=True, x=class_names, y=class_names,
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Precision / recall / F1 by class")
        report_df = pd.DataFrame(report).transpose().loc[class_names]
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    if "mri_model" not in st.session_state:
        st.caption("No model trained yet in this session — click **Train model** above to get started.")

# --------------------------------------------------------------------------------------
# TAB 5: PREDICT ON A NEW IMAGE
# --------------------------------------------------------------------------------------
with tab_predict:
    st.markdown("## 🔮 Classify a new MRI scan")
    if "mri_model" not in st.session_state:
        st.info("Train (or load) a model in the **Train & Accuracy** tab first, then come back here.")
    else:
        uploaded = st.file_uploader("Upload an MRI scan (jpg/png)", type=["jpg", "jpeg", "png"])
        use_sample = st.checkbox("...or use a random sample image from the dataset instead")
        img = None
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
        elif use_sample:
            sample_path = df_f["path"].sample(1, random_state=None).iloc[0]
            img = Image.open(sample_path).convert("RGB")
            st.caption(f"Sampled: {sample_path}")

        if img is not None:
            img_size = st.session_state["mri_img_size"]
            class_names = st.session_state["mri_class_names"]
            model = st.session_state["mri_model"]

            resized = img.resize((img_size, img_size))
            arr = np.expand_dims(np.array(resized), axis=0)
            preds = model.predict(arr, verbose=0)[0]
            result = pd.Series(preds, index=class_names).sort_values(ascending=False)

            colA, colB = st.columns([1, 1.3])
            with colA:
                st.image(img, caption="Input scan", use_container_width=True)
            with colB:
                st.success(f"Predicted class: **{result.index[0]}** ({result.iloc[0]*100:.1f}% confidence)")
                fig = px.bar(result, x=result.values, y=result.index, orientation="h",
                             color=result.index, color_discrete_map=CLASS_COLORS,
                             labels={"x": "Predicted probability", "y": ""})
                fig.update_layout(showlegend=False, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "⚠️ This is a demo classifier for learning purposes only — not a medical diagnostic tool. "
                "Any real clinical decision must involve a qualified radiologist."
            )