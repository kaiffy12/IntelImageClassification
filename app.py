import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🏔️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Project links — update the GitHub/live-demo URLs after you create the repo
# ---------------------------------------------------------------------------
LINKS = [
    ("GitHub", "Source code", "https://github.com/kaiffy12/IntelImageClassification"),
    ("Live demo", "Try the hosted app", "https://intelimageclassification-kaif.streamlit.app/"),
    ("Dataset", "Kaggle Intel images", "https://www.kaggle.com/datasets/puneet6060/intel-image-classification"),
]

# Order must match the training label order (alphabetical, as produced by
# keras.utils.image_dataset_from_directory)
LABELS = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CLASS_INFO = {
    "buildings": {
        "name": "Buildings",
        "color": "#B45309",
        "tint": "#FDF3E7",
        "note": "Man-made structures: houses, offices and towers.",
    },
    "forest": {
        "name": "Forest",
        "color": "#15803D",
        "tint": "#E9F7EF",
        "note": "Dense tree cover and woodland scenes.",
    },
    "glacier": {
        "name": "Glacier",
        "color": "#0284C7",
        "tint": "#E8F4FD",
        "note": "Ice fields and frozen landscapes.",
    },
    "mountain": {
        "name": "Mountain",
        "color": "#57534E",
        "tint": "#F1EFEC",
        "note": "Peaks, ridges and high terrain.",
    },
    "sea": {
        "name": "Sea",
        "color": "#0E7490",
        "tint": "#E6F6FA",
        "note": "Open water: oceans, seas and coastlines.",
    },
    "street": {
        "name": "Street",
        "color": "#7C3AED",
        "tint": "#F1EBFE",
        "note": "Urban roads and city streets.",
    },
}

# ---------------------------------------------------------------------------
# Styling (same look as the Brain Tumor app)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');

:root {
    --ink: #12263A;
    --muted: #5B6B7B;
    --line: #DCE5EC;
    --paper: #FFFFFF;
    --wash: #F3F7FA;
    --teal: #0E8F8A;
    --teal-deep: #0A6F6B;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--ink);
}
.stApp {
    background:
        radial-gradient(900px 420px at 85% -10%, #D9EEF0 0%, rgba(217,238,240,0) 60%),
        var(--wash);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 780px; }

h1, h2, h3 { font-family: 'Outfit', 'Inter', sans-serif; letter-spacing: -0.01em; }

/* Hero */
.hero { padding: 0.5rem 0 1.4rem 0; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--paper); border: 1px solid var(--line);
    padding: 0.3rem 0.8rem; border-radius: 999px;
    font-size: 0.82rem; color: var(--muted); font-weight: 500;
}
.hero-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--teal); }
.hero h1 {
    font-size: 2.9rem; line-height: 1.05; font-weight: 700;
    margin: 1rem 0 0.7rem 0; color: var(--ink);
}
.hero p { font-size: 1.05rem; line-height: 1.6; color: var(--muted); max-width: 34rem; margin: 0; }

/* Quick link pills under the hero */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.2rem; }
.pill {
    display: inline-block; padding: 0.42rem 0.95rem; border-radius: 999px;
    background: var(--paper); border: 1px solid var(--line);
    color: var(--ink) !important; font-size: 0.85rem; font-weight: 500;
    text-decoration: none !important; transition: border-color .15s, color .15s;
}
.pill:hover { border-color: var(--teal); color: var(--teal-deep) !important; }
.pill.primary { background: var(--teal); border-color: var(--teal); color: #fff !important; }
.pill.primary:hover { background: var(--teal-deep); border-color: var(--teal-deep); color: #fff !important; }

/* Upload panel */
.section-title { font-size: 1.15rem; font-weight: 600; margin: 1.8rem 0 0.2rem 0; font-family: 'Outfit', sans-serif; }
.section-sub { color: var(--muted); font-size: 0.92rem; margin-bottom: 0.8rem; }

[data-testid="stFileUploader"] section {
    background: var(--paper);
    border: 1.5px dashed #9FB6C6;
    border-radius: 16px;
    padding: 1.6rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--teal); }
[data-testid="stFileUploader"] button {
    border-radius: 10px; border: 1px solid var(--teal);
    color: var(--teal-deep); background: #fff; font-weight: 500;
}

/* Image preview */
[data-testid="stImage"] img {
    border-radius: 14px; border: 1px solid var(--line);
    box-shadow: 0 6px 24px rgba(18, 38, 58, 0.08);
}
[data-testid="stImage"] figcaption { color: var(--muted); }

/* Result card */
.result {
    border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1.2rem;
    border: 1px solid var(--line); background: var(--paper);
    border-left-width: 6px;
}
.result .label { font-size: 0.85rem; color: var(--muted); font-weight: 500; }
.result .name { font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0.15rem 0 0.3rem 0; }
.result .conf { font-size: 0.95rem; color: var(--ink); font-weight: 500; }
.result .note { font-size: 0.92rem; color: var(--muted); margin-top: 0.6rem; line-height: 1.5; }

/* Probability bars */
.probs { background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem 1.5rem; margin-top: 1rem; }
.prob-row { display: grid; grid-template-columns: 9.5rem 1fr 3.6rem; align-items: center; gap: 0.8rem; padding: 0.35rem 0; }
.prob-name { font-size: 0.92rem; font-weight: 500; }
.prob-track { height: 10px; background: #E9EFF4; border-radius: 999px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; }
.prob-val { text-align: right; font-size: 0.88rem; color: var(--muted); font-variant-numeric: tabular-nums; }

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-top: 0.6rem; }
.stat-card {
    background: var(--paper); border: 1px solid var(--line);
    border-radius: 14px; padding: 0.9rem 1rem; text-align: center;
}
.stat-card .v { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--ink); }
.stat-card .k { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }

/* Resource cards */
.res-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; margin-top: 0.4rem; }
.res-card {
    display: block; background: var(--paper); border: 1px solid var(--line);
    border-radius: 14px; padding: 0.95rem 1.1rem; text-decoration: none !important;
    transition: border-color .15s, box-shadow .15s;
}
.res-card:hover { border-color: var(--teal); box-shadow: 0 4px 16px rgba(14, 143, 138, 0.12); }
.res-card .t { color: var(--ink); font-weight: 600; font-size: 0.98rem; }
.res-card .s { color: var(--muted); font-size: 0.85rem; margin-top: 0.1rem; }

.footer { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 1.6rem; }

a:focus-visible, button:focus-visible { outline: 3px solid #7CC9C5; outline-offset: 2px; }

@media (max-width: 640px) {
    .hero h1 { font-size: 2.2rem; }
    .res-grid { grid-template-columns: 1fr; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .prob-row { grid-template-columns: 7rem 1fr 3.2rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model — the full fine-tuned model is saved as one .keras file, so we just
# load it directly (no architecture rebuild needed).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ("best_model.keras", "intel_final_model.keras"):
        path = os.path.join(current_dir, fname)
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    st.error(
        "Model file not found. Add 'best_model.keras' (or 'intel_final_model.keras') "
        "to the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_pills():
    parts = []
    for title, _, url in LINKS:
        cls = "pill primary" if title == "GitHub" else "pill"
        parts.append(f'<a class="{cls}" href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>')
    return '<div class="pill-row">' + "".join(parts) + "</div>"


def render_result(class_key, confidence):
    info = CLASS_INFO[class_key]
    return (
        f'<div class="result" style="border-left-color:{info["color"]}; background:{info["tint"]};">'
        f'<div class="label">Prediction</div>'
        f'<div class="name" style="color:{info["color"]};">{info["name"]}</div>'
        f'<div class="conf">{confidence:.1f}% confidence</div>'
        f'<div class="note">{info["note"]}</div>'
        f"</div>"
    )


def render_probs(probs):
    order = np.argsort(probs)[::-1]
    rows = []
    for i in order:
        key = LABELS[i]
        info = CLASS_INFO[key]
        pct = float(probs[i]) * 100
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-name">{info["name"]}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%; background:{info["color"]};"></div></div>'
            f'<div class="prob-val">{pct:.1f}%</div>'
            f"</div>"
        )
    return '<div class="probs">' + "".join(rows) + "</div>"


def render_stats():
    stats = [
        ("90%", "Test accuracy"),
        ("91.2%", "Val accuracy"),
        ("6", "Classes"),
        ("14k", "Train images"),
    ]
    cards = "".join(
        f'<div class="stat-card"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for v, k in stats
    )
    return f'<div class="stat-grid">{cards}</div>'


def render_resources():
    cards = "".join(
        f'<a class="res-card" href="{url}" target="_blank" rel="noopener noreferrer">'
        f'<div class="t">{title}</div><div class="s">{sub}</div></a>'
        for title, sub, url in LINKS
    )
    return f'<div class="res-grid">{cards}</div>'


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<span class="hero-badge"><span class="dot"></span>EfficientNetB0 · Transfer learning · 6 classes</span>'
    "<h1>Intel image classification</h1>"
    "<p>Upload a photo of a landscape or city scene and the model will classify it as "
    "buildings, forest, glacier, mountain, sea or street, with a confidence score for each class.</p>"
    + render_pills()
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Upload a photo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">JPG, JPEG or PNG. Landscapes and city scenes work best.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(image, caption="Uploaded photo", use_container_width=True)

    with right:
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (150, 150))
        # NOTE: no /255 rescaling here. The model was trained on raw [0,255]
        # pixels (EfficientNet rescales internally), so the app must do the
        # exact same preprocessing as training.
        img_array = np.expand_dims(img_array, axis=0).astype("float32")

        with st.spinner("Analyzing photo..."):
            prediction = model.predict(img_array, verbose=0)

        probs = prediction[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100

        st.markdown(render_result(LABELS[class_idx], confidence), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Probability by class</div>', unsafe_allow_html=True)
    st.markdown(render_probs(probs), unsafe_allow_html=True)

st.markdown('<div class="section-title">Model stats</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Measured on the sealed test set after fine-tuning.</div>', unsafe_allow_html=True)
st.markdown(render_stats(), unsafe_allow_html=True)

st.markdown('<div class="section-title">Project resources</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Code, data and training details.</div>', unsafe_allow_html=True)
st.markdown(render_resources(), unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Streamlit and TensorFlow · A student deep-learning project</div>', unsafe_allow_html=True)
