import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import io
import base64
from streamlit_image_comparison import image_comparison

# -------------------- CONFIG --------------------
st.set_page_config(page_title="ArtMorph 🎨", layout="centered")

# -------------------- BACKGROUND --------------------
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background.png")  # Background image must exist

# -------------------- HEADER --------------------
st.image("header.png", use_container_width=True)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .stSelectbox > div, .stRadio > div, .stFileUploader > div {
        border: 2px solid black !important;
        border-radius: 8px;
        padding: 4px;
    }
    .stButton > button, .stDownloadButton > button {
        border: 2px solid black !important;
        background-color: white;
        color: black;
        font-weight: bold;
        border-radius: 6px;
        padding: 6px 14px;
    }
    img {
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- FOLDERS --------------------
CONTENT_DIR = "images/content"
STYLE_DIR = "images/style"

content_images = sorted([f for f in os.listdir(CONTENT_DIR) if f.endswith((".jpg", ".jpeg", ".png"))])
style_images = sorted([f for f in os.listdir(STYLE_DIR) if f.endswith((".jpg", ".jpeg", ".png"))])

# -------------------- LOADERS --------------------
def load_image_from_path(path, max_dim=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    img = np.array(img) / 255.0
    return tf.constant(img[np.newaxis, ...], dtype=tf.float32)

def load_image_from_uploaded_file(file, max_dim=512):
    img = Image.open(file).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    img = np.array(img) / 255.0
    return tf.constant(img[np.newaxis, ...], dtype=tf.float32)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor[0])

def display_image_with_border(img_path_or_bytes, caption="Image"):
    if isinstance(img_path_or_bytes, str):  # path
        with open(img_path_or_bytes, "rb") as f:
            img_bytes = f.read()
    else:  # bytes
        img_bytes = img_path_or_bytes
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(f"""
        <div style="border:2px solid black; border-radius:8px; padding:5px; margin-bottom:10px;">
            <img src="data:image/png;base64,{encoded}" style="width:100%;" alt="{caption}">
        </div>
    """, unsafe_allow_html=True)

# -------------------- UI INPUTS --------------------
st.markdown("### 🎯 Select or Upload Images")

col1, col2 = st.columns(2)

# --- Content Image ---
with col1:
    st.markdown("**🖼️ Content Image**")
    content_option = st.radio("Choose input method:", ["Upload", "Select from folder"], key="content_radio")

    content_image = None
    if content_option == "Upload":
        content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content_upload")
        if content_file:
            content_image = load_image_from_uploaded_file(content_file)
            img = Image.open(content_file)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            display_image_with_border(buf.getvalue(), caption="Uploaded Content")
    else:
        content_choice = st.selectbox("Select Content Image", content_images)
        content_path = os.path.join(CONTENT_DIR, content_choice)
        content_image = load_image_from_path(content_path)
        display_image_with_border(content_path, caption="Selected Content")

# --- Style Image ---
with col2:
    st.markdown("**🎨 Style Image**")
    style_option = st.radio("Choose input method:", ["Upload", "Select from folder"], key="style_radio")

    style_image = None
    if style_option == "Upload":
        style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style_upload")
        if style_file:
            style_image = load_image_from_uploaded_file(style_file)
            img = Image.open(style_file)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            display_image_with_border(buf.getvalue(), caption="Uploaded Style")
    else:
        style_choice = st.selectbox("Select Style Image", style_images)
        style_path = os.path.join(STYLE_DIR, style_choice)
        style_image = load_image_from_path(style_path)
        display_image_with_border(style_path, caption="Selected Style")

# -------------------- ACTION BUTTONS --------------------
colA, colB = st.columns([1, 1])
create_clicked = colA.button("🎨 Create Stylized Art")
reset_clicked = colB.button("🔄 Reset")

if reset_clicked:
    st.rerun()

# -------------------- STYLE TRANSFER --------------------
if create_clicked and content_image is not None and style_image is not None:
    st.markdown("---")
    with st.spinner("🧠 Applying Style Transfer..."):
        model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
        stylized_image = model(content_image, style_image)[0]
        result_pil = tensor_to_image(stylized_image)

    st.success("✅ Style Transfer Complete!")

    st.markdown("<div style='text-align:center'><h4>🖼️ Before vs After</h4></div>", unsafe_allow_html=True)
    image_comparison(
        img1=tensor_to_image(content_image),
        img2=result_pil,
        label1="Original",
        label2="Stylized",
    )

    # Download
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Stylized Image",
        data=byte_im,
        file_name="artmorph_result.png",
        mime="image/png"
    )

# -------------------- FOOTER --------------------
st.markdown("""
    <hr style="border: 1px solid #000000;">
    <div style="text-align: center; color: #555555; font-size: 14px;">
        Made with ❤️ by <strong>Himani Joshi</strong> using Streamlit | © 2025 <strong>ArtMorph</strong>
    </div>
""", unsafe_allow_html=True)
