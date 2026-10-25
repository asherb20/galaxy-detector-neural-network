import os
import streamlit as st
from PIL import Image
import torch
from cnn.network import ConvNeuralNet
from cnn.transforms import eval_transform

st.set_page_config(page_title="Galaxy Detector", layout="centered")

# Config
MODEL_PATH = "models/galaxy_morph_cnn.pth"
GALAXY_THRESHOLD = 0.5
UNCERTAIN_THRESHOLD = 0.4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (shared with training)
preprocess = eval_transform()

@st.cache_resource
def load_model(path: str):
    checkpoint = torch.load(path, map_location=device)
    model = ConvNeuralNet(len(checkpoint["classes"]), in_channels=checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])  # strict: fail loudly on a mismatch
    model.to(device)
    model.eval()
    return model, checkpoint["classes"]

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Train it with `python -m cnn.handler`.")
    st.stop()
try:
    model, classes = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Galaxy Detector")
st.write("Upload an image and the model will predict whether it contains a galaxy, and if so, its morphology.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width='stretch')

    if st.button("Classify"):
        try:
            inp = preprocess(image).unsqueeze(0).to(device)  # (1, C, H, W)
            with torch.no_grad():
                detect_logits, morph_logits = model(inp)
                prob_galaxy = float(torch.softmax(detect_logits, dim=1)[0, 1])
                morph_probs = torch.softmax(morph_logits, dim=1)[0].cpu().tolist()

            if prob_galaxy >= GALAXY_THRESHOLD:
                st.success(f"Galaxy ({prob_galaxy*100:.1f}% confidence)")
                ranked = sorted(zip(classes, morph_probs), key=lambda pair: pair[1], reverse=True)
                top_class, top_prob = ranked[0]
                label = top_class.replace("_", " ")
                if top_prob < UNCERTAIN_THRESHOLD:
                    st.warning(f"Morphology uncertain, best guess: {label} ({top_prob*100:.1f}%)")
                else:
                    st.info(f"Morphology: {label} ({top_prob*100:.1f}%)")
                for name, prob in ranked:
                    st.progress(prob, text=f"{name.replace('_', ' ')}: {prob*100:.1f}%")
            else:
                st.info(f"Not a galaxy ({(1 - prob_galaxy)*100:.1f}% confidence)")
        except Exception as e:
            st.error(f"Inference failed: {e}")
else:
    st.info("Upload an image to get started.")
