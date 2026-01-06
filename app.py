import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from cnn.network import ConvNeuralNet

st.set_page_config(page_title="Galaxy Detector", layout="centered")

# Config
MODEL_PATH = "models/galaxy_cnn.pth"
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@st.cache_resource
def load_model(path: str):
    model = ConvNeuralNet(NUM_CLASSES)
    model.to(device)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except Exception:
            # try loading only state_dict key if saved as checkpoint
            ckpt = torch.load(path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_model(MODEL_PATH)

st.title("Galaxy Detector")
st.write("Upload an image and the model will predict whether it contains a galaxy.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width='stretch')

    if st.button("Classify"):
        try:
            inp = preprocess(image).unsqueeze(0).to(device)  # (1, C, H, W)
            with torch.no_grad():
                out = model(inp)
                probs = torch.softmax(out, dim=1).cpu().squeeze()
                prob_non_galaxy = float(probs[0])
                prob_galaxy = float(probs[1])
                if prob_galaxy >= prob_non_galaxy:
                    st.success(f"Galaxy ({prob_galaxy*100:.1f}% confidence)")
                else:
                    st.info(f"Non-Galaxy ({prob_non_galaxy*100:.1f}% confidence)")
        except Exception as e:
            st.error(f"Inference failed: {e}")
else:
    st.info("Upload an image to get started.")