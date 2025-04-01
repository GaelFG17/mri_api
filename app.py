from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1TQ2_ozS3crjqchAPXNCBuyVs2SvZdf9R"
MODEL_PATH = "tumor_classifier.h5"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

app = Flask(__name__)
CORS(app)

# Cargar los modelos
resnet_model = load_model("tumor_classifier.h5")    # espera RGB (128x128x3)
unet_model = load_model("segmentacion.keras")         # espera grises (256x256x1)

def image_to_base64(img_arr):
    if img_arr.ndim == 3 and img_arr.shape[-1] == 1:
        img_arr = img_arr.squeeze(axis=-1)
    img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Clasificador → RGB (128x128x3)
    img_rgb = Image.open(file).resize((128, 128)).convert('RGB')
    arr_rgb = np.array(img_rgb) / 255.0
    input_rgb = np.expand_dims(arr_rgb, axis=0)

    tumor_prob = resnet_model.predict(input_rgb)[0][0]

    # Segmentador → Grayscale (256x256x1)
    file.seek(0)
    img_gray = Image.open(file).resize((256, 256)).convert('L')
    arr_gray = np.array(img_gray) / 255.0
    input_gray = np.expand_dims(arr_gray, axis=(0, -1))

    if tumor_prob < 0.0002:
        mri_base64 = image_to_base64(arr_gray)
        return jsonify({
            "result": "No hay tumor",
            "probability": float(tumor_prob),
            "mri": mri_base64
        })

    # Segmentación
    pred_mask = unet_model.predict(input_gray)[0]
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8).squeeze()

    # Crear overlay en color
    overlay = np.stack([arr_gray]*3, axis=-1)
    overlay[pred_mask_bin == 1] = [1, 0, 0]

    # Convertir imágenes a base64
    mri_base64 = image_to_base64(arr_gray)
    mask_base64 = image_to_base64(np.stack([pred_mask_bin]*3, axis=-1))
    overlay_base64 = image_to_base64(overlay)

    return jsonify({
        "result": "Tumor detectado",
        "probability": float(tumor_prob),
        "mri": mri_base64,
        "mask": mask_base64,
        "overlay": overlay_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
