from flask import Flask, request, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io
import os
import gdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# === MODELOS ===
CLASSIFIER_URL = "https://drive.google.com/uc?id=1TQ2_ozS3crjqchAPXNCBuyVs2SvZdf9R"
CLASSIFIER_PATH = "tumor_classifier.h5"

SEGMENTATION_URL = "https://drive.google.com/uc?id=1-8qtBsW7FTAGf9nqMToHFIwUmlB0PtVb"
SEGMENTATION_PATH = "segmentacion.keras"

# Descargar modelos si no existen
if not os.path.exists(CLASSIFIER_PATH):
    print("Descargando modelo de clasificación...")
    gdown.download(CLASSIFIER_URL, CLASSIFIER_PATH, quiet=False)

if not os.path.exists(SEGMENTATION_PATH):
    print("Descargando modelo de segmentación...")
    gdown.download(SEGMENTATION_URL, SEGMENTATION_PATH, quiet=False)

# === FLASK APP ===
app = Flask(__name__)
CORS(app)

# Cargar modelos
resnet_model = load_model(CLASSIFIER_PATH)          # (128x128x3)
unet_model = load_model(SEGMENTATION_PATH)          # (256x256x1)

# === UTILIDADES ===
def array_to_pil_image(arr):
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def pil_to_bytes_io(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf

# === ENDPOINT ===
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['data']

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

    if tumor_prob < 0.0885:
        img_pil = array_to_pil_image(arr_gray)
        img_buf = pil_to_bytes_io(img_pil)

        # Crear PDF con solo la imagen original
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        c.drawString(100, 750, "Resultado: No hay tumor detectado.")
        c.drawString(100, 735, f"Probabilidad: {tumor_prob:.6f}")
        c.drawImage(ImageReader(img_buf), 100, 450, width=256, height=256)
        c.save()
        pdf_buf.seek(0)

        return send_file(pdf_buf, mimetype='application/pdf', download_name='resultado.pdf')

    # Segmentación
    pred_mask = unet_model.predict(input_gray)[0]
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8).squeeze()

    # Crear overlay en color
    overlay = np.stack([arr_gray]*3, axis=-1)
    overlay[pred_mask_bin == 1] = [1, 0, 0]

    # Convertir imágenes a PIL
    mri_img = array_to_pil_image(arr_gray)
    mask_img = array_to_pil_image(np.stack([pred_mask_bin]*3, axis=-1))
    overlay_img = array_to_pil_image(overlay)

    # Buffers para insertar en PDF
    mri_buf = pil_to_bytes_io(mri_img)
    mask_buf = pil_to_bytes_io(mask_img)
    overlay_buf = pil_to_bytes_io(overlay_img)

    # Crear PDF
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    c.drawString(100, 750, "Resultado: Tumor detectado")
    c.drawString(100, 735, f"Probabilidad: {tumor_prob:.6f}")
    c.drawString(100, 700, "Imagen original:")
    c.drawImage(ImageReader(mri_buf), 100, 450, width=256, height=256)
    c.drawString(100, 430, "Máscara:")
    c.drawImage(ImageReader(mask_buf), 100, 200, width=256, height=256)
    c.drawString(380, 430, "Overlay:")
    c.drawImage(ImageReader(overlay_buf), 380, 200, width=256, height=256)
    c.save()
    pdf_buf.seek(0)

    return send_file(pdf_buf, mimetype='application/pdf', download_name='resultado.pdf')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
