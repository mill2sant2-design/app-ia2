import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

st.set_page_config(page_title="Detección de Placas", layout="wide")
st.title("🚗 Detección, Recorte y Lectura de Placas Vehiculares")

# ── Cargar modelos (cacheados) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es'], gpu=False)

model    = load_model()
reader   = load_ocr()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
conf_threshold  = st.sidebar.slider("Confianza detección", 0.0, 1.0, 0.5)
conf_ocr        = st.sidebar.slider("Confianza OCR",       0.0, 1.0, 0.4)

# ── Subida de imagen ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _    = image.shape

    # Predicción YOLO
    results          = model.predict(image, conf=conf_threshold)
    annotated_image  = results[0].plot()
    boxes            = results[0].boxes

    # ── Layout principal ───────────────────────────────────────────────────────
    col_det, col_placas = st.columns(2)

    with col_det:
        st.subheader("📌 Detecciones")
        st.image(annotated_image, channels="BGR")

    with col_placas:
        st.subheader("🔍 Placas recortadas")

        if boxes is not None and len(boxes) > 0:

            for i, box in enumerate(boxes):

                # ── Recorte ────────────────────────────────────────────────────
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cropped = image[y1:y2, x1:x2]

                # ── Preprocesamiento para OCR ──────────────────────────────────
                gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray    = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                gray    = cv2.GaussianBlur(gray, (3, 3), 0)
                _, proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # ── OCR ────────────────────────────────────────────────────────
                resultado   = reader.readtext(proc)
                fragmentos  = [
                    txt.upper()
                    for (_, txt, prob) in resultado
                    if prob >= conf_ocr
                ]
                texto_placa = " ".join(fragmentos) if fragmentos else "No legible"

                # ── Mostrar resultado ──────────────────────────────────────────
                st.image(cropped, channels="BGR", caption=f"Placa {i+1}")
                confianza_yolo = float(box.conf[0])

                if fragmentos:
                    st.success(f"🔤 **{texto_placa}**")
                else:
                    st.warning("⚠️ Texto no legible")

                with st.expander(f"Detalles — Placa {i+1}"):
                    st.write(f"**Confianza YOLO:** {confianza_yolo:.2%}")
                    st.write(f"**Fragmentos OCR detectados:** {resultado}")
                    st.image(proc, caption="Imagen procesada para OCR")

        else:
            st.warning("No se detectaron placas en la imagen.")
