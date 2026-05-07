import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Detección de Placas", layout="wide")

st.title("🚗 Detección y Recorte de Placas Vehiculares")

# -------------------------------
# Cargar modelo (cacheado)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")  # <-- ajusta la ruta

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Configuración")
conf_threshold = st.sidebar.slider("Confianza", 0.0, 1.0, 0.5)

# -------------------------------
# Subida de imagen
# -------------------------------
uploaded_file = st.file_uploader(
    "Sube una imagen", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predicción
    results = model.predict(image, conf=conf_threshold)

    # Imagen con detecciones
    annotated_image = results[0].plot()

    # Layout en columnas (simula "dos canvas")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Detecciones")
        st.image(annotated_image, channels="BGR")

    with col2:
        st.subheader("🔍 Placas recortadas")

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            h, w, _ = image.shape

            for i, box in enumerate(boxes):
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Asegurar límites válidos
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Recorte
                cropped = image[y1:y2, x1:x2]

                # Mostrar cada placa
                st.image(cropped, channels="BGR", caption=f"Placa {i+1}")

        else:
            st.warning("No se detectaron placas en la imagen")