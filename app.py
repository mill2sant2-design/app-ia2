import re
import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

st.set_page_config(page_title="Detección de Placas", layout="wide")
st.title("🚗 Detección, Recorte y Lectura de Placas Vehiculares")

# ── Cargar modelos ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es'], gpu=False)

model  = load_model()
reader = load_ocr()

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ── Corrección posicional ──────────────────────────────────────────────────────
NUMERO_A_LETRA = str.maketrans({
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S',
    '6': 'G', '8': 'B', '4': 'A', '9': 'P'
})
LETRA_A_NUMERO = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
    'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4',
    'T': '7', 'E': '3', 'P': '9', 'U': '0', 'J': '1'
})

def corregir_placa(t: str) -> str:
    """Corrección posicional sobre exactamente 6 caracteres."""
    if len(t) != 6:
        return t
    letras = t[:3].translate(NUMERO_A_LETRA)
    nums   = t[3:5].translate(LETRA_A_NUMERO)
    ultimo = t[5]
    final  = ultimo if ultimo.isalpha() else ultimo.translate(LETRA_A_NUMERO)
    return letras + nums + final

def extraer_placa(texto_completo: str) -> str | None:
    """
    Busca un patrón de 6 caracteres alfanuméricos dentro del texto crudo
    (maneja prefijos/sufijos fantasma como 'I', '1', espacios, guiones).
    """
    patron_valido = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')

    # Limpiar: solo letras y números
    limpio = re.sub(r'[^A-Z0-9]', '', texto_completo.upper())

    # Buscar todas las ventanas de 6 caracteres consecutivos
    for i in range(len(limpio) - 5):
        candidato = limpio[i:i+6]
        corregido = corregir_placa(candidato)
        if patron_valido.match(corregido):
            return corregido

    return None

def preprocesar(cropped: np.ndarray) -> tuple:
    """
    Recorta el 65% superior (elimina ciudad) y genera variantes.
    Devuelve (lista_variantes, imagen_display_para_UI).
    """
    # ── Eliminar franja inferior (ciudad) ──────────────────────────────────────
    h = cropped.shape[0]
    solo_numero = cropped[:int(h * 0.65), :]   # <── clave

    # Escalar x3
    grande = cv2.resize(solo_numero, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)

    # Variantes
    _, otsu    = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_c     = clahe.apply(gray)
    _, otsu_c  = cv2.threshold(gray_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel     = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp      = cv2.filter2D(gray, -1, kernel)
    _, otsu_s  = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variantes = [grande, otsu, otsu_c, otsu_s, cv2.bitwise_not(otsu)]
    return variantes, otsu

def ocr_multi_variante(variantes: list, conf_ocr: float) -> tuple:
    """Prueba cada variante y devuelve la primera con placa válida."""
    for img in variantes:
        resultado = reader.readtext(
            img,
            allowlist=ALLOWLIST,
            contrast_ths=0.3,
            adjust_contrast=0.7,
            text_threshold=0.4,
            low_text=0.3,
            paragraph=False,
        )
        # Unir todos los fragmentos con confianza suficiente
        texto_unido = ''.join(
            txt for (_, txt, prob) in resultado if prob >= conf_ocr
        )
        placa = extraer_placa(texto_unido)
        if placa:
            fragmentos = [txt.upper() for (_, txt, prob) in resultado if prob >= conf_ocr]
            return placa, fragmentos, resultado

    # Fallback: primera variante sin filtro de confianza
    resultado  = reader.readtext(variantes[0], allowlist=ALLOWLIST, paragraph=False)
    texto_unido = ''.join(txt for (_, txt, _prob) in resultado)
    fragmentos  = [txt.upper() for (_, txt, _prob) in resultado]
    return extraer_placa(texto_unido), fragmentos, resultado

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
conf_threshold = st.sidebar.slider("Confianza detección", 0.0, 1.0, 0.25)
conf_ocr       = st.sidebar.slider("Confianza OCR",       0.0, 1.0, 0.15)
st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Si la placa no se lee correctamente, "
    "baja ambos sliders. Valores recomendados: "
    "detección 0.1–0.3 · OCR 0.1–0.2"
)

# ── Subida de imagen ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _    = image.shape

    results         = model.predict(image, conf=conf_threshold)
    annotated_image = results[0].plot()
    boxes           = results[0].boxes

    col_det, col_placas = st.columns(2)

    with col_det:
        st.subheader("📌 Detecciones")
        st.image(annotated_image, channels="BGR")

    with col_placas:
        st.subheader("🔍 Placas recortadas")

        if boxes is not None and len(boxes) > 0:

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cropped = image[y1:y2, x1:x2]

                variantes, proc_display = preprocesar(cropped)
                placa_valida, fragmentos, resultado_crudo = ocr_multi_variante(
                    variantes, conf_ocr
                )

                st.image(cropped, channels="BGR", caption=f"Placa {i+1}")

                if placa_valida:
                    st.success(f"🔤 **{placa_valida}**")
                elif fragmentos:
                    st.warning(f"⚠️ No coincide con formato: {' '.join(fragmentos)}")
                else:
                    st.warning("⚠️ Texto no legible")

                with st.expander(f"Detalles — Placa {i+1}"):
                    st.write(f"**Confianza YOLO:** {float(box.conf[0]):.2%}")
                    st.write(f"**Fragmentos OCR:** {fragmentos}")
                    st.write(f"**OCR crudo:** {resultado_crudo}")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(proc_display, caption="Otsu (sin ciudad)")
                    with col_v2:
                        st.image(variantes[2], caption="CLAHE + Otsu (sin ciudad)")

        else:
            st.warning("No se detectaron placas en la imagen.")
