import re
import streamlit as st
import cv2
import numpy as np
import easyocr
from paddleocr import PaddleOCR
from ultralytics import YOLO

st.set_page_config(page_title="Detección de Placas", layout="wide")
st.title("🚗 Detección, Recorte y Lectura de Placas Vehiculares")

# ── Cargar modelos ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_resource
def load_easy():
    return easyocr.Reader(['es'], gpu=False)

@st.cache_resource
def load_paddle():
    try:
        return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    except TypeError:
        try:
            return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        except Exception:
            return None

model  = load_model()
easy   = load_easy()
paddle = load_paddle()

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
    if len(t) != 6:
        return t
    letras = t[:3].translate(NUMERO_A_LETRA)
    nums   = t[3:5].translate(LETRA_A_NUMERO)
    ultimo = t[5]
    final  = ultimo if ultimo.isalpha() else ultimo.translate(LETRA_A_NUMERO)
    return letras + nums + final

def extraer_placa(texto: str) -> tuple:
    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')
    limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())

    mejor_placa        = None
    menor_correcciones = float('inf')

    for i in range(len(limpio) - 5):
        candidato = limpio[i:i+6]
        corregido = corregir_placa(candidato)
        if patron.match(corregido):
            correcciones = sum(1 for a, b in zip(candidato, corregido) if a != b)
            if correcciones < menor_correcciones:
                menor_correcciones = correcciones
                mejor_placa        = corregido

    return mejor_placa, menor_correcciones

def filtrar_por_altura(resultado: list, img_h: int, min_ratio: float = 0.30) -> list:
    return [
        (bbox, txt, prob) for (bbox, txt, prob) in resultado
        if (max(p[1] for p in bbox) - min(p[1] for p in bbox)) / img_h >= min_ratio
    ]

def preprocesar(cropped: np.ndarray) -> tuple:
    grande = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)

    _, otsu   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    _, otsu_c = cv2.threshold(clahe.apply(gray), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    _, otsu_s = cv2.threshold(cv2.filter2D(gray, -1, kernel), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variantes = [grande, otsu, otsu_c, otsu_s, cv2.bitwise_not(otsu)]
    return variantes, otsu

# ── OCR con EasyOCR ────────────────────────────────────────────────────────────
def ocr_easy(variantes: list, conf_ocr: float) -> tuple:
    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')
    for img in variantes:
        img_h     = img.shape[0]
        resultado = easy.readtext(
            img,
            allowlist=ALLOWLIST,
            contrast_ths=0.3,
            adjust_contrast=0.7,
            text_threshold=0.4,
            low_text=0.3,
            paragraph=False,
        )
        filtrado    = filtrar_por_altura(resultado, img_h)
        texto       = ''.join(t for (_, t, p) in filtrado if p >= conf_ocr)
        placa, corr = extraer_placa(texto)
        if placa and patron.match(placa):
            return placa, corr
    return None, float('inf')

# ── OCR con PaddleOCR ──────────────────────────────────────────────────────────
def ocr_paddle(variantes: list) -> tuple:
    if paddle is None:
        return None, float('inf')

    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')
    for img in variantes:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            resultado = paddle.ocr(img, cls=True)
        except Exception:
            continue
        if not resultado or not resultado[0]:
            continue
        texto = ''.join(
            re.sub(r'[^A-Z0-9]', '', line[1][0].upper())
            for line in resultado[0]
        )
        placa, corr = extraer_placa(texto)
        if placa and patron.match(placa):
            return placa, corr
    return None, float('inf')

# ── Votación entre motores ─────────────────────────────────────────────────────
def ocr_votacion(variantes: list, conf_ocr: float) -> tuple:
    placa_easy,   corr_easy   = ocr_easy(variantes, conf_ocr)
    placa_paddle, corr_paddle = ocr_paddle(variantes)

    if placa_easy and placa_paddle:
        if placa_easy == placa_paddle:
            return placa_easy, "✅ Ambos motores coinciden"
        elif corr_easy <= corr_paddle:
            return placa_easy,   f"⚖️ EasyOCR gana ({corr_easy} vs {corr_paddle} correcciones)"
        else:
            return placa_paddle, f"⚖️ PaddleOCR gana ({corr_paddle} vs {corr_easy} correcciones)"
    elif placa_easy:
        return placa_easy,   "🔵 Solo EasyOCR encontró placa"
    elif placa_paddle:
        return placa_paddle, "🟡 Solo PaddleOCR encontró placa"
    else:
        return None, "❌ Ningún motor encontró placa válida"

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
                placa_valida, fuente    = ocr_votacion(variantes, conf_ocr)

                st.image(cropped, channels="BGR", caption=f"Placa {i+1}")

                if placa_valida:
                    st.success(f"🔤 **{placa_valida}**")
                    st.caption(fuente)
                else:
                    st.warning(f"⚠️ {fuente}")

                with st.expander(f"Detalles — Placa {i+1}"):
                    st.write(f"**Confianza YOLO:** {float(box.conf[0]):.2%}")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(proc_display, caption="Otsu")
                    with col_v2:
                        st.image(variantes[2], caption="CLAHE + Otsu")

        else:
            st.warning("No se detectaron placas en la imagen.")
