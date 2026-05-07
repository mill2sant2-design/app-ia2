import re
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

model  = load_model()
reader = load_ocr()

# ── Mapas de corrección por posición ──────────────────────────────────────────
# Posiciones 0-2: deben ser LETRAS → corregir dígitos que parecen letras
NUMERO_A_LETRA = str.maketrans({
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S',
    '6': 'G', '8': 'B', '4': 'A'
})

# Posiciones 3-5: deben ser NÚMEROS → corregir letras que parecen dígitos
LETRA_A_NUMERO = str.maketrans({
    'O': '0', 'Q': '0', 'I': '1', 'L': '1',
    'Z': '2', 'S': '5', 'G': '6', 'B': '8',
    'A': '4', 'T': '7', 'E': '3'
})

def corregir_placa(texto: str) -> str:
    """
    Aplica corrección posicional al texto crudo:
      - Pos 0-2 → forzar letras
      - Pos 3-5 → forzar números  (carro: AAA000)
      - Pos 3-4 números, pos 5 letra (moto: AAA00A)
    """
    t = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(t) != 6:
        return t  # no se puede corregir si no tiene 6 chars

    parte_letras  = t[:3].translate(NUMERO_A_LETRA)  # pos 0-2 → letras
    parte_nums    = t[3:5].translate(LETRA_A_NUMERO)  # pos 3-4 → números

    # Pos 5: número si es carro, letra si es moto
    ultimo = t[5]
    if ultimo.isalpha():
        # moto: dejar como letra
        parte_final = ultimo
    else:
        # carro: forzar número
        parte_final = ultimo.translate(LETRA_A_NUMERO)

    return parte_letras + parte_nums + parte_final


def extraer_placa(fragmentos: list) -> str | None:
    """
    Une fragmentos, limpia y valida formato colombiano.
    Devuelve la placa corregida o None.
    """
    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')

    # Intentar cada fragmento solo y luego combinaciones de 2
    candidatos = list(fragmentos)
    if len(fragmentos) >= 2:
        candidatos.append(''.join(fragmentos[:2]))
        candidatos.append(''.join(fragmentos))

    for cand in candidatos:
        limpio    = re.sub(r'[^A-Z0-9]', '', cand.upper())
        corregido = corregir_placa(limpio)
        if patron.match(corregido):
            return corregido

    return None


def preprocesar(cropped: np.ndarray) -> list:
    """
    Genera múltiples versiones preprocesadas del recorte
    para aumentar las chances de lectura correcta.
    """
    variantes = []

    # Escalar
    grande = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)

    # 1. Otsu directo
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu)

    # 2. CLAHE + Otsu (mejora contraste local)
    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_clahe = clahe.apply(gray)
    _, otsu_c  = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu_c)

    # 3. Sharpening + Otsu
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, otsu_s = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu_s)

    # 4. Invertido de Otsu (para placas oscuras con texto claro)
    variantes.append(cv2.bitwise_not(otsu))

    return variantes, otsu  # también devuelve una para mostrar en UI


def ocr_multi_variante(variantes: list, conf_ocr: float) -> tuple:
    """
    Corre EasyOCR sobre cada variante y devuelve
    la mejor placa encontrada + los fragmentos de esa variante.
    """
    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')

    for img in variantes:
        resultado  = reader.readtext(img)
        fragmentos = [txt.upper() for (_, txt, prob) in resultado if prob >= conf_ocr]
        placa      = extraer_placa(fragmentos)
        if placa and patron.match(placa):
            return placa, fragmentos, resultado

    # Si ninguna dio placa válida, devolver lo que sea de la primera variante
    resultado  = reader.readtext(variantes[0])
    fragmentos = [txt.upper() for (_, txt, prob) in resultado if prob >= conf_ocr]
    return None, fragmentos, resultado


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
conf_threshold = st.sidebar.slider("Confianza detección", 0.0, 1.0, 0.5)
conf_ocr       = st.sidebar.slider("Confianza OCR",       0.0, 1.0, 0.4)

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
                    st.warning(f"⚠️ No coincide con formato colombiano: {' '.join(fragmentos)}")
                else:
                    st.warning("⚠️ Texto no legible")

                with st.expander(f"Detalles — Placa {i+1}"):
                    st.write(f"**Confianza YOLO:** {float(box.conf[0]):.2%}")
                    st.write(f"**Fragmentos OCR:** {fragmentos}")
                    st.write(f"**OCR crudo:** {resultado_crudo}")
                    st.image(proc_display, caption="Preprocesamiento aplicado")

        else:
            st.warning("No se detectaron placas en la imagen.")
