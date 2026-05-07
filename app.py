import re
import streamlit as st
import cv2
import numpy as np
import easyocr
import pytesseract
from ultralytics import YOLO

st.set_page_config(page_title="Deteccion de Placas", layout="wide")
st.title("Deteccion, Recorte y Lectura de Placas Vehiculares")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es'], gpu=False)

model      = load_model()
ocr_reader = load_ocr()

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

EASY_CONFIGS = [
    dict(contrast_ths=0.3, adjust_contrast=0.7, text_threshold=0.4, low_text=0.3),
    dict(contrast_ths=0.2, adjust_contrast=0.8, text_threshold=0.3, low_text=0.2),
    dict(contrast_ths=0.4, adjust_contrast=0.6, text_threshold=0.5, low_text=0.4),
    dict(contrast_ths=0.1, adjust_contrast=0.9, text_threshold=0.3, low_text=0.2),
    dict(contrast_ths=0.5, adjust_contrast=0.5, text_threshold=0.6, low_text=0.4),
]

TESS_PSMS = [
    r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    r'--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
]

NUMERO_A_LETRA = str.maketrans({
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S',
    '6': 'G', '8': 'B', '4': 'A', '9': 'P'
})
LETRA_A_NUMERO = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
    'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4',
    'T': '7', 'E': '3', 'P': '9', 'U': '0', 'J': '1'
})

def corregir_placa(t):
    if len(t) != 6:
        return t
    letras = t[:3].translate(NUMERO_A_LETRA)
    nums   = t[3:5].translate(LETRA_A_NUMERO)
    ultimo = t[5]
    final  = ultimo if ultimo.isalpha() else ultimo.translate(LETRA_A_NUMERO)
    return letras + nums + final

def extraer_placa(texto):
    patron = re.compile(r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$')
    limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
    mejor    = None
    min_corr = float('inf')
    for i in range(len(limpio) - 5):
        candidato = limpio[i:i+6]
        corregido = corregir_placa(candidato)
        if patron.match(corregido):
            correcciones = sum(1 for a, b in zip(candidato, corregido) if a != b)
            if correcciones < min_corr:
                min_corr = correcciones
                mejor    = corregido
    return mejor

def filtrar_por_altura(resultado, img_h, min_ratio=0.30):
    filtrados = []
    for (bbox, txt, prob) in resultado:
        ys     = [p[1] for p in bbox]
        char_h = max(ys) - min(ys)
        if (char_h / img_h) >= min_ratio:
            filtrados.append((bbox, txt, prob))
    return filtrados

def preprocesar(cropped):
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

def run_easyocr(variantes, conf_ocr):
    candidatos = []
    for img in variantes:
        img_h = img.shape[0]
        for cfg in EASY_CONFIGS:
            try:
                res      = ocr_reader.readtext(img, allowlist=ALLOWLIST,
                                               paragraph=False, **cfg)
                filtrado = filtrar_por_altura(res, img_h, min_ratio=0.30)
                texto    = ''.join(t for (_, t, p) in filtrado if p >= conf_ocr)
                placa    = extraer_placa(texto)
                if placa:
                    candidatos.append(placa)
            except Exception:
                continue
    return candidatos

def run_tesseract(variantes):
    candidatos = []
    for img in variantes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
        for cfg in TESS_PSMS:
            try:
                texto = pytesseract.image_to_string(gray, config=cfg).strip()
                placa = extraer_placa(texto)
                if placa:
                    candidatos.append(placa)
            except Exception:
                continue
    return candidatos

def votar_placa(candidatos):
    candidatos = [p for p in candidatos if p and len(p) == 6]
    if not candidatos:
        return None
    resultado = ''
    for i in range(6):
        chars   = [p[i] for p in candidatos]
        ganador = max(set(chars), key=chars.count)
        resultado += ganador
    return resultado

def procesar_placa(cropped, conf_ocr):
    variantes, proc_display = preprocesar(cropped)
    cands_easy = run_easyocr(variantes, conf_ocr)
    cands_tess = run_tesseract(variantes)

    # Tesseract recibe 5x mas peso porque lee mejor caracteres dificiles
    todos = cands_easy + (cands_tess * 10)

    placa_final = votar_placa(todos)
    return placa_final, cands_easy, cands_tess, proc_display, variantes

st.sidebar.header('Configuracion')
conf_threshold = st.sidebar.slider('Confianza deteccion', 0.0, 1.0, 0.25)
conf_ocr       = st.sidebar.slider('Confianza OCR',       0.0, 1.0, 0.15)
st.sidebar.markdown('---')
st.sidebar.caption('Motores: EasyOCR + Tesseract (peso 5x)')
st.sidebar.caption('Recomendado: deteccion 0.1-0.3, OCR 0.1-0.2')

uploaded_file = st.file_uploader('Sube una imagen', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _    = image.shape

    results         = model.predict(image, conf=conf_threshold)
    annotated_image = results[0].plot()
    boxes           = results[0].boxes

    col_det, col_placas = st.columns(2)

    with col_det:
        st.subheader('Detecciones')
        st.image(annotated_image, channels='BGR')

    with col_placas:
        st.subheader('Placas recortadas')

        if boxes is not None and len(boxes) > 0:

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cropped = image[y1:y2, x1:x2]

                with st.spinner('Leyendo placa ' + str(i + 1) + '...'):
                    placa_final, cands_easy, cands_tess, proc_display, variantes = \
                        procesar_placa(cropped, conf_ocr)

                st.image(cropped, channels='BGR', caption='Placa ' + str(i + 1))

                if placa_final:
                    st.success('Placa detectada: ' + placa_final)
                else:
                    st.warning('No se pudo leer la placa')

                with st.expander('Detalles Placa ' + str(i + 1)):
                    st.write('Confianza YOLO: ' + str(round(float(box.conf[0]) * 100, 2)) + '%')
                    st.write('EasyOCR: ' + str(cands_easy))
                    st.write('Tesseract: ' + str(cands_tess))
                    st.write('Placa final: ' + str(placa_final))
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(proc_display, caption='Otsu')
                    with col_v2:
                        st.image(variantes[2], caption='CLAHE + Otsu')

        else:
            st.warning('No se detectaron placas en la imagen.')
