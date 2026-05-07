import re
import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
    PADDLE_DISPONIBLE = True
except ImportError:
    PADDLE_DISPONIBLE = False

st.set_page_config(page_title="Deteccion de Placas", layout="wide")
st.title("Deteccion, Recorte y Lectura de Placas Vehiculares")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

@st.cache_resource
def load_easy():
    return easyocr.Reader(['es'], gpu=False)

@st.cache_resource
def load_paddle():
    if PADDLE_DISPONIBLE:
        return PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )
    return None

model         = load_model()
easy_reader   = load_easy()
paddle_reader = load_paddle()

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

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

def ocr_easy(img, img_h, conf_ocr):
    resultado = easy_reader.readtext(
        img,
        allowlist=ALLOWLIST,
        contrast_ths=0.3,
        adjust_contrast=0.7,
        text_threshold=0.4,
        low_text=0.3,
        paragraph=False,
    )
    filtrado    = filtrar_por_altura(resultado, img_h, min_ratio=0.30)
    texto_unido = ''.join(txt for (_, txt, prob) in filtrado if prob >= conf_ocr)
    placa       = extraer_placa(texto_unido)
    return [placa] if placa else []

def ocr_paddle(img, conf_ocr):
    if paddle_reader is None:
        return []
    try:
        if len(img.shape) == 2:
            img_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_p = img
        result = paddle_reader.ocr(img_p, cls=True)
        if not result or not result[0]:
            return []
        img_h  = img_p.shape[0]
        textos = []
        for linea in result[0]:
            bbox_raw, (txt, conf) = linea
            bbox   = [[int(p[0]), int(p[1])] for p in bbox_raw]
            ys     = [p[1] for p in bbox]
            char_h = max(ys) - min(ys)
            if conf >= conf_ocr and (char_h / img_h) >= 0.30:
                limpio = re.sub(r'[^A-Z0-9]', '', txt.upper())
                textos.append(limpio)
        texto_unido = ''.join(textos)
        placa = extraer_placa(texto_unido)
        return [placa] if placa else []
    except Exception:
        return []

def votar_placa(candidatos):
    candidatos = [p for p in candidatos if p and len(p) == 6]
    if not candidatos:
        return None
    if len(candidatos) == 1:
        return candidatos[0]
    resultado = ''
    for i in range(6):
        chars   = [p[i] for p in candidatos]
        ganador = max(set(chars), key=chars.count)
        resultado += ganador
    return resultado

def procesar_placa(cropped, conf_ocr):
    variantes, proc_display = preprocesar(cropped)
    todos_candidatos        = []
    log_motores             = {'easy': [], 'paddle': []}

    for img in variantes:
        img_h = img.shape[0]

        placas_easy = ocr_easy(img, img_h, conf_ocr)
        todos_candidatos.extend(placas_easy)
        log_motores['easy'].extend(placas_easy)

        placas_paddle = ocr_paddle(img, conf_ocr)
        todos_candidatos.extend(placas_paddle)
        log_motores['paddle'].extend(placas_paddle)

    placa_final = votar_placa(todos_candidatos)
    return placa_final, todos_candidatos, log_motores, proc_display, variantes

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header('Configuracion')
conf_threshold = st.sidebar.slider('Confianza deteccion', 0.0, 1.0, 0.25)
conf_ocr       = st.sidebar.slider('Confianza OCR',       0.0, 1.0, 0.15)
st.sidebar.markdown('---')
motor_info = '✅ EasyOCR + PaddleOCR activos' if PADDLE_DISPONIBLE \
             else '⚠️ Solo EasyOCR (PaddleOCR no disponible)'
st.sidebar.caption(motor_info)
st.sidebar.caption(
    'Si la placa no se lee correctamente, baja ambos sliders. '
    'Valores recomendados: deteccion 0.1-0.3, OCR 0.1-0.2'
)

# ── Subida de imagen ───────────────────────────────────────────────────────────
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
                    placa_final, candidatos, log_motores, proc_display, variantes = \
                        procesar_placa(cropped, conf_ocr)

                st.image(cropped, channels='BGR', caption='Placa ' + str(i + 1))

                if placa_final:
                    st.success('Placa detectada: ' + placa_final)
                else:
                    st.warning('No se pudo leer la placa')

                with st.expander('Detalles Placa ' + str(i + 1)):
                    st.write('Confianza YOLO: ' + str(round(float(box.conf[0]) * 100, 2)) + '%')
                    st.write('Candidatos EasyOCR: ' + str(log_motores['easy']))
                    st.write('Candidatos PaddleOCR: ' + str(log_motores['paddle']))
                    st.write('Todos los votos: ' + str(candidatos))
                    st.write('Placa votada: ' + str(placa_final))
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(proc_display, caption='Otsu')
                    with col_v2:
                        st.image(variantes[2], caption='CLAHE + Otsu')

        else:
            st.warning('No se detectaron placas en la imagen.')
