import easyocr
import cv2

# Carrega imagem
imagem = cv2.imread("exemplos/placa_teste.jpg")

# Inicializa o OCR
reader = easyocr.Reader(['pt', 'en'])

# Faz leitura
resultados = reader.readtext(imagem)

for (bbox, texto, confianca) in resultados:
    print(f"Texto detectado: {texto} (confiança: {confianca:.2f})")
