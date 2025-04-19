from django.shortcuts import render
from django.http import JsonResponse
import cv2
import easyocr
import os
from django.conf import settings
from ultralytics import YOLO

# Caminho absoluto para o modelo YOLOv8
modelo_path = os.path.join(settings.BASE_DIR, 'models', 'license_plate_detector.pt')
modelo = YOLO(modelo_path)  # Carrega o modelo

def index(request):
    return render(request, 'index.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def capturar_placa(request):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        caminho = os.path.join(settings.MEDIA_ROOT, 'placa_teste.jpg')
        cv2.imwrite(caminho, frame)
        return JsonResponse({'mensagem': 'Imagem capturada com sucesso!'})
    return JsonResponse({'mensagem': 'Erro ao capturar imagem!'})

def reconhecer_placa(request):
    caminho = os.path.join(settings.MEDIA_ROOT, "placa_teste.jpg")
    if not os.path.exists(caminho):
        return JsonResponse({'mensagem': 'Imagem não encontrada!'})

    imagem = cv2.imread(caminho)

    # Detecta placas com YOLOv8
    resultados = modelo.predict(source=imagem, conf=0.4, verbose=False)[0]  # conf=0.4 para ajustar sensibilidade

    if len(resultados.boxes) == 0:
        return JsonResponse({'mensagem': 'Nenhuma placa detectada.'})

    reader = easyocr.Reader(['pt', 'en'])
    placas_filtradas = []

    ignorar = {"brasil", "br", "ba", "jaguaquara", "salvador"}

    for box in resultados.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        recorte = imagem[y1:y2, x1:x2]  # recorta a região da placa
        ocr_result = reader.readtext(recorte)

        for (_, texto, _) in ocr_result:
            if texto.lower() not in ignorar:
                placas_filtradas.append(texto)

    return JsonResponse({'placas': placas_filtradas})
