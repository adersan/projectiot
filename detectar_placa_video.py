import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Carrega o modelo YOLO pré-treinado
model = YOLO('yolov8n.pt')

# Inicializa o leitor OCR
reader = easyocr.Reader(['en'])

# Inicia a webcam (0 = webcam padrão)
cap = cv2.VideoCapture(0)

texto_detectado = ""
padding = 10  # Ajuste se necessário

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        cls = int(result.cls[0])

        # Aplica padding para garantir que placas menores (como de moto) sejam lidas corretamente
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(frame.shape[1], x2 + padding)
        y2_p = min(frame.shape[0], y2 + padding)

        cropped = frame[y1_p:y2_p, x1_p:x2_p]

        # Converte para RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # OCR
        results_ocr = reader.readtext(cropped_rgb)

        # Junta todos os textos detectados com probabilidade alta
        placa_ocr = " ".join([text for (_, text, prob) in results_ocr if prob > 0.5])

        if placa_ocr:
            texto_detectado = placa_ocr.upper()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, texto_detectado, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibe o texto detectado no canto superior da tela
    if texto_detectado:
        cv2.putText(frame, f"Placa detectada: {texto_detectado}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostra o resultado
    cv2.imshow("Leitura de Placas", frame)

    # Sai ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
