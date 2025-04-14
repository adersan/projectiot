import cv2

# Abre a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Captura da Placa", frame)

    # Pressione 's' para salvar uma imagem
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("exemplos/placa_teste.jpg", frame)
        print("Imagem salva como placa_teste.jpg")
        break

cap.release()
cv2.destroyAllWindows()
