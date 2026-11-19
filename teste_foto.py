import cv2

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print('Leitura ok:', ok)

if ok:
    print('Formato do quadro:', frame.shape)
    cv2.imwrite('foto_teste.jpg', frame)
    print('Foto salva em foto_teste.jpg')

cap.release()