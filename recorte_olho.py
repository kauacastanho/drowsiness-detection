import time
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# pontos do contorno de um olho na malha facial
PONTOS_OLHO = [33, 160, 159, 158, 133, 153, 145, 144]

cap = cv2.VideoCapture(0)
ultimo_ts = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Falha ao ler o quadro')
            break

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ultimo_ts = max(ultimo_ts + 1, int(time.time() * 1000))
        resultado = landmarker.detect_for_video(mp_image, ultimo_ts)

        if resultado.face_landmarks:
            pontos = resultado.face_landmarks[0]

            xs = [pontos[i].x * w for i in PONTOS_OLHO]
            ys = [pontos[i].y * h for i in PONTOS_OLHO]

            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            lado = (max(xs) - min(xs)) * 1.6

            x1 = max(0, int(cx - lado / 2))
            x2 = min(w, int(cx + lado / 2))
            y1 = max(0, int(cy - lado / 2))
            y2 = min(h, int(cy + lado / 2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            olho = cinza[y1:y2, x1:x2]

            if olho.size > 0:
                olho = cv2.resize(olho, (96, 96))
                cv2.imshow('Recorte do olho (96x96)', cv2.resize(olho, (288, 288)))

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()