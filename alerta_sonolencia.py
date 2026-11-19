import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model('modelo_sonolencia.keras')

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

PONTOS_OLHO = [33, 160, 159, 158, 133, 153, 145, 144]
LIMIAR = 0.5
LIMITE_SEGUNDOS = 2.0          # NOVO: tempo de olho fechado para disparar o alerta

cap = cv2.VideoCapture(0)
ultimo_ts = 0
inicio_fechado = None          # NOVO: momento em que o olho fechou (None = não está fechado)

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

        texto = 'SEM ROSTO'
        cor = (0, 255, 255)
        olho_fechado = False   # NOVO: vale True só quando o modelo disser "fechado"

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

            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            olho = cinza[y1:y2, x1:x2]

            if olho.size > 0:
                olho = cv2.resize(olho, (96, 96))
                entrada = olho.astype('float32') / 255.0
                entrada = entrada.reshape(1, 96, 96, 1)

                prob_aberto = float(model(entrada, training=False).numpy()[0][0])

                if prob_aberto >= LIMIAR:
                    texto = f'ABERTO ({prob_aberto:.2f})'
                    cor = (0, 255, 0)
                else:
                    texto = f'FECHADO ({prob_aberto:.2f})'
                    cor = (0, 0, 255)
                    olho_fechado = True      # NOVO

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.imshow('Recorte do olho (96x96)', cv2.resize(olho, (288, 288)))

        # NOVO: cronômetro
        agora = time.time()
        if olho_fechado:
            if inicio_fechado is None:
                inicio_fechado = agora
            tempo_fechado = agora - inicio_fechado
        else:
            inicio_fechado = None
            tempo_fechado = 0.0

        cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

        # NOVO: contador e alerta
        if tempo_fechado >= LIMITE_SEGUNDOS:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 12)
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, 'ALERTA: SONOLENCIA!', (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        elif tempo_fechado > 0:
            cv2.putText(frame, f'Fechado ha {tempo_fechado:.1f}s', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()