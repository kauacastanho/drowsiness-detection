import time
import cv2
import numpy as np
import mediapipe as mp

face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
LEFT  = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]

def ear(p):
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

t_prev, fps = time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    texto = "sem rosto"
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        pts = lambda idx: np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
        e = (ear(pts(LEFT)) + ear(pts(RIGHT))) / 2
        texto = f"EAR: {e:.3f}"

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (now - t_prev))
    t_prev = now

    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("EAR + FPS (ESC para sair)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()