import cv2
import mediapipe as mp

face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
PONTOS = [362, 385, 387, 263, 373, 380,   # olho esquerdo
          33, 160, 158, 133, 153, 144,    # olho direito
          13, 14, 78, 308]                # boca

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam não abriu. Feche outros apps que usem a câmera.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        for i in PONTOS:
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 2, (0, 255, 0), -1)
    cv2.imshow("teste (ESC para sair)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()