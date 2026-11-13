"""Salva um quadro de cada vídeo e testa se o MediaPipe acha o rosto."""
from pathlib import Path
import cv2
import pandas as pd
import mediapipe as mp

RAIZ = Path(__file__).resolve().parent.parent
inv = pd.read_csv(RAIZ / "data" / "interim" / "inventario.csv")
PASTA = RAIZ / "data" / "interim" / "quadros"
PASTA.mkdir(parents=True, exist_ok=True)

detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

for _, v in inv.iterrows():
    cap = cv2.VideoCapture(str(RAIZ / v["arquivo"]))
    cap.set(cv2.CAP_PROP_POS_MSEC, 60_000)   # minuto 1 do vídeo
    ok, quadro = cap.read()
    cap.release()
    if not ok:
        print(f"pessoa {v['pessoa']} rotulo {v['rotulo']}: não leu o quadro")
        continue

    h, w = quadro.shape[:2]
    escala = 640 / max(h, w)
    quadro = cv2.resize(quadro, (int(w * escala), int(h * escala)))

    res = detector.process(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
    achou = res.multi_face_landmarks is not None

    cv2.imwrite(str(PASTA / f"{v['pessoa']}_{v['rotulo']}.jpg"), quadro)
    print(f"pessoa {v['pessoa']}  rotulo {v['rotulo']:>2}  {w}x{h}  rosto: {'SIM' if achou else 'NAO'}")