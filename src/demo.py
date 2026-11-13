"""Demo em tempo real: webcam -> medidas do rosto -> modelo treinado."""
import sys
import time
from collections import deque
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

try:
    import winsound            # só existe no Windows
except ImportError:
    winsound = None

RAIZ = Path(__file__).resolve().parent.parent
pacote = joblib.load(RAIZ / "models" / "modelo.joblib")
modelo, FEATURAS = pacote["modelo"], pacote["features"]

JANELA = 10          # segundos analisados (igual ao treino)
TAXA = 15            # medidas por segundo (igual ao treino)
LIMIAR_EAR = 0.2     # abaixo disso o olho conta como fechado
LIMIAR_PROB = 0.5    # probabilidade a partir da qual o modelo avisa

OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33, 160, 158, 133, 153, 144]


def ponto(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def dist(lm, a, b, w, h):
    return np.linalg.norm(ponto(lm, a, w, h) - ponto(lm, b, w, h))


def ear(lm, ids, w, h):
    return (dist(lm, ids[1], ids[5], w, h) + dist(lm, ids[2], ids[4], w, h)) / \
           (2 * dist(lm, ids[0], ids[3], w, h))


def mar(lm, w, h):
    v = (dist(lm, 13, 14, w, h) + dist(lm, 82, 87, w, h) + dist(lm, 312, 317, w, h)) / 3
    return v / dist(lm, 78, 308, w, h)


def maior_sequencia(fechado, dt):
    melhor = atual = 0
    for f in fechado:
        atual = atual + 1 if f else 0
        melhor = max(melhor, atual)
    return melhor * dt


def features_janela(buf):
    t = np.array([b[0] for b in buf])
    e = np.array([b[1] for b in buf])
    m = np.array([b[2] for b in buf])
    if np.isnan(e).mean() > 0.3:
        return None                      # pouco rosto na janela
    dt = float(np.median(np.diff(t)))
    fechado = e < LIMIAR_EAR
    return {
        "perclos": fechado.mean(),
        "ear_medio": np.nanmean(e),
        "ear_min": np.nanmin(e),
        "ear_desvio": np.nanstd(e),
        "maior_fechamento_s": maior_sequencia(fechado, dt),
        "piscadas": int(np.sum(np.diff(fechado.astype(int)) == 1)),
        "mar_medio": np.nanmean(m),
        "boca_aberta": np.nanmean(m > 0.5),
    }


face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=False)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") \
    else cv2.VideoCapture(0)

buf = deque()
t0 = time.time()
prox_medida = 0.0
ultimo_calculo = 0.0
ultimo_bip = -10.0
prob = None
seguidos = 0
fechado_desde = None
e = np.nan

while True:
    ok, frame = cap.read()
    if not ok:
        break
    agora = time.time() - t0
    h, w = frame.shape[:2]

    # 1) mede o rosto 15 vezes por segundo
    if agora >= prox_medida:
        prox_medida = agora + 1.0 / TAXA
        res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            e = (ear(lm, OLHO_ESQ, w, h) + ear(lm, OLHO_DIR, w, h)) / 2
            m = mar(lm, w, h)
        else:
            e = m = np.nan
        buf.append((agora, e, m))
        while buf and agora - buf[0][0] > JANELA:
            buf.popleft()

        if not np.isnan(e) and e < LIMIAR_EAR:
            if fechado_desde is None:
                fechado_desde = agora
        else:
            fechado_desde = None

    # 2) a cada 1 segundo, pergunta ao modelo
    pronto = bool(buf) and agora - buf[0][0] >= JANELA - 0.5
    if pronto and agora - ultimo_calculo >= 1.0:
        ultimo_calculo = agora
        f = features_janela(buf)
        if f is None:
            prob, seguidos = None, 0
        else:
            prob = modelo.predict_proba(pd.DataFrame([f])[FEATURAS])[0, 1]
            seguidos = seguidos + 1 if prob >= LIMIAR_PROB else 0

    alarme_modelo = seguidos >= 2
    fechado_seg = 0.0 if fechado_desde is None else agora - fechado_desde
    alarme_regra = fechado_seg > 2.0

    # 3) desenha na tela
    if not pronto:
        status = f"calibrando... {max(0, JANELA - (agora - buf[0][0] if buf else 0)):.0f}s"
    elif prob is None:
        status = "rosto nao detectado"
    else:
        status = f"modelo: {prob:.0%} sonolento"
    cv2.putText(frame, f"EAR {e:.2f}   olhos fechados: {fechado_seg:.1f}s", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if alarme_modelo or alarme_regra:
        motivo = "MODELO" if alarme_modelo else "REGRA 2s"
        cv2.putText(frame, f"ALERTA: SONOLENCIA! ({motivo})", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if winsound and agora - ultimo_bip > 3:
            winsound.Beep(1000, 400)
            ultimo_bip = agora

    cv2.imshow("Detector de sonolencia (ESC para sair)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()