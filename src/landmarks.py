"""Extrai landmarks do rosto de cada vídeo e salva um arquivo por vídeo."""
import time
from pathlib import Path
import cv2
import pandas as pd
import mediapipe as mp

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data" / "interim" / "landmarks"
SAIDA.mkdir(parents=True, exist_ok=True)

TAXA = 15            # quadros analisados por segundo de vídeo
LADO_MAX = 640       # redimensiona para no máximo 640 px
SO_UM_VIDEO = False   # True = teste com 1 vídeo; depois mude para False

OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33, 160, 158, 133, 153, 144]
BOCA = [13, 82, 312, 14, 87, 317, 78, 308]
CABECA = [1, 152, 61, 291]
IDS = sorted(set(OLHO_ESQ + OLHO_DIR + BOCA + CABECA))

inv = pd.read_csv(RAIZ / "data" / "interim" / "inventario.csv")
inv = inv[inv["rotulo"].isin([0, 10])]
if SO_UM_VIDEO:
    inv = inv.head(1)

for _, v in inv.iterrows():
    nome = f"{v['lote']}_{int(v['pessoa']):02d}_{v['rotulo']}"
    destino = SAIDA / f"{nome}.parquet"
    if destino.exists():
        print(nome, "já feito, pulando")
        continue

    inicio = time.time()
    cap = cv2.VideoCapture(str(RAIZ / v["arquivo"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False)

    linhas = []
    indice = 0
    proximo = 0.0
    while True:
        if not cap.grab():          # avança um quadro
            break
        t = indice / fps            # tempo real do quadro, em segundos
        indice += 1
        if t + 1e-6 < proximo:      # pula quadros para chegar em 15 por segundo
            continue
        proximo += 1.0 / TAXA

        ok, quadro = cap.retrieve()
        if not ok:
            continue
        h, w = quadro.shape[:2]
        escala = LADO_MAX / max(h, w)
        if escala < 1:
            quadro = cv2.resize(quadro, (int(w * escala), int(h * escala)))
            h, w = quadro.shape[:2]

        res = detector.process(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
        linha = {"t": round(t, 3), "w": w, "h": h}
        if res.multi_face_landmarks:
            pontos = res.multi_face_landmarks[0].landmark
            linha["rosto"] = True
            for k in IDS:
                linha[f"x{k}"] = pontos[k].x
                linha[f"y{k}"] = pontos[k].y
        else:
            linha["rosto"] = False
        linhas.append(linha)

    cap.release()
    detector.close()

    tabela = pd.DataFrame(linhas)
    tabela.to_parquet(destino, index=False)
    minutos = (time.time() - inicio) / 60
    print(f"{nome}: {len(tabela)} quadros, rosto em {tabela['rosto'].mean():.1%}, "
          f"levou {minutos:.1f} min")

# Resumo de todos os vídeos já processados
resumo = []
for arq in sorted(SAIDA.glob("*.parquet")):
    d = pd.read_parquet(arq, columns=["t", "rosto"])
    resumo.append({"video": arq.stem, "quadros": len(d),
                   "duracao_min": round(d["t"].max() / 60, 1),
                   "deteccao": round(d["rosto"].mean(), 3)})
resumo = pd.DataFrame(resumo)
resumo.to_csv(RAIZ / "data" / "interim" / "deteccao.csv", index=False)
print("\n", resumo.to_string(index=False))