"""Inventário dos vídeos: lê data/raw e salva data/interim/inventario.csv"""
from pathlib import Path
import cv2
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
SAIDA = RAIZ / "data" / "interim" / "inventario.csv"
SAIDA.parent.mkdir(parents=True, exist_ok=True)

linhas = []
for arq in sorted(RAW.rglob("*")):
    if arq.suffix.lower() not in (".mov", ".mp4"):
        continue

    cap = cv2.VideoCapture(str(arq))
    abriu = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if abriu else 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if abriu else 0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if abriu else 0
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if abriu else 0
    cap.release()

    linhas.append({
        "lote": arq.relative_to(RAW).parts[0],
        "pessoa": arq.parent.name,
        "rotulo": int(arq.stem),
        "arquivo": str(arq.relative_to(RAIZ)),
        "tamanho_gb": round(arq.stat().st_size / 1e9, 2),
        "abriu": abriu,
        "largura": largura,
        "altura": altura,
        "fps": round(fps, 2),
        "duracao_min": round(frames / fps / 60, 1) if fps > 0 else 0,
    })

tabela = pd.DataFrame(linhas)
tabela.to_csv(SAIDA, index=False)

colunas = ["lote", "pessoa", "rotulo", "tamanho_gb", "abriu",
           "largura", "altura", "fps", "duracao_min"]
print(tabela[colunas].to_string(index=False))
print("\nTotal de vídeos:", len(tabela))
print("Vídeos que abriram:", int(tabela["abriu"].sum()))