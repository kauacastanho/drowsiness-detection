import sys

print("Python:", sys.version.split()[0])

pacotes = ["cv2", "mediapipe", "numpy", "pandas", "sklearn",
           "xgboost", "matplotlib", "seaborn", "joblib",
           "tqdm", "pyarrow", "scipy"]

falhas = []
for nome in pacotes:
    try:
        mod = __import__(nome)
        print(f"OK     {nome:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FALHA  {nome}: {e}")
        falhas.append(nome)

import mediapipe as mp
print("API legada do MediaPipe (mp.solutions):",
      "OK" if hasattr(mp, "solutions") else "AUSENTE")

import cv2
cap = cv2.VideoCapture(0)
print("Webcam abre:", "OK" if cap.isOpened() else "FALHA")
cap.release()

print("\nRESULTADO:", "tudo certo" if not falhas else f"corrigir: {falhas}")