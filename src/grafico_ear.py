"""Calcula o EAR dos landmarks salvos e compara alerta x sonolento."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # salva a figura sem abrir janela
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data" / "interim" / "landmarks"
FIG = RAIZ / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33, 160, 158, 133, 153, 144]


def ear_olho(d, ids):
    p = [np.column_stack([d[f"x{i}"] * d["w"], d[f"y{i}"] * d["h"]]) for i in ids]
    vertical = np.linalg.norm(p[1] - p[5], axis=1) + np.linalg.norm(p[2] - p[4], axis=1)
    horizontal = np.linalg.norm(p[0] - p[3], axis=1)
    return vertical / (2.0 * horizontal)


def carregar(nome):
    d = pd.read_parquet(PASTA / f"{nome}.parquet")
    d["ear"] = (ear_olho(d, OLHO_ESQ) + ear_olho(d, OLHO_DIR)) / 2.0
    return d


fig, eixos = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
print(f"{'video':<20}{'EAR mediano':>12}{'% tempo EAR<0,2':>18}")

for n, pessoa in enumerate(range(1, 7)):
    for rotulo, cor, nome_rot in [(0, "tab:blue", "alerta"), (10, "tab:red", "sonolento")]:
        nome = f"Fold1_part1_{pessoa:02d}_{rotulo}"
        d = carregar(nome)
        suave = d["ear"].rolling(5, center=True, min_periods=1).median()
        eixos[n].plot(d["t"] / 60, suave, color=cor, lw=0.5, alpha=0.8, label=nome_rot)
        print(f"{nome:<20}{d['ear'].median():>12.3f}{(d['ear'] < 0.2).mean():>17.1%}")
    eixos[n].axhline(0.2, color="gray", ls="--", lw=0.8)
    eixos[n].set_ylabel(f"pessoa {pessoa:02d}\nEAR")
    eixos[n].set_ylim(0, 0.5)
    if n == 0:
        eixos[n].legend(loc="upper right")

eixos[-1].set_xlabel("minutos de vídeo")
plt.tight_layout()
plt.savefig(FIG / "ear_por_pessoa.png", dpi=110)
print("\nFigura salva em reports/figures/ear_por_pessoa.png")