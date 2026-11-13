"""Treina e avalia o detector de sonolência (alerta x sonolento)."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data" / "interim" / "landmarks"
MODELOS = RAIZ / "models"
MODELOS.mkdir(exist_ok=True)

JANELA = 10         # segundos analisados de cada vez
PASSO = 5           # a janela anda de 5 em 5 segundos
LIMIAR_EAR = 0.75   # olho fechado = abaixo de 75% do EAR normal da pessoa

OLHO_ESQ = [362, 385, 387, 263, 373, 380]
OLHO_DIR = [33, 160, 158, 133, 153, 144]


# ---------- medidas do rosto ----------
def xy(d, i):
    return np.column_stack([d[f"x{i}"].to_numpy() * d["w"].to_numpy(),
                            d[f"y{i}"].to_numpy() * d["h"].to_numpy()])


def dist(d, a, b):
    return np.linalg.norm(xy(d, a) - xy(d, b), axis=1)


def ear(d, ids):  # abertura do olho
    return (dist(d, ids[1], ids[5]) + dist(d, ids[2], ids[4])) / (2 * dist(d, ids[0], ids[3]))


def maior_sequencia(fechado, dt):
    melhor = atual = 0
    for f in fechado:
        atual = atual + 1 if f else 0
        melhor = max(melhor, atual)
    return melhor * dt


# ---------- 0. referência de cada pessoa (vídeo de alerta) ----------
base = {}
for arq in PASTA.glob("*_0.parquet"):
    d = pd.read_parquet(arq)
    base[arq.stem.rsplit("_", 1)[0]] = np.nanmedian((ear(d, OLHO_ESQ) + ear(d, OLHO_DIR)) / 2)

# ---------- 1. montar a tabela de janelas ----------
linhas = []
for arq in sorted(PASTA.glob("*.parquet")):
    pessoa, rotulo = arq.stem.rsplit("_", 1)      # ex.: Fold1_part1_01 e 0
    d = pd.read_parquet(arq)
    e = (ear(d, OLHO_ESQ) + ear(d, OLHO_DIR)) / 2 / base[pessoa]   # EAR relativo
    t = d["t"].to_numpy()
    dt = float(np.median(np.diff(t)))

    ini = 0.0
    while ini + JANELA <= t.max():
        sel = (t >= ini) & (t < ini + JANELA)
        ej = e[sel]
        ini += PASSO
        if np.isnan(ej).mean() > 0.3:             # janela com pouco rosto: ignora
            continue
        fechado = ej < LIMIAR_EAR
        linhas.append({
            "pessoa": pessoa,
            "sonolento": int(rotulo == "10"),
            "perclos": fechado.mean(),
            "ear_medio": np.nanmean(ej),
            "ear_min": np.nanmin(ej),
            "ear_desvio": np.nanstd(ej),
            "maior_fechamento_s": maior_sequencia(fechado, dt),
            "piscadas": int(np.sum(np.diff(fechado.astype(int)) == 1)),
        })

df = pd.DataFrame(linhas)
FEATURAS = [c for c in df.columns if c not in ("pessoa", "sonolento")]
print(f"{len(df)} janelas de {JANELA}s, de {df['pessoa'].nunique()} pessoas")
print(df.groupby("sonolento").size().rename({0: "alerta", 1: "sonolento"}).to_string())

X, y, grupo = df[FEATURAS], df["sonolento"], df["pessoa"]


# ---------- 2. treinar e testar: cada pessoa fica de fora uma vez ----------
prob = np.zeros(len(df))
for treino, teste in LeaveOneGroupOut().split(X, y, grupo):
    modelo = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    min_samples_leaf=10, random_state=42)
    modelo.fit(X.iloc[treino], y.iloc[treino])
    prob[teste] = modelo.predict_proba(X.iloc[teste])[:, 1]

previsto = (prob >= 0.5).astype(int)
regra = (df["perclos"] > 0.2).astype(int)         # regra simples para comparar


def mostrar(nome, pred, pontuacao):
    print(f"\n=== {nome} ===")
    print("AUC (0.5 = chute)          :", round(roc_auc_score(y, pontuacao), 3))
    print("recall (sonolentos achados):", round(recall_score(y, pred), 3))
    print("precisão (alarmes certos)  :", round(precision_score(y, pred, zero_division=0), 3))
    print("matriz [linhas=real, colunas=previsto], ordem alerta, sonolento:")
    print(confusion_matrix(y, pred))


mostrar("MODELO (testado em pessoas que nunca viu)", previsto, prob)
mostrar("REGRA SIMPLES (perclos > 0.2)", regra, df["perclos"])

print("\nAcerto por pessoa (modelo):")
df["acertou"] = previsto == y
print(df.groupby("pessoa")["acertou"].mean().round(3).to_string())


# ---------- 3. modelo final ----------
final = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               min_samples_leaf=10, random_state=42)
final.fit(X, y)
joblib.dump({"modelo": final, "features": FEATURAS, "referencia_ear": base},
            MODELOS / "modelo.joblib")

print("\nMedidas mais importantes:")
imp = pd.Series(final.feature_importances_, index=FEATURAS).sort_values(ascending=False)
print(imp.round(3).to_string())
print("\nModelo salvo em models/modelo.joblib")