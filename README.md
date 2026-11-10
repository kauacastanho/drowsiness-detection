# Detecção de Sonolência do Motorista

Projeto acadêmico de aprendizado de máquina: classifica o estado do motorista
(alerta ou sonolento) a partir de landmarks faciais extraídos de vídeo, com
demonstração em tempo real por webcam.

## Abordagem

Landmarks faciais (MediaPipe Face Mesh) -> features geométricas (EAR, MAR,
pose da cabeça) -> features temporais em janelas deslizantes (PERCLOS,
piscadas, bocejos) -> classificador clássico (Random Forest, SVM, XGBoost).
O objetivo prioriza o recall da classe sonolento.

## Estrutura

```
data/        dados (não versionados)
models/      modelos treinados (não versionados)
notebooks/   análises exploratórias
reports/     figuras e resultados
src/         código-fonte
```

## Como reproduzir o ambiente

Testado em Linux Mint 22.3 com Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/check_env.py
```

## Status

- [x] Etapa 2: ambiente e testes de verificação (MediaPipe, EAR, FPS)
- [ ] Etapa 3: aquisição e exploração dos dados
- [ ] Etapas seguintes: landmarks, features, baseline, modelagem, demo

## Datasets

A definir: UTA-RLDD (principal) e NTHU-DDD (teste externo). Os dados não são
distribuídos neste repositório; consulte os termos de uso de cada dataset.