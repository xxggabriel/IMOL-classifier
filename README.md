# MLOps Pipeline – Predição de Fases e Operações

Este projeto disponibiliza uma pipeline modular para treinar e servir dois modelos encadeados:
1. O primeiro modelo prevê as fases associadas a uma peça.
2. O segundo modelo consome as probabilidades do primeiro como features para classificar as operações.

A solução oferece tanto scripts de linha de comando quanto uma API REST construída com FastAPI, além de artefatos prontos para contêiner (Docker).

---

## Estrutura do Projeto

- `data/` – arquivos de entrada (ex.: `planilha_bruta.xlsx`).
- `artifacts/` – modelos e codificadores exportados (`*.joblib`).
- `scripts/train.py` – entrada CLI para treinar os modelos.
- `scripts/predict.py` – entrada CLI para realizar inferência em planilhas Excel.
- `src/mlops_pipeline/` – pacote com configuração, feature engineering, treinamento, inferência e API.
- `Dockerfile`, `docker-compose.yml` – recursos para empacotar e servir a API.

---

---

## Instalação (ambiente local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Treinamento

1. Garanta que o arquivo `data/planilha_bruta.xlsx` esteja presente.
2. Execute:

```bash
python scripts/train.py
# ou
python -m mlops_pipeline.training.train
```

Os artefatos treinados são gravados em `artifacts/`:
- `model_fase.joblib`
- `model_op.joblib`
- `mlb_fase.joblib`
- `mlb_op.joblib`

---

## Inferência via CLI

```bash
python scripts/predict.py data/entrada.xlsx --output_file saida.xlsx
```

O arquivo resultante conterá as colunas `PREDICOES_FASE` e `PREDICOES_OPERACAO` com as previsões por peça.

---

## Servindo a API FastAPI

### Executar localmente (sem Docker)

```bash
PYTHONPATH=src uvicorn mlops_pipeline.api:app --reload --host 0.0.0.0 --port 8000
```

### Executar via Docker Compose

```bash
docker compose up --build
```

O serviço expõe a porta `8000` e monta os diretórios `./artifacts` e `./data` dentro do contêiner, permitindo reutilizar os modelos treinados e arquivos de entrada.

---

## Consumo da API

- Endpoint: `POST /infer/json`
- Corpo: lista de objetos com ao menos o campo `id_peca`. Campos adicionais presentes na base original são aceitos (configuração `extra=allow` no schema).

### Exemplo de payload (`payload.json`)

```json
[
  {
    "id_peca": 123,
    "MASSA": 5.2,
    "OP_SUP": "Sim",
    "...": "outros atributos necessários para o modelo"
  }
]
```

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict/json \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
```

### Exemplo de resposta

```json
[
  {
    "id_peca": 123,
    "fases": ["10", "20"],
    "operacao": ["03", "09"]
  }
]
```

Erros comuns retornam mensagens claras (ex.: ausência de artefatos, payload vazio, `id_peca` inválido).

---

## Notas Adicionais

- Antes de servir a API, garanta que os artefatos `.joblib` existam (execute o treinamento ao menos uma vez ou disponibilize-os manualmente em `artifacts/`).
- Para alterar hiperparâmetros ou colunas utilizadas, edite `src/mlops_pipeline/config.py` e os transformers em `src/mlops_pipeline/features/transforms.py`.
- O projeto utiliza `iterative-stratification` para split estratificado multilabel e Random Forest (`sklearn`) como base dos classificadores.

