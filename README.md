Projeto de classificacao de fases e operacoes

Estrutura modular para treinar e servir dois modelos encadeados.
Primeiro classifica as fases e em seguida usa as probabilidades de fases como features adicionais para classificar operacoes.

Uso
1. Crie um ambiente virtual e instale as dependencias com pip install -r requirements.txt
2. Coloque o arquivo planilha_bruta.xlsx em data
3. Rode o treino com python -m mlops_pipeline.training.train ou python scripts/train.py
4. Rode a inferencia com python scripts/predict.py data/arquivo.xlsx --output_file saida.xlsx