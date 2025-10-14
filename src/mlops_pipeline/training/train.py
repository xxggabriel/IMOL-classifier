from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from ..config import (
    ARTIFACTS_DIR, DEFAULT_TRAIN_FILE, FASE_COL, OPER_COL, TARGET_FASES,
    CAM_COLS, TEST_SIZE, RANDOM_STATE
)
from ..features.transforms import CamLabelGenerator, ensure_list
from ..models.pipelines import create_training_pipelines
from ..utils.io import load_excel

def run_training(train_path: str | None = None):
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    path = train_path or DEFAULT_TRAIN_FILE
    print(f"Carregando dados de: {path}")
    df_raw = load_excel(path)

    print("Gerando rotulos a partir dos CAM tokens")
    label_generator = CamLabelGenerator(cam_cols=CAM_COLS)
    label_generator.fit(df_raw)
    df_with_labels = label_generator.transform(df_raw)

    df_with_labels[FASE_COL] = df_with_labels[FASE_COL].apply(ensure_list)
    df_with_labels[OPER_COL] = df_with_labels[OPER_COL].apply(ensure_list)
    df_with_labels["LABELS_FASE_FILTRADAS"] = df_with_labels[FASE_COL].apply(
        lambda f: [int(i) for i in f if str(i).isdigit() and int(i) in TARGET_FASES]
    )
    mask_keep = (df_with_labels["LABELS_FASE_FILTRADAS"].apply(len) > 0) | (df_with_labels[OPER_COL].apply(len) > 0)
    df_raw_filtered = df_raw[mask_keep].copy().reset_index(drop=True)
    df_labels_filtered = df_with_labels[mask_keep].copy().reset_index(drop=True)

    mlb_fase = MultiLabelBinarizer(classes=sorted(list(TARGET_FASES)))
    Y_fase = mlb_fase.fit_transform(df_labels_filtered["LABELS_FASE_FILTRADAS"])

    mlb_op = MultiLabelBinarizer()
    Y_op = mlb_op.fit_transform(df_labels_filtered[OPER_COL])

    X = df_raw_filtered

    Y_joint = np.hstack([Y_fase, Y_op])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(msss.split(X, Y_joint))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Yf_train, Yf_test = Y_fase[train_idx], Y_fase[test_idx]
    Yo_train, Yo_test = Y_op[train_idx], Y_op[test_idx]
    print(f"Split: {len(X_train)} treino, {len(X_test)} teste")

    model_fase, model_op = create_training_pipelines()

    print("Treinando FASE")
    model_fase.fit(X_train, Yf_train)

    print("Treinando OPERACAO")
    model_op.named_steps['add_fase_preds'].fase_model = Pipeline(model_fase.steps[2:])
    model_op.named_steps['add_fase_preds'].mlb_fase = mlb_fase
    model_op.fit(X_train, Yo_train)

    print("Avaliacao FASE")
    Yf_pred = model_fase.predict(X_test)
    print(classification_report(Yf_test, Yf_pred, target_names=[str(c) for c in mlb_fase.classes_], zero_division=0))

    print("Avaliacao OPERACAO")
    Yo_pred = model_op.predict(X_test)
    print(classification_report(Yo_test, Yo_pred, target_names=[str(c) for c in mlb_op.classes_], zero_division=0))

    joblib.dump(model_fase, ARTIFACTS_DIR / "model_fase.joblib")
    joblib.dump(model_op, ARTIFACTS_DIR / "model_op.joblib")
    joblib.dump(mlb_fase, ARTIFACTS_DIR / "mlb_fase.joblib")
    joblib.dump(mlb_op, ARTIFACTS_DIR / "mlb_op.joblib")
    print(f"Artefatos salvos em {ARTIFACTS_DIR}")

if __name__ == "__main__":
    run_training()