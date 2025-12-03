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
from ..features.transforms import parse_cam_token
from ..models.pipelines import create_training_pipelines
from ..utils.io import load_excel


def _extract_aligned_labels(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive phase and operation labels keeping only operations tied to target phases."""
    fases, ops = [], []
    for _, row in df_raw.iterrows():
        fases_row, ops_row = [], []
        seen_fases, seen_ops = set(), set()
        for col in CAM_COLS:
            token = parse_cam_token(row.get(col))
            if not token:
                continue
            fase_raw = token.get("fase")
            op_raw = token.get("operacao")
            if fase_raw is None:
                continue
            fase_str = str(fase_raw).strip()
            if not fase_str.isdigit():
                continue
            fase_int = int(fase_str)
            if fase_int not in TARGET_FASES:
                continue
            if fase_int not in seen_fases:
                seen_fases.add(fase_int)
                fases_row.append(fase_int)
            if op_raw is None:
                continue
            op_str = str(op_raw).strip()
            if op_str and op_str not in seen_ops:
                seen_ops.add(op_str)
                ops_row.append(op_str)
        fases.append(fases_row)
        ops.append(ops_row)

    return pd.DataFrame({FASE_COL: fases, OPER_COL: ops})


def run_training(train_path: str | None = None):
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    path = train_path or DEFAULT_TRAIN_FILE
    print(f"Carregando dados de: {path}")
    df_raw = load_excel(path)

    print("Gerando rotulos a partir dos CAM tokens (sincronizando fases e operacoes)")
    df_labels = _extract_aligned_labels(df_raw)

    def _has_op_phase_overlap(row) -> bool:
        fases_set = {str(f).strip() for f in row[FASE_COL] if str(f).strip()}
        return any(str(op).strip() in fases_set for op in row[OPER_COL])

    mask_keep = (
        ((df_labels[FASE_COL].apply(len) > 0) | (df_labels[OPER_COL].apply(len) > 0))
        & ~df_labels.apply(_has_op_phase_overlap, axis=1)
    )
    df_raw_filtered = df_raw[mask_keep].copy().reset_index(drop=True)
    df_labels_filtered = df_labels[mask_keep].copy().reset_index(drop=True)

    mlb_fase = MultiLabelBinarizer(classes=sorted(list(TARGET_FASES)))
    Y_fase = mlb_fase.fit_transform(df_labels_filtered[FASE_COL])

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
