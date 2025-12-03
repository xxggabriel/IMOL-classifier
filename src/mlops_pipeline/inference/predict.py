from functools import lru_cache
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    NUMERIC_FLOAT_FEATURES,
    NUMERIC_INT_FEATURES,
)
from ..utils.io import load_excel, save_excel


@lru_cache(maxsize=1)
def load_artifacts():
    """Load and cache inference artifacts."""
    model_fase = joblib.load(ARTIFACTS_DIR / "model_fase.joblib")
    model_op = joblib.load(ARTIFACTS_DIR / "model_op.joblib")
    mlb_fase = joblib.load(ARTIFACTS_DIR / "mlb_fase.joblib")
    mlb_op = joblib.load(ARTIFACTS_DIR / "mlb_op.joblib")
    return model_fase, model_op, mlb_fase, mlb_op


def make_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    model_fase, model_op, mlb_fase, mlb_op = load_artifacts()

    pred_fase_bin = model_fase.predict(df)
    predicted_fases = mlb_fase.inverse_transform(pred_fase_bin)

    op_proba = model_op.predict_proba(df)
    if isinstance(op_proba, list):
        # OneVsRestClassifier may return a list in some sklearn versions.
        op_proba = np.vstack([p[:, 1] for p in op_proba]).T

    aligned_ops = _align_ops_to_fases(predicted_fases, op_proba, mlb_op.classes_)

    results = df.copy()
    results["PREDICOES_FASE"] = [list(p) for p in predicted_fases]
    results["PREDICOES_OPERACAO"] = aligned_ops
    return results


def predict_file(input_file, output_file):
    df = load_excel(input_file)
    out = make_prediction_df(df)
    save_excel(out, output_file)
    return output_file


def predict_records(records: Iterable[dict]) -> pd.DataFrame:
    """Run inference from an iterable of dictionaries."""
    df = pd.DataFrame(list(records))
    df = _prepare_input_dataframe(df)
    if df.empty:
        raise ValueError("Nenhum registro fornecido para predição")
    return make_prediction_df(df)


def _prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming dataframe to align with training data expectations."""
    if df.empty:
        return df

    df = df.replace(to_replace=r"^\s*$", value=pd.NA, regex=True)

    numeric_cols = set(NUMERIC_FLOAT_FEATURES) | set(NUMERIC_INT_FEATURES)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _align_ops_to_fases(predicted_fases, op_proba: np.ndarray, op_classes) -> list[list]:
    """Keep operations count consistent with predicted phases using probability ranking."""
    op_classes = list(op_classes)
    aligned_ops: list[list] = []
    for fases_row, proba_row in zip(predicted_fases, op_proba):
        fase_count = len(fases_row)
        if fase_count == 0:
            aligned_ops.append([])
            continue
        sorted_idx = np.argsort(proba_row)[::-1]
        selected: list = []
        for idx in sorted_idx:
            if proba_row[idx] >= 0.5:
                selected.append(op_classes[idx])
            if len(selected) == fase_count:
                break
        if len(selected) < fase_count:
            for idx in sorted_idx:
                op_label = op_classes[idx]
                if op_label in selected:
                    continue
                selected.append(op_label)
                if len(selected) == fase_count:
                    break
        aligned_ops.append(selected)
    return aligned_ops
