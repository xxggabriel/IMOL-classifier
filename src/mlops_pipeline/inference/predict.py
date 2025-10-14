from functools import lru_cache
from typing import Iterable

import joblib
import pandas as pd

from ..config import ARTIFACTS_DIR
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
    pred_op_bin = model_op.predict(df)

    predicted_fases = mlb_fase.inverse_transform(pred_fase_bin)
    predicted_ops = mlb_op.inverse_transform(pred_op_bin)

    results = df.copy()
    results["PREDICOES_FASE"] = [list(p) for p in predicted_fases]
    results["PREDICOES_OPERACAO"] = [list(p) for p in predicted_ops]
    return results

def predict_file(input_file, output_file):
    df = load_excel(input_file)
    out = make_prediction_df(df)
    save_excel(out, output_file)
    return output_file


def predict_records(records: Iterable[dict]) -> pd.DataFrame:
    """Run inference from an iterable of dictionaries."""
    df = pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("Nenhum registro fornecido para predição")
    return make_prediction_df(df)
