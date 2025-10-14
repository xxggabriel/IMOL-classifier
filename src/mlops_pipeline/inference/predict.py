import joblib
import pandas as pd
from ..config import ARTIFACTS_DIR
from ..utils.io import load_excel, save_excel

def make_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    model_fase = joblib.load(ARTIFACTS_DIR / "model_fase.joblib")
    model_op = joblib.load(ARTIFACTS_DIR / "model_op.joblib")
    mlb_fase = joblib.load(ARTIFACTS_DIR / "mlb_fase.joblib")
    mlb_op = joblib.load(ARTIFACTS_DIR / "mlb_op.joblib")

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