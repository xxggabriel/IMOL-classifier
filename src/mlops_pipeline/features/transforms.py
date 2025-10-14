import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def to_binary(val: object):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NA
    s = str(val).strip().upper()
    s = (s.replace("Ã", "A").replace("Á", "A").replace("Â", "A").replace("À", "A")
           .replace("Õ", "O").replace("Ó", "O").replace("Ô", "O")
           .replace("É", "E").replace("Ê", "E").replace("Í", "I").replace("Ú", "U"))
    if s in {"SIM", "YES", "TRUE", "VERDADEIRO", "1"}:
        return 1
    if s in {"NAO", "NAO.", "N", "NA", "NO", "FALSE", "FALSO", "0"}:
        return 0
    if re.fullmatch(r"\d+", s):
        return 1 if int(s) != 0 else 0
    return pd.NA

def first_5_digits_stream(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)): return ""
    only_digits = re.sub(r"\D", "", str(text))
    return only_digits[:5] if only_digits else ""

def is_empty(x: object) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)): return True
    return str(x).strip() == ""

def parse_cam_token(cam_value: object) -> dict:
    if cam_value is None or (isinstance(cam_value, float) and pd.isna(cam_value)):
        return {}
    parts = [p for p in str(cam_value).strip().split("|") if p != ""]
    if len(parts) < 3:
        return {}
    def clean(v):
        if v is None: return None
        s = str(v).strip()
        return None if s in {"0", "-", "NA", "N/A"} else s
    return {
        "sequencia": clean(parts[0]),
        "operacao":  clean(parts[1] if len(parts) > 1 else None),
        "maquina":   clean(parts[2] if len(parts) > 2 else None),
        "fase":      clean(parts[4] if len(parts) > 4 else None),
    }

def ensure_list(x: Any) -> list:
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            import ast
            v = ast.literal_eval(x)
            if isinstance(v, list): return v
        except Exception:
            pass
        return [x] if x.strip() else []
    return [x]

class BinarizeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_binary: list[str]):
        self.cols_binary = cols_binary
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols_to_process = [c for c in self.cols_binary if c in df.columns]
        for c in cols_to_process:
            df[c] = df[c].apply(to_binary).astype("Int64")
        return df

class CamLabelGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, cam_cols: list[str]):
        self.cam_cols = cam_cols
        self.unique_ops_ = []
        self.unique_maqs_ = []
        self.unique_fases_ = []
    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in self.cam_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Colunas CAM faltando: {missing}")
        parsed_data = {}
        for col in self.cam_cols:
            parsed = X[col].apply(parse_cam_token)
            parsed_data[f"{col}__OPERACAO"] = parsed.apply(lambda d: d.get("operacao"))
            parsed_data[f"{col}__MAQUINA"] = parsed.apply(lambda d: d.get("maquina"))
            parsed_data[f"{col}__FASE"] = parsed.apply(lambda d: d.get("fase"))
        def collect_uniques(tag: str) -> list[str]:
            vals = set()
            import pandas as pd
            for col in self.cam_cols:
                series = pd.Series(parsed_data[f"{col}__{tag}"]).dropna().astype(str).str.strip()
                vals.update([v for v in series.unique() if v not in {"None", "0", "-", "nan", ""}])
            return sorted(list(vals))
        self.unique_ops_ = collect_uniques("OPERACAO")
        self.unique_maqs_ = collect_uniques("MAQUINA")
        self.unique_fases_ = collect_uniques("FASE")
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        import pandas as pd
        parsed_cols = {}
        for col in self.cam_cols:
            parsed = df[col].apply(parse_cam_token)
            parsed_cols[f"{col}__OPERACAO"] = parsed.apply(lambda d: d.get("operacao"))
            parsed_cols[f"{col}__MAQUINA"] = parsed.apply(lambda d: d.get("maquina"))
            parsed_cols[f"{col}__FASE"] = parsed.apply(lambda d: d.get("fase"))
        df_parsed = pd.DataFrame(parsed_cols, index=df.index)
        def join_labels(row: pd.Series, tag: str, universe: list[str]) -> list[str]:
            seen = set()
            for cam_col in self.cam_cols:
                val = row.get(f"{cam_col}__{tag}")
                if pd.notna(val) and str(val).strip():
                    seen.add(str(val).strip())
            return [v for v in universe if v in seen]
        df["LABELS_OPERACAO"] = df_parsed.apply(lambda r: join_labels(r, "OPERACAO", self.unique_ops_), axis=1)
        df["LABELS_MAQUINA"] = df_parsed.apply(lambda r: join_labels(r, "MAQUINA", self.unique_maqs_), axis=1)
        df["LABELS_FASE"] = df_parsed.apply(lambda r: join_labels(r, "FASE", self.unique_fases_), axis=1)
        return df

class CodeGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        def find_col(df_in, candidates):
            for c in df_in.columns:
                if str(c).upper().strip() in candidates:
                    return c
            return None
        codigo_col = find_col(df, {"CODIGO", "CÓDIGO"})
        ref_col = find_col(df, {"REFERENCIA", "REFERÊNCIA"})
        codmat_col = find_col(df, {"CODIGO_MATERIAL", "CÓDIGO_MATERIAL"})
        if codigo_col and ref_col:
            df["CODIGO_5"] = df.apply(
                lambda r: first_5_digits_stream(r.get(ref_col)) if is_empty(r.get(codigo_col)) else first_5_digits_stream(r.get(codigo_col)),
                axis=1
            )
        if codmat_col:
            df["CODIGO_MATERIAL_5"] = df[codmat_col].apply(first_5_digits_stream)
        return df

class FasePredictionFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, fase_model, mlb_fase):
        self.fase_model = fase_model
        self.mlb_fase = mlb_fase
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not hasattr(self.fase_model, "predict_proba"):
            raise ValueError("fase_model precisa de predict_proba")
        probs = self.fase_model.predict_proba(X)
        import numpy as np
        if isinstance(probs, list):
            probs = np.vstack([p[:, 1] for p in probs]).T
        return np.hstack([X, probs])