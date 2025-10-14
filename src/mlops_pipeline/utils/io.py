import pandas as pd
from pathlib import Path

def load_excel(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)

def save_excel(df: pd.DataFrame, path: str | Path) -> None:
    df.to_excel(path, index=False)