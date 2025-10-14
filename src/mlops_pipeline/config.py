from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

DEFAULT_TRAIN_FILE = DATA_DIR / "planilha_bruta.xlsx"

FASE_COL = "LABELS_FASE"
OPER_COL = "LABELS_OPERACAO"
TARGET_FASES = {10, 15, 20, 26, 30, 34, 40, 50}

CATEGORICAL_FEATURES = ["CODIGO_5", "CODIGO_MATERIAL_5"]
BINARY_FEATURES = [
    "OP_SUP", "OP_INF", "OP_LAT",
    "FU_SUP", "FU_INF", "FU_LAT",
    "CAV_SUP", "CAV_INF", "CAV_LAT",
    "FRE_SUP", "FRE_INF", "FRE_LAT",
    "RECTANGULAR", "MATTER_GRAIN_ORIENTATION", "GRAIN_ORIENTATION"
]
NUMERIC_FLOAT_FEATURES = ["MASSA", "SURFACE", "COMP", "LARG", "ESP"]
NUMERIC_INT_FEATURES = ["LEVEL_ATTRIBUTE", "INDEX_3D"]

CAM_COLS = [
    "CAM_FILE_NAME_0", "CAM_FILE_NAME_1", "CAM_FILE_NAME_2",
    "CAM_FILE_NAME_3", "CAM_FILE_NAME_4", "CAM_FILE_NAME_6"
]

RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1
}

TEST_SIZE = 0.25
RANDOM_STATE = 42