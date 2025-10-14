from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from ..config import (
    CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERIC_FLOAT_FEATURES, NUMERIC_INT_FEATURES,
    CAM_COLS, RF_PARAMS, TARGET_FASES
)
from ..features.transforms import (
    BinarizeColumns, CamLabelGenerator, CodeGenerator, FasePredictionFeatureAdder
)

def create_training_pipelines():
    feature_engineering = Pipeline([
        ('binarizer', BinarizeColumns(cols_binary=BINARY_FEATURES)),
        ('cam_label_generator', CamLabelGenerator(cam_cols=CAM_COLS)),
        ('code_generator', CodeGenerator())
    ])

    final_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
            ("bin", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))]), BINARY_FEATURES),
            ("numf", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_FLOAT_FEATURES),
            ("numi", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("sc", StandardScaler())]), NUMERIC_INT_FEATURES),
        ],
        remainder="drop"
    )

    model_fase = Pipeline([
        ('feature_engineering', feature_engineering),
        ('final_preprocessor', final_preprocessor),
        ('clf', OneVsRestClassifier(RandomForestClassifier(**RF_PARAMS), n_jobs=-1))
    ])

    fase_predictor_pipeline = Pipeline(model_fase.steps[2:])
    mlb_fase_temp = MultiLabelBinarizer(classes=sorted(list(TARGET_FASES)))

    model_op = Pipeline([
        ('feature_engineering', clone(feature_engineering)),
        ('final_preprocessor', clone(final_preprocessor)),
        ('add_fase_preds', FasePredictionFeatureAdder(fase_model=fase_predictor_pipeline, mlb_fase=mlb_fase_temp)),
        ('clf', OneVsRestClassifier(RandomForestClassifier(**RF_PARAMS), n_jobs=-1))
    ])

    return model_fase, model_op