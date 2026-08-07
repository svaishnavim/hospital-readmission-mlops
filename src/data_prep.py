"""
data_prep.py — Stage 2: data cleaning, feature engineering, leakage-safe
preprocessing and splitting for the Hospital Readmission pipeline.

Implement the functions below; they are imported by Data_Preparation.ipynb and by
src/train.py. Expected end state: a model frame of ~69,973 rows at ~9% positive, and a
stratified train/val/test split with the preprocessor fit on the TRAIN split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw():
    """TODO: read cfg.RAW_CSV with na_values=['?'] so '?' becomes NaN; return the DataFrame."""
    df = pd.read_csv(cfg.RAW_CSV, na_values=["?"])
    return df


def clean(df):
    """TODO (Stage 2.1): keep the FIRST encounter per patient (drop_duplicates('patient_nbr'))
    to prevent patient-level leakage; drop expired/hospice discharges
    (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS); build the binary target
    cfg.TARGET = 1 if readmitted == '<30' else 0; drop the cfg.DROP_COLS that are present.
    Return the cleaned frame (~69,973 rows, ~9% positive)."""

    df = df.copy()
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    df = df[
        ~df["discharge_disposition_id"].isin(
            cfg.EXPIRED_HOSPICE_DISPOSITIONS
        )
    ]
    df[cfg.TARGET] = (df["readmitted"] == "<30").astype(int)
    drop_cols = [c for c in cfg.DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def engineer_features(df):
    """TODO (Stage 2.2): group ICD9 diag_1/2/3 into clinical categories; map the age band to a
    numeric midpoint; service_utilization = number_outpatient + number_emergency + number_inpatient;
    num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; collapse high-cardinality
    medical_specialty to top-k + 'Other'. Return the engineered frame."""
    df = df.copy()

    def group_icd9(code):
        if pd.isna(code):
            return "Unknown"
        code = str(code)
        if code.startswith("V"):
            return "Supplementary"
        if code.startswith("E"):
            return "Injury"
        try:
            code = float(code)
        except ValueError:
            return "Other"
        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif 580 <= code <= 629:
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(group_icd9)

    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    df["age_midpoint"] = df["age"].map(age_map)
    df["service_utilization"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )
    df["num_med_changes"] = (
        df[cfg.MED_COLS]
        .isin(["Up", "Down"])
        .sum(axis=1)
    )
    top_k = 10
    top_specialties = (
        df["medical_specialty"]
        .value_counts()
        .nlargest(top_k)
        .index
    )
    df["medical_specialty_grouped"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_specialties),
        "Other"
    )
    return df


def get_feature_lists(df):
    """TODO: return (numeric_cols, categorical_cols), excluding cfg.TARGET."""
    feature_df = df.drop(columns=[cfg.TARGET])
    numeric_cols = feature_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric, categorical):
    """TODO (Stage 2.3): return ONE ColumnTransformer — numeric: median SimpleImputer + StandardScaler;
    categorical: constant SimpleImputer + OneHotEncoder(handle_unknown='ignore'). Do NOT fit it here
    (it is fit on the TRAIN split only)."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )
    return preprocessor


def get_model_frame():
    """TODO: return engineer_features(clean(load_raw()))."""
    return engineer_features(clean(load_raw()))


def get_splits(df=None):
    """TODO (Stage 2.4): create a STRATIFIED train/val/test split using cfg.RANDOM_STATE,
    cfg.TEST_SIZE and cfg.VAL_SIZE (val taken from the train split). Return
    X_train, X_val, X_test, y_train, y_val, y_test. The class ratio must be preserved across splits;
    the preprocessor is fit on TRAIN only (in the notebook / train.py), never on the full data."""
    if df is None:
        df = get_model_frame()
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=cfg.VAL_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y_train_val
    )
    return (
        X_train, X_val, X_test,y_train,y_val,y_test,
    )
