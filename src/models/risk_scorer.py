"""
Risk scoring model for PSEN1 mutations using ESM-2 features, BLOSUM62, and PolyPhen labels.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


def train_risk_model(df_features: pd.DataFrame,
                     df_labels: pd.DataFrame,
                     feature_cols: list = None,
                     test_size: float = 0.2,
                     random_state: int = 42) -> LogisticRegression:
    """
    Join ESM-2 features and PolyPhen labels on (position_1based, wt_aa, mut_aa),
    then train a logistic regression model.
    
    Args:
        df_features: DataFrame with ESM-2 features (delta_loglik, entropy, etc.)
        df_labels: DataFrame with PolyPhen labels (position_1based, wt_aa, mut_aa, polyphen_label)
        feature_cols: List of feature column names. Default: ['delta_loglik', 'entropy', 'blosum62']
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Trained LogisticRegression model
    """
    if feature_cols is None:
        feature_cols = ['delta_loglik', 'entropy', 'blosum62']
    
    # Inner join ensures we only use mutations that are labeled by PolyPhen
    df_train = df_features.merge(
        df_labels,
        on=["position_1based", "wt_aa", "mut_aa"],
        how="inner"
    )

    print(f"[TRAIN] Number of labeled mutations: {len(df_train)}")
    if len(df_train) < 10:
        print("[WARN] Very few labeled examples. Metrics may be unstable.")

    X = df_train[feature_cols]
    y = df_train["polyphen_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # try to preserve class balance in train/val
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)

    # If validation set accidentally has only one class, skip AUC.
    if len(np.unique(y_val)) < 2:
        print("[TRAIN] Warning: validation set has only one class; ROC AUC is undefined.")
        auc = float("nan")
    else:
        auc = roc_auc_score(y_val, y_prob)

    acc = accuracy_score(y_val, y_pred)

    print(f"[TRAIN] Validation AUC: {auc:.4f}")
    print(f"[TRAIN] Validation ACC: {acc:.4f}")

    return clf


def score_variants(df_features: pd.DataFrame,
                   model: LogisticRegression,
                   feature_cols: list = None,
                   output_col: str = 'risk_score') -> pd.DataFrame:
    """
    Use the trained logistic regression model to compute risk scores
    for every variant in the DataFrame.
    
    Args:
        df_features: DataFrame with mutation features
        model: Trained LogisticRegression model
        feature_cols: List of feature column names. Default: ['delta_loglik', 'entropy', 'blosum62']
        output_col: Name of output column for risk scores
    
    Returns:
        DataFrame with added risk_score column
    """
    if feature_cols is None:
        feature_cols = ['delta_loglik', 'entropy', 'blosum62']
    
    df = df_features.copy()
    X_all = df[feature_cols]
    df[output_col] = model.predict_proba(X_all)[:, 1]
    return df


def save_model(model: LogisticRegression, path: Path):
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[MODEL] Saved trained model to: {path}")


def load_model(path: Path) -> LogisticRegression:
    """Load trained model from disk."""
    return joblib.load(path)

