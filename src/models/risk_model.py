import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DEFAULT_FEATURE_COLS = ["delta_loglik", "entropy", "blosum62"]

def train_risk_model(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Robustly joins features and labels, and AUTO-AUGMENTS benign samples
    using high-confidence ESM predictions (proxy labels).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    print(f"[TRAIN] Raw Features: {len(df_features)} rows, Raw Labels: {len(df_labels)} rows")

    # --- 1. Internal Data Cleaning ---
    df_feat_clean = df_features.copy()
    df_lbl_clean = df_labels.copy()

    # Cast to safe types
    if "position_1based" in df_feat_clean.columns:
        df_feat_clean["position_1based"] = (
            pd.to_numeric(df_feat_clean["position_1based"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
    if "position_1based" in df_lbl_clean.columns:
        df_lbl_clean["position_1based"] = (
            pd.to_numeric(df_lbl_clean["position_1based"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

    for col in ["wt_aa", "mut_aa"]:
        df_feat_clean[col] = df_feat_clean[col].astype(str).str.strip()
        df_lbl_clean[col] = df_lbl_clean[col].astype(str).str.strip()

    # --- 2. Merge Known Labels ---
    df_train = df_feat_clean.merge(
        df_lbl_clean,
        on=["position_1based", "wt_aa", "mut_aa"],
        how="inner",
    )

    # Current label distribution
    n_pos = len(df_train[df_train["polyphen_label"] == 1])
    n_neg = len(df_train[df_train["polyphen_label"] == 0])

    print(f"[TRAIN] Original Labeled Data: {len(df_train)} samples. (Pos: {n_pos}, Neg: {n_neg})")

    # --- 3. Auto-Augment Benign Samples (Proxy Labels) ---
    # If benign samples are too few, we mine high-confidence benign candidates
    # from the ESM feature table.
    if n_neg < 50:
        # Target is to roughly balance positives and negatives 1:1
        n_needed = n_pos - n_neg
        # Cap the number of augmented samples to avoid overwhelming real labels
        n_augment = min(n_needed, 300)

        print(
            f"[TRAIN] ⚠️ Benign samples are scarce. "
            f"Augmenting with {n_augment} high-likelihood proxy labels..."
        )

        # A. Build a join key to perform an anti-join
        df_feat_clean["join_key"] = (
            df_feat_clean["position_1based"].astype(str)
            + df_feat_clean["wt_aa"]
            + df_feat_clean["mut_aa"]
        )
        df_train["join_key"] = (
            df_train["position_1based"].astype(str)
            + df_train["wt_aa"]
            + df_train["mut_aa"]
        )

        existing_keys = set(df_train["join_key"])

        # B. Filter mutations that are not already labeled
        df_unlabeled = df_feat_clean[~df_feat_clean["join_key"].isin(existing_keys)].copy()

        # C. Sort: select mutations that ESM considers most benign
        # Note: delta_loglik is usually negative; closer to 0 is "safer"
        df_proxies = df_unlabeled.sort_values(
            by="delta_loglik",
            ascending=False,
        ).head(n_augment)

        # D. Assign proxy label 0 (benign)
        df_proxies["polyphen_label"] = 0

        # E. Concatenate with the labeled training set
        cols_to_use = list(df_train.columns)
        common_cols = [c for c in cols_to_use if c in df_proxies.columns]

        df_augmented_benign = df_proxies[common_cols]
        df_train = pd.concat([df_train, df_augmented_benign], axis=0, ignore_index=True)

        print(f"[TRAIN] Augmented Dataset Size: {len(df_train)}")
        print(f"[TRAIN] New Class Balance: {df_train['polyphen_label'].value_counts().to_dict()}")

    # --- 4. Train/Test Split & Training ---
    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = df_train[feature_cols].fillna(0)
    y = df_train["polyphen_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    if X_val is not None and len(X_val) > 0:
        y_prob = clf.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_prob)
            pr_auc = average_precision_score(y_val, y_prob)
            print(f"[TRAIN] Val ROC-AUC: {auc:.4f}")
            print(f"[TRAIN] Val PR-AUC : {pr_auc:.4f}")
        except ValueError:
            print("[TRAIN] Validation set issue (single class).")

    return clf


def score_variants(
    df_features: pd.DataFrame,
    model,
    feature_cols: Optional[List[str]] = None,
    output_col: str = "risk_score",
) -> pd.DataFrame:
    """
    Apply model to new data.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    df = df_features.copy()

    # Data cleaning for scoring as well (handling whitespace)
    for col in ['wt_aa', 'mut_aa']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Try filling missing columns with 0 if they don't exist (e.g. blosum if forgot to add)
        print(f"[WARN] Missing columns {missing} for scoring. Filling with 0.")
        for c in missing:
            df[c] = 0

    X_all = df[feature_cols].fillna(0)
    
    # Predict Probability of Class 1 (Damaging)
    df[output_col] = model.predict_proba(X_all)[:, 1]
    
    return df


def save_model(model, path: Path) -> None:
    """
    Save model using joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[MODEL] Saved model to: {path}")

def load_model(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def plot_risk_distribution(df: pd.DataFrame, 
                           score_col: str = "risk_score", 
                           save_path: Optional[Path] = None):
    """
    Plots a histogram of risk scores to visualize the distribution.
    """
    if score_col not in df.columns:
        print(f"[PLOT] Column {score_col} not found. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.histplot(df[score_col], bins=50, kde=True, color="skyblue", edgecolor="black")
    
    plt.title("Distribution of Risk Scores", fontsize=15)
    plt.xlabel("Risk Score (0=Benign, 1=Pathogenic)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    mean_val = df[score_col].mean()
    median_val = df[score_col].median()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[PLOT] Saved histogram to {save_path}")
    else:
        plt.show()
    
    plt.close()