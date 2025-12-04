#!/usr/bin/env python

from pathlib import Path

import pandas as pd
from Bio.Align import substitution_matrices

from src.models.risk_model import (
    train_risk_model,
    score_variants,
    save_model,
    plot_risk_distribution
)


PROJECT_ROOT = Path("/jet/home/xwang54/GenAIProject_PSEN1")

# ---- Paths ----
PSEN1_ESM = PROJECT_ROOT / "data/outputs/esm2/psen1_esm2_all.csv"
APP_ESM   = PROJECT_ROOT / "data/outputs/esm2/app_esm2_all.csv"

PSEN1_POLY = PROJECT_ROOT / "data/processed/polyphen2/psen1_polyphen_scores.txt"
APP_POLY   = PROJECT_ROOT / "data/processed/polyphen2/app_polyphen_scores.txt"

OUT_DIR_PSEN1 = PROJECT_ROOT / "data/outputs/psen1"
OUT_DIR_APP   = PROJECT_ROOT / "data/outputs/app"


def add_blosum62(df: pd.DataFrame,
                 wt_col: str = "wt_aa",
                 mut_col: str = "mut_aa",
                 out_col: str = "blosum62") -> pd.DataFrame:
    mat = substitution_matrices.load("BLOSUM62")

    def score_pair(wt: str, mut: str) -> int:
        wt = str(wt).strip()
        mut = str(mut).strip()
        pair = (wt, mut)
        rev = (mut, wt)
        if pair in mat:
            return mat[pair]
        if rev in mat:
            return mat[rev]
        return 0

    df = df.copy()
    df[out_col] = df.apply(lambda row: score_pair(row[wt_col], row[mut_col]), axis=1)
    return df

def _label_from_prediction(pred: str) -> int:
    """
    Map PolyPhen-2 textual prediction to a binary label:
      1 = (possibly / probably) damaging
      0 = benign or unknown.
    """
    if not isinstance(pred, str):
        return 0
    t = pred.strip().lower()
    if "benign" in t:
        return 0
    if "possibly damaging" in t or "probably damaging" in t or "damaging" in t:
        return 1
    return 0


def prepare_polyphen_df(path: Path) -> pd.DataFrame:
    """Load PolyPhen scores and normalize column names for merge."""
    df = pd.read_csv(path, sep="\t")

    df.columns = df.columns.str.strip()

    rename_map = {}

    # position
    if "position_1based" not in df.columns:
        if "pos" in df.columns:
            rename_map["pos"] = "position_1based"
        elif "position" in df.columns:
            rename_map["position"] = "position_1based"

    # wt aa
    if "wt_aa" not in df.columns:
        if "aa1" in df.columns:
            rename_map["aa1"] = "wt_aa"
        elif "wt" in df.columns:
            rename_map["wt"] = "wt_aa"

    # mut aa
    if "mut_aa" not in df.columns:
        if "aa2" in df.columns:
            rename_map["aa2"] = "mut_aa"
        elif "mut" in df.columns:
            rename_map["mut"] = "mut_aa"

    df = df.rename(columns=rename_map)

    if "polyphen_label" not in df.columns:
        if "prediction" in df.columns:
            df["polyphen_label"] = df["prediction"].apply(_label_from_prediction)
        elif "polyphen_prediction" in df.columns:
            df["polyphen_label"] = df["polyphen_prediction"].apply(_label_from_prediction)
        else:
            prob_col = None
            for cand in ["polyphen_score", "pph2_prob", "HumDiv", "humdiv_prob"]:
                if cand in df.columns:
                    prob_col = cand
                    break

            if prob_col is None:
                print(f"[DEBUG] Columns in {path}: {df.columns.tolist()}")
                raise ValueError(
                    f"PolyPhen file {path} has no 'polyphen_label', "
                    f"'prediction', 'polyphen_prediction' or known prob column; "
                    f"cannot build supervision labels."
                )

            df["polyphen_label"] = (df[prob_col] >= 0.85).astype(int)

    return df

def main():
    print("=== Step 1: Load ESM2 features ===")
    df_psen1_esm = pd.read_csv(PSEN1_ESM)
    df_app_esm   = pd.read_csv(APP_ESM)

    print(f"[PSEN1] ESM variants: {len(df_psen1_esm)}")
    print(f"[APP]   ESM variants: {len(df_app_esm)}")

    print("=== Step 2: Add BLOSUM62 scores ===")
    df_psen1_esm = add_blosum62(df_psen1_esm)
    df_app_esm   = add_blosum62(df_app_esm)

    print("=== Step 3: Load PolyPhen label tables ===")
    df_psen1_poly = prepare_polyphen_df(PSEN1_POLY)
    df_app_poly   = prepare_polyphen_df(APP_POLY)

    print(f"[PSEN1] PolyPhen-labeled mutations: {len(df_psen1_poly)}")
    print(f"[APP]   PolyPhen-labeled mutations: {len(df_app_poly)}")
    print("[DEBUG] PSEN1 columns:", df_psen1_poly.columns.tolist())



    print("=== Step 4: Train PSEN1 risk model ===")
    OUT_DIR_PSEN1.mkdir(parents=True, exist_ok=True)
    OUT_DIR_APP.mkdir(parents=True, exist_ok=True)

    model = train_risk_model(
        df_features=df_psen1_esm,
        df_labels=df_psen1_poly,
        feature_cols=["delta_loglik", "entropy", "blosum62"],
        test_size=0.2,
        random_state=42,
    )

    save_model(model, OUT_DIR_PSEN1 / "psen1_risk_scorer.joblib")

    print("=== Step 5: Score all PSEN1 variants ===")
    df_psen1_scored = score_variants(
        df_features=df_psen1_esm,
        model=model,
        feature_cols=["delta_loglik", "entropy", "blosum62"],
        output_col="risk_score",
    )
    df_psen1_scored.to_csv(OUT_DIR_PSEN1 / "psen1_scored_variants.csv", index=False)
    print(f"[PSEN1] Saved scored variants → {OUT_DIR_PSEN1 / 'psen1_scored_variants.csv'}")

    plot_path_psen1 = OUT_DIR_PSEN1 / "psen1_risk_dist.png"
    plot_risk_distribution(df_psen1_scored, score_col="risk_score", save_path=plot_path_psen1)

    print("=== Step 6: Apply PSEN1 model to APP variants (zero-shot) ===")
    df_app_scored = score_variants(
        df_features=df_app_esm,
        model=model,
        feature_cols=["delta_loglik", "entropy", "blosum62"],
        output_col="risk_score",
    )
    df_app_scored.to_csv(OUT_DIR_APP / "app_scored_variants_psen1_model.csv", index=False)
    print(f"[APP] Saved scored variants → {OUT_DIR_APP / 'app_scored_variants_psen1_model.csv'}")

    plot_path_app = OUT_DIR_APP / "app_risk_dist.png"
    plot_risk_distribution(df_app_scored, score_col="risk_score", save_path=plot_path_app)

    print("=== Done. ===")


if __name__ == "__main__":
    main()
