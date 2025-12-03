#!/usr/bin/env python

"""
End-to-end analysis pipeline for PSEN1 variants:

Input:
  1) ESM-2 mutation proposals with per-mutation features:
       - /jet/home/xwang54/esm2/logs/res/psen1_esm2_top3.csv
     This file should contain columns:
       position_1based, wt_aa, mut_aa, delta_loglik, entropy, ...

  2) PolyPhen-2 HumDiv batch output:
       - /jet/home/xwang54/esm2/dataset/known_mutations_score.txt
     This is the raw tab-separated text file from PolyPhen-2.

Steps in this script:
  - Parse PolyPhen-2 output into a clean label table.
  - Add BLOSUM62 substitution scores to all ESM-2 variants.
  - Join labeled variants and train a logistic regression risk model.
  - Apply this model to score ALL ESM-2-generated variants.
  - Save:
      - labeled training set with features
      - final scored variants table
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.polyphen_parser import load_polyphen_labels
from src.utils.blosum62 import add_blosum62_scores
from src.models.risk_scorer import train_risk_model, score_variants, save_model


# ---------------------------------------------------------------------------
# Paths (edit here if you change layout)
# ---------------------------------------------------------------------------

ESM_CSV = Path("data/processed/psen1_esm2_top3.csv")
POLYPHEN_TXT = Path("data/external/known_mutations_score.txt")
OUT_DIR = Path("outputs/psen1")

LABELED_CSV = OUT_DIR / "psen1_training_labeled.csv"
MODEL_PATH = OUT_DIR / "psen1_risk_scorer.joblib"
SCORED_CSV = OUT_DIR / "psen1_scored_variants.csv"

def mark_known_variants(df_scored: pd.DataFrame,
                        df_poly: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column 'is_known' to the scored variants table.

    A variant is considered "known" if it appears in the PolyPhen-2
    batch file (same position_1based, wt_aa, mut_aa).
    """
    # Build a set of known (position, wt, mut) triples
    known_keys = set(
        zip(
            df_poly["position_1based"].astype(int),
            df_poly["wt_aa"].astype(str).str.strip(),
            df_poly["mut_aa"].astype(str).str.strip(),
        )
    )

    def is_known(row) -> bool:
        key = (
            int(row["position_1based"]),
            str(row["wt_aa"]).strip(),
            str(row["mut_aa"]).strip(),
        )
        return key in known_keys

    df_scored = df_scored.copy()
    df_scored["is_known"] = df_scored.apply(is_known, axis=1)
    return df_scored


def save_training_data(df_train: pd.DataFrame, output_path: Path):
    """Save labeled training data for inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_path, index=False)
    print(f"[TRAIN] Saved labeled training table to: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=== PSEN1 ESM-2 + PolyPhen analysis pipeline ===")

    # Load ESM-2 top-k mutation proposals
    if not ESM_CSV.exists():
        raise FileNotFoundError(f"ESM CSV not found: {ESM_CSV}")
    df_esm = pd.read_csv(ESM_CSV)
    print(f"[ESM] Loaded {len(df_esm)} ESM-generated variants from: {ESM_CSV}")

    # Add BLOSUM62 scores to all ESM variants
    df_esm = add_blosum62_scores(df_esm)
    print("[ESM] Added BLOSUM62 scores.")

    # Load PolyPhen-2 labels
    df_poly = load_polyphen_labels(POLYPHEN_TXT)
    print(f"[PolyPhen] Loaded {len(df_poly)} labeled mutations from: {POLYPHEN_TXT}")

    # Prepare training data (join features and labels)
    df_train = df_esm.merge(
        df_poly,
        on=["position_1based", "wt_aa", "mut_aa"],
        how="inner"
    )
    save_training_data(df_train, LABELED_CSV)

    # Train logistic regression risk model
    model = train_risk_model(df_esm, df_poly)

    # Save model
    save_model(model, MODEL_PATH)

    # Score all ESM-generated variants (including those without PolyPhen labels)
    df_scored = score_variants(df_esm, model)

    # Mark known vs novel variants
    df_scored = mark_known_variants(df_scored, df_poly)

    # Sort by risk_score descending for convenience
    df_scored = df_scored.sort_values("risk_score", ascending=False)

    # Save final table
    df_scored.to_csv(SCORED_CSV, index=False)
    print(f"[SCORE] Saved scored variants to: {SCORED_CSV}")
    print(f"[SCORE] Total variants scored: {len(df_scored)}")

    # ---------------- Summary statistics for the paper ----------------
    total = len(df_scored)
    known = int(df_scored["is_known"].sum())
    novel = total - known

    high_thr = 0.9  
    df_high = df_scored[df_scored["risk_score"] >= high_thr]
    high_total = len(df_high)
    high_known = int(df_high["is_known"].sum())
    high_novel = high_total - high_known

    print("\n[SUMMARY] Variant counts")
    print(f"  Total variants: {total}")
    print(f"    Known variants: {known}")
    print(f"    Novel variants: {novel}")

    print(f"\n[SUMMARY] High-risk variants (risk_score >= {high_thr})")
    print(f"  High-risk total: {high_total}")
    print(f"    Known high-risk: {high_known}")
    print(f"    Novel high-risk: {high_novel}")

    # Print top 5 variants as a quick sanity check
    print("\n[TOP 5] Highest-risk variants:")
    print(df_scored[[
        "position_1based", "wt_aa", "mut_aa",
        "delta_loglik", "entropy", "blosum62", "risk_score", "is_known"
    ]].head(5))


if __name__ == "__main__":
    main()
