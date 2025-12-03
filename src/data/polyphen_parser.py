"""
Parser for PolyPhen-2 batch output files.
"""
import pandas as pd
from pathlib import Path


def label_from_prediction(pred: str) -> int:
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


def load_polyphen_labels(polyphen_path: Path) -> pd.DataFrame:
    """
    Load raw PolyPhen-2 batch output (TAB-separated, no header)
    and convert it into a compact label DataFrame.

    Expected 13 columns per row:
      0: protein ID
      1: position (integer, 1-based)
      2: wt_aa (one-letter)
      3: mut_aa (one-letter)
      4: unknown flag
      5: UniProt ID
      6: position (again or float)
      7: wt_aa again
      8: mut_aa again
      9: prediction text ("probably damaging", "benign", etc.)
     10: prediction class code
     11: HumDiv probability score
     12: HumVar probability score
    
    Returns:
        DataFrame with columns: position_1based, wt_aa, mut_aa, 
        polyphen_prediction, polyphen_score, polyphen_label
    """
    if not polyphen_path.exists():
        raise FileNotFoundError(f"PolyPhen file not found: {polyphen_path}")

    df = pd.read_csv(
        polyphen_path,
        sep="\t",
        header=None,
        comment="#"
    )

    if df.shape[1] < 13:
        raise ValueError(
            f"Expected >= 13 columns in PolyPhen file, got {df.shape[1]}.\n"
            f"Please open the file and verify format."
        )

    out = pd.DataFrame()
    out["position_1based"] = df[1].astype(int)
    out["wt_aa"] = df[2].astype(str).str.strip()
    out["mut_aa"] = df[3].astype(str).str.strip()
    out["polyphen_prediction"] = df[9].astype(str).str.strip()
    out["polyphen_score"] = df[11].astype(float)  # HumDiv score

    out["polyphen_label"] = out["polyphen_prediction"].apply(label_from_prediction)
    return out

