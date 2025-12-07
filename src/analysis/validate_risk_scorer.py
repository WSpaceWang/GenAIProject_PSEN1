import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os
import argparse
import joblib
import numpy as np
from Bio.Align import substitution_matrices

# ==========================================
# 1. Feature Engineering (BLOSUM62)
# ==========================================
def add_blosum62(df: pd.DataFrame, wt_col: str = "wt_aa", mut_col: str = "mut_aa") -> pd.DataFrame:
    """
    Calculates BLOSUM62 score for WT -> Mut pair.
    Required because the trained model expects this feature.
    """
    print("   -> Calculating BLOSUM62 scores...")
    try:
        mat = substitution_matrices.load("BLOSUM62")
    except Exception:
        # Fallback if Biopython not configured usually, but generic load should work
        print("Warning: Could not load generic BLOSUM62, trying internal dict fallback if needed.")
        return df

    def score_pair(wt, mut):
        wt = str(wt).strip().upper()
        mut = str(mut).strip().upper()
        # Biopython substitution matrices use tuples like ('A', 'V')
        if (wt, mut) in mat:
            return mat[(wt, mut)]
        elif (mut, wt) in mat: # Symmetric
            return mat[(mut, wt)]
        return 0 # Default/Error

    df = df.copy()
    df['blosum62'] = df.apply(lambda row: score_pair(row[wt_col], row[mut_col]), axis=1)
    return df

# ==========================================
# 2. Data Loading
# ==========================================
def load_polyphen(filepath):
    """Robust PolyPhen loader."""
    print(f"Loading PolyPhen: {filepath}")
    try:
        # Try Tab first
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
        if len(df.columns) < 5:
            df = pd.read_csv(filepath, sep='\s+', on_bad_lines='skip')
        
        df.columns = [c.replace('#', '').strip() for c in df.columns]
        
        # Determine position column
        if 'o_pos' in df.columns:
            pos_col = 'o_pos'
        elif 'pos' in df.columns:
            pos_col = 'pos'
        else:
            return None

        # Clean types for merging
        df['pos_str'] = pd.to_numeric(df[pos_col], errors='coerce').fillna(0).astype(int).astype(str)
        
        # Clean AA
        aa1 = 'o_aa1' if 'o_aa1' in df.columns else 'aa1'
        aa2 = 'o_aa2' if 'o_aa2' in df.columns else 'aa2'
        
        df['aa1_clean'] = df[aa1].astype(str).str.strip().str.upper()
        df['aa2_clean'] = df[aa2].astype(str).str.strip().str.upper()
        
        df['merge_key'] = df['pos_str'] + '_' + df['aa1_clean'] + '_' + df['aa2_clean']
        return df
    except Exception as e:
        print(f"❌ Error loading PolyPhen: {e}")
        return None

def load_esm2_features(filepath):
    print(f"Loading ESM2 features: {filepath}")
    df = pd.read_csv(filepath)
    
    # Clean types
    df['pos_str'] = pd.to_numeric(df['position_1based'], errors='coerce').fillna(0).astype(int).astype(str)
    df['wt_clean'] = df['wt_aa'].astype(str).str.strip().str.upper()
    df['mut_clean'] = df['mut_aa'].astype(str).str.strip().str.upper()
    
    df['merge_key'] = df['pos_str'] + '_' + df['wt_clean'] + '_' + df['mut_clean']
    
    # Add BLOSUM62 feature!
    df = add_blosum62(df, wt_col='wt_clean', mut_col='mut_clean')
    
    return df

# ==========================================
# 3. Main Logic
# ==========================================
def run_validation(model_path, esm2_path, polyphen_path, output_dir):
    
    # 1. Load Data
    esm_df = load_esm2_features(esm2_path)
    pp_df = load_polyphen(polyphen_path)
    
    if esm_df is None or pp_df is None:
        print("Data load failed.")
        return

    # 2. Load Model
    print(f"Loading Model: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Predict
    # The model expects these exact columns
    features = ["delta_loglik", "entropy", "blosum62"]
    
    # Verify columns exist
    missing = [c for c in features if c not in esm_df.columns]
    if missing:
        print(f"❌ Missing features in ESM2 file: {missing}")
        return

    print(f"Predicting scores using features: {features}")
    # predict_proba returns [prob_class_0, prob_class_1]. We want class 1 (Pathogenic)
    try:
        # Check if model is classifier or regressor
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(esm_df[features].fillna(0))[:, 1]
        else:
            preds = model.predict(esm_df[features].fillna(0))
    except Exception as e:
        print(f"❌ Prediction runtime error: {e}")
        return

    esm_df['my_model_score'] = preds

    # 4. Merge with PolyPhen
    merged = pd.merge(esm_df, pp_df, on='merge_key', how='inner')
    print(f"Matched {len(merged)} mutations.")

    if len(merged) < 50:
        print("⚠️ Too few matches to plot.")
        return

    # 5. Correlation & Plotting
    clean = merged.dropna(subset=['pph2_prob', 'my_model_score'])
    
    p_corr, _ = pearsonr(clean['pph2_prob'], clean['my_model_score'])
    s_corr, _ = spearmanr(clean['pph2_prob'], clean['my_model_score'])
    
    print(f"\n📊 Correlation Results (N={len(clean)}):")
    print(f"   Pearson  r: {p_corr:.4f}")
    print(f"   Spearman r: {s_corr:.4f}")
    
    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    sns.regplot(
        data=clean,
        x='pph2_prob',
        y='my_model_score',
        scatter_kws={'alpha': 0.1, 's': 10, 'color': '#8E44AD'}, 
        line_kws={'color': '#2C3E50', 'linewidth': 2.5}
    )
    
    plt.title('Validation: Custom Risk Scorer vs PolyPhen-2', fontsize=16)
    plt.xlabel('PolyPhen-2 Probability (0=Benign, 1=Damaging)', fontsize=12)
    plt.ylabel('My Model Predicted Probability', fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05) # Since both are probabilities 0-1
    
    stats_text = f"Pearson r = {p_corr:.3f}\nSpearman r = {s_corr:.3f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    
    out_file = os.path.join(output_dir, 'validation_risk_scorer_vs_polyphen.png')
    plt.savefig(out_file, dpi=300)
    print(f"✅ Plot saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--esm2', required=True)
    parser.add_argument('--polyphen', required=True)
    parser.add_argument('--outdir', required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    run_validation(args.model, args.esm2, args.polyphen, args.outdir)