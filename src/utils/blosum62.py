"""
BLOSUM62 substitution matrix utilities.
"""
from Bio.Align import substitution_matrices


def get_blosum62_score(wt_aa: str, mut_aa: str) -> int:
    """
    Get BLOSUM62 substitution score for a wild-type to mutant amino acid pair.
    
    Args:
        wt_aa: Wild-type amino acid (one-letter code)
        mut_aa: Mutant amino acid (one-letter code)
    
    Returns:
        BLOSUM62 score (integer)
    """
    mat = substitution_matrices.load("BLOSUM62")
    wt_aa = wt_aa.strip()
    mut_aa = mut_aa.strip()
    pair = (wt_aa, mut_aa)
    rev = (mut_aa, wt_aa)
    
    if pair in mat:
        return mat[pair]
    if rev in mat:
        return mat[rev]
    # If pair is not present (rare), assign neutral 0
    return 0


def add_blosum62_scores(df, wt_col='wt_aa', mut_col='mut_aa', output_col='blosum62'):
    """
    Add BLOSUM62 scores to a DataFrame.
    
    Args:
        df: DataFrame with wild-type and mutant amino acid columns
        wt_col: Name of wild-type amino acid column
        mut_col: Name of mutant amino acid column
        output_col: Name of output column for BLOSUM62 scores
    
    Returns:
        DataFrame with added BLOSUM62 scores
    """
    df = df.copy()
    df[output_col] = df.apply(
        lambda row: get_blosum62_score(row[wt_col], row[mut_col]),
        axis=1
    )
    return df

