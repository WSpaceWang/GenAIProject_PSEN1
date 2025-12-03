#!/usr/bin/env python3
"""
Convert raw mutation CSV (APP / PSEN1) into PolyPhen batch format:
    <UNIPROT_ID> <position> <wt_aa> <mut_aa>

Usage:
    python src/data/convert_polyphen.py \
        --protein psen1 \
        --input data/raw/psen1_mutations_raw.csv \
        --output data/processed/psen1_known_mutations_batch.txt
"""

import argparse
import os
import pandas as pd

# ============================================
# Protein → UniProt Mapping
# ============================================
UNIPROT_MAP = {
    "psen1": "PSN1_HUMAN",   # Presenilin 1
    "app":   "A4_HUMAN",     # APP (Amyloid precursor protein)
}


# ============================================
# Parse mutation string: A79V → (79, A, V)
# ============================================
def parse_mutation(mut_str: str):
    if not isinstance(mut_str, str) or len(mut_str) < 3:
        return None

    wt = mut_str[0]
    mut = mut_str[-1]
    pos = mut_str[1:-1]

    if not pos.isdigit():
        return None

    return int(pos), wt, mut


# ============================================
# Main conversion
# ============================================
def convert(input_file, output_file, protein_key):

    if protein_key not in UNIPROT_MAP:
        raise ValueError(f"Unknown protein: {protein_key}")

    uniprot_id = UNIPROT_MAP[protein_key]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    if "Mutation" not in df.columns:
        raise ValueError(f"CSV must contain column 'Mutation', found: {df.columns}")

    mutations_out = []

    for m in df["Mutation"]:
        parsed = parse_mutation(m)
        if parsed:
            pos, wt, mut = parsed
            line = f"{uniprot_id} {pos} {wt} {mut}"
            mutations_out.append(line)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write
    with open(output_file, "w") as f:
        for line in mutations_out:
            f.write(line + "\n")

    print(f"✅ Saved {len(mutations_out)} mutations → {output_file}")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--protein", required=True, type=str.lower,
        help="Protein name: psen1 or app")

    parser.add_argument("--input", required=True,
        help="Raw mutation CSV file path")

    parser.add_argument("--output", required=True,
        help="Path to save PolyPhen-style batch file")

    args = parser.parse_args()
    convert(args.input, args.output, args.protein)
