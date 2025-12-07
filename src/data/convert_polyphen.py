#!/usr/bin/env python3
"""
Convert mutation CSV into PolyPhen batch format:
    <UNIPROT_ID> <position> <wt_aa> <mut_aa>

Supports two input formats:
1. Raw: A column 'Mutation' with strings like "A79V"
2. ESM2: Columns 'position_1based', 'wt_aa', 'mut_aa'

Usage:
    # For Raw 'Mutation' column style
    python src/data/convert_polyphen.py --protein psen1 --input raw.csv --output out.txt --format raw

    # For ESM2 output style
    python src/data/convert_polyphen.py --protein app --input esm2.csv --output out.txt --format esm2
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
# Helper: Parse mutation string: A79V → (79, A, V)
# ============================================
def parse_mutation_string(mut_str: str):
    if not isinstance(mut_str, str) or len(mut_str) < 3:
        return None

    wt = mut_str[0]
    mut = mut_str[-1]
    pos = mut_str[1:-1]

    if not pos.isdigit():
        return None

    return int(pos), wt, mut


# ============================================
# Main conversion logic
# ============================================
def convert(input_file, output_file, protein_key, fmt="raw"):

    if protein_key not in UNIPROT_MAP:
        raise ValueError(f"Unknown protein: {protein_key}")

    uniprot_id = UNIPROT_MAP[protein_key]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading {input_file} in [{fmt}] mode...")
    df = pd.read_csv(input_file)
    
    mutations_out = []

    # -------------------------------------------------
    # MODE 1: ESM2 Output Format
    # Headers: protein, position_1based, wt_aa, mut_aa, ...
    # -------------------------------------------------
    if fmt == "esm2":
        required_cols = ["position_1based", "wt_aa", "mut_aa"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"ESM2 format requires columns: {required_cols}. Found: {df.columns.tolist()}")

        for _, row in df.iterrows():
            pos = int(row["position_1based"])
            wt = row["wt_aa"]
            mut = row["mut_aa"]

            # Filter out synonymous mutations (e.g. M -> M) to save PolyPhen resources
            if wt == mut:
                continue

            # Format: UNIPROT POS WT MUT
            line = f"{uniprot_id} {pos} {wt} {mut}"
            mutations_out.append(line)

    # -------------------------------------------------
    # MODE 2: Raw / Clinical Data Format
    # Headers: Mutation (e.g., "A79V")
    # -------------------------------------------------
    else:
        if "Mutation" not in df.columns:
            raise ValueError(f"Raw format requires column 'Mutation'. Found: {df.columns.tolist()}")

        for m in df["Mutation"]:
            parsed = parse_mutation_string(m)
            if parsed:
                pos, wt, mut = parsed
                # Ensure we don't write broken lines
                if wt == mut: 
                    continue
                
                line = f"{uniprot_id} {pos} {wt} {mut}"
                mutations_out.append(line)

    # -------------------------------------------------
    # Save Output
    # -------------------------------------------------
    if not mutations_out:
        print("Warning: No valid mutations found to convert.")
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for line in mutations_out:
            f.write(line + "\n")

    print(f"Saved {len(mutations_out)} mutations for [{protein_key.upper()}] to: {output_file}")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--protein", required=True, type=str.lower,
        help="Protein name: psen1 or app")

    parser.add_argument("--input", required=True,
        help="Input CSV file path")

    parser.add_argument("--output", required=True,
        help="Output PolyPhen batch file path")
    
    parser.add_argument("--format", type=str.lower, default="raw", choices=["raw", "esm2"],
        help="Input format: 'raw' (Mutation col) or 'esm2' (position_1based, wt_aa, mut_aa cols)")

    args = parser.parse_args()
    
    convert(args.input, args.output, args.protein, args.format)