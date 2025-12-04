"""
Run ESM-2 (650M) on a single protein FASTA and perform Saturation Mutagenesis.
Calculates delta_loglik and entropy for ALL possible mutations at every position.

Usage:
    python -m scripts.run_esm2_saturation \
        --fasta data/raw/P49768.fasta \
        --protein PSEN1 \
        --output data/outputs/esm2/psen1_esm2_all.csv
"""

import argparse
import os
from typing import List

import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import esm

# Use the same model as before
MODEL_NAME = "esm2_t33_650M_UR50D"


def load_sequence(fasta_path: str) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")
    seq = str(records[0].seq)
    # Clean sequence
    seq = seq.replace(" ", "").replace("\n", "").upper()
    return seq


def compute_entropy(log_probs_pos: torch.Tensor) -> float:
    """
    Compute Shannon entropy for a position given log probabilities.
    H(x) = - sum(p(x) * log(p(x)))
    """
    probs = log_probs_pos.exp()
    entropy = -(probs * log_probs_pos).sum()
    return float(entropy.item())


def run_esm2_saturation(fasta_path: str, protein_name: str, output_csv: str) -> None:
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ESM] Using device: {device}")

    # 2. Load Sequence
    seq = load_sequence(fasta_path)
    seq_len = len(seq)
    print(f"[ESM] Loaded {protein_name} (Length: {seq_len})")

    # 3. Load Model
    print(f"[ESM] Loading model: {MODEL_NAME} ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    data = [(protein_name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 4. Inference (One pass for the whole sequence)
    print("[ESM] Running inference...")
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[33], return_contacts=False)
        logits = out["logits"]  # Shape: [1, L, V]
        log_probs = torch.log_softmax(logits, dim=-1)  # Shape: [1, L, V]

    # Remove batch dimension
    log_probs = log_probs[0]  # Shape: [L, V]
    tokens = batch_tokens[0]  # Shape: [L]
    aa_to_idx = alphabet.tok_to_idx

    # Get standard amino acids (ignore tokens like <cls>, <pad>, etc.)
    standard_aas = alphabet.standard_toks # ['L', 'A', 'G', 'V', ...]
    standard_indices = [aa_to_idx[aa] for aa in standard_aas]

    records: List[dict] = []

    # 5. Iterate through sequence (skip <cls> at 0 and <eos> at end)
    # tokens sequence includes start/end tokens, so actual seq is 1..L-1
    for pos in range(1, seq_len + 1):
        token_idx = tokens[pos].item()
        token_str = alphabet.get_tok(token_idx)

        # Skip non-standard tokens if any appear in middle of seq
        if token_str not in standard_aas:
            continue

        wt_aa = token_str
        
        # Get log-probs for this position
        lp_pos = log_probs[pos]
        wt_logp = lp_pos[token_idx].item()
        
        # Compute Entropy for this position
        entropy = compute_entropy(lp_pos)

        # Loop through ALL 20 standard amino acids (Saturation Mutagenesis)
        for mut_aa in standard_aas:
            if mut_aa == wt_aa:
                continue # Skip Wild Type (optional, usually we want variants only)

            mut_idx = aa_to_idx[mut_aa]
            mut_logp = lp_pos[mut_idx].item()
            
            # Delta Log Likelihood: log(P(mut)) - log(P(wt))
            # Negative values = Mutation is less likely than WT (deleterious?)
            delta_ll = mut_logp - wt_logp

            records.append({
                "protein": protein_name,
                "position_1based": pos, # aligns with standard biology numbering
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "wt_logp": wt_logp,
                "mut_logp": mut_logp,
                "delta_loglik": delta_ll,
                "entropy": entropy
            })

    # 6. Save
    df = pd.DataFrame(records)
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"[ESM] Done. Generated {len(df)} variants.")
    print(f"[ESM] Saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Path to input FASTA")
    parser.add_argument("--protein", required=True, help="Protein name (e.g. PSEN1)")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    run_esm2_saturation(args.fasta, args.protein, args.output)


if __name__ == "__main__":
    main()