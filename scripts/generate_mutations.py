#!/usr/bin/env python
import math
import os
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import esm


FASTA_PATH = "data/raw/P49768.fasta"     
OUTPUT_CSV = "data/processed/psen1_esm2_top3.csv"

MODEL_NAME = "esm2_t33_650M_UR50D"   # 650M 版本

def load_ps1_sequence(fasta_path: str) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")
    seq = str(records[0].seq)
    seq = seq.replace(" ", "").replace("\n", "").upper()
    return seq


def compute_entropy(log_probs: torch.Tensor) -> float:
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    return float(entropy.item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seq = load_ps1_sequence(FASTA_PATH)
    print(f"Loaded PS1 sequence length: {len(seq)}")

    print(f"Loading model: {MODEL_NAME} ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    data = [("PSEN1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[33], return_contacts=False)
        logits = out["logits"]  # [1, L, V]
        log_probs = torch.log_softmax(logits, dim=-1)  # [1, L, V]

    log_probs = log_probs[0]        # [L, V]
    tokens = batch_tokens[0]        # [L]
    aa_to_idx = alphabet.tok_to_idx

    records: List[dict] = []

    L = log_probs.size(0)

    for pos in range(1, L - 1):
        token_idx = tokens[pos].item()
        token_str = alphabet.get_tok(token_idx)

        if token_str not in alphabet.standard_toks:
            continue

        wt_aa = token_str
        wt_idx = token_idx

        lp_pos = log_probs[pos]  # [V]
        wt_logp = lp_pos[wt_idx].item()

        entropy = compute_entropy(lp_pos)

        standard_indices = [aa_to_idx[a] for a in alphabet.standard_toks]
        standard_lp = lp_pos[standard_indices]  # [20]
        top_vals, top_indices = torch.topk(standard_lp, k=4, dim=-1)

        candidates: List[Tuple[str, float, float]] = []  # (aa, logp, delta)
        for v, idx_in_standard in zip(top_vals, top_indices):
            global_idx = standard_indices[idx_in_standard.item()]
            aa = alphabet.get_tok(global_idx)
            if aa == wt_aa:
                continue
            logp = v.item()
            delta = logp - wt_logp  
            candidates.append((aa, logp, delta))

        candidates = candidates[:3]

        if len(candidates) == 0:
            continue

        for rank, (mut_aa, mut_logp, delta_ll) in enumerate(candidates, start=1):
            records.append(
                {
                    "position_1based": pos,  # 注意：这还是 token index，不一定完全等于结构编号
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "rank": rank,
                    "wt_logp": wt_logp,
                    "mut_logp": mut_logp,
                    "delta_loglik": delta_ll,
                    "entropy": entropy,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} mutation candidates to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
