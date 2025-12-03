# PSEN1 Mutation Risk Prediction Pipeline

A lightweight computational pipeline for predicting pathogenic mutations in Presenilin-1 (PSEN1) using ESM-2 protein language model, evolutionary features (BLOSUM62), and PolyPhen-2 labels.

## Overview

This project addresses the challenge of predicting pathogenic PSEN1 mutations with limited labeled data (only ~369 known mutations). The pipeline combines:

- **ESM-2**: Pre-trained protein language model for mutation likelihood estimation
- **BLOSUM62**: Evolutionary substitution matrix
- **PolyPhen-2**: Weak supervision labels for training
- **Logistic Regression**: Lightweight, interpretable risk prediction model

## Project Structure

```
GenAIProject_PSEN1/
├── src/
│   ├── data/              # Data processing modules
│   │   ├── preprocessing.py      # Raw data cleaning
│   │   ├── convert_polyphen.py   # Mutation format conversion
│   │   └── polyphen_parser.py    # PolyPhen-2 output parser
│   ├── models/            # Model training and scoring
│   │   └── risk_scorer.py        # Logistic regression risk model
│   ├── utils/             # Utility functions
│   │   └── blosum62.py           # BLOSUM62 scoring utilities
│   └── visualization/     # Visualization tools
│       └── visualize_polyphen.py  # PolyPhen distribution plots
├── scripts/               # Main execution scripts
│   ├── generate_mutations.py     # ESM-2 mutation generation
│   ├── run_analysis.py           # End-to-end analysis pipeline
│   └── analyze_blosum.py         # BLOSUM62 analysis
├── data/
│   ├── raw/               # Raw input data
│   ├── processed/         # Processed intermediate data
│   └── external/          # External data (PolyPhen outputs)
├── outputs/               # Results and model outputs
│   └── psen1/            # PSEN1-specific results
└── logs/                  # Log files and model checkpoints
```

## Workflow

### 1. Data Preprocessing

Clean and standardize raw PSEN1 mutation data:

```bash
cd GenAIProject_PSEN1
python -m src.data.preprocessing
```

This processes `data/raw/psen1_mutations_raw.csv` and outputs `data/processed/psen1_mutations_tidy.csv`.

### 2. Convert Mutations for PolyPhen-2

Convert mutations to PolyPhen-2 batch input format:

```bash
python -m src.data.convert_polyphen
```

Outputs `data/processed/known_mutations_batch.txt` for PolyPhen-2 submission.

### 3. Generate Mutations with ESM-2

Use ESM-2 to generate top-3 plausible mutations per position:

```bash
python scripts/generate_mutations.py
```

Requires:
- `data/raw/P49768.fasta` (PSEN1 sequence)
- ESM-2 model (downloaded automatically)

Outputs `data/processed/psen1_esm2_top3.csv` with features:
- `delta_loglik`: Change in log-likelihood
- `entropy`: Position uncertainty
- `wt_logp`, `mut_logp`: Wild-type and mutant log-probabilities

### 4. Run Full Analysis Pipeline

Train risk model and score all variants:

```bash
python scripts/run_analysis.py
```

This script:
1. Loads ESM-2 generated mutations
2. Adds BLOSUM62 scores
3. Loads PolyPhen-2 labels
4. Trains logistic regression model
5. Scores all variants
6. Saves results to `outputs/psen1/`

Outputs:
- `psen1_training_labeled.csv`: Training data with features and labels
- `psen1_risk_scorer.joblib`: Trained model
- `psen1_scored_variants.csv`: All variants with risk scores

### 5. Visualization

Visualize PolyPhen-2 prediction distribution:

```bash
python -m src.visualization.visualize_polyphen
```

## Key Features

### Mutation Generation
- Uses ESM-2 (650M parameters) to compute mutation likelihoods
- Generates top-3 most plausible mutations per position
- Extracts position-level entropy for uncertainty quantification

### Risk Scoring
- **Features**: `delta_loglik`, `entropy`, `blosum62`
- **Model**: Logistic Regression (lightweight, interpretable)
- **Training**: Weak supervision from PolyPhen-2 labels
- **Output**: Risk scores (0-1) for all candidate mutations

### Validation
- PolyPhen-2 predictions as weak supervision
- AlphaFold structure filtering (future work)
- Known vs. novel variant separation

## Requirements

See `requirements.txt` for full dependencies. Key packages:

- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning
- `biopython`: Sequence analysis and BLOSUM62
- `torch`, `fair-esm`: ESM-2 model
- `matplotlib`, `seaborn`: Visualization

Install with:

```bash
pip install -r requirements.txt
```

## Data Requirements

1. **Raw mutation data**: `data/raw/psen1_mutations_raw.csv`
2. **PSEN1 sequence**: `data/raw/P49768.fasta` (UniProt format)
3. **PolyPhen-2 output**: `data/external/known_mutations_score.txt`

## Results

The pipeline generates:
- Risk scores for all candidate mutations
- Separation of known vs. novel variants
- High-risk variant prioritization
- Model performance metrics (AUC, accuracy)

## Future Extensions

- Support for additional proteins (e.g., APP)
- AlphaFold structure-based filtering
- Ensemble methods for improved predictions
- Web interface for variant prioritization

## Citation

If you use this pipeline, please cite:
- ESM-2: [Lin et al., 2022](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- PolyPhen-2: [Adzhubei et al., 2010](https://www.nature.com/articles/nmeth0410-248)
