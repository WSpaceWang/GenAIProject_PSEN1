# PSEN1 Mutation Risk Prediction Pipeline

A lightweight computational pipeline for predicting pathogenic mutations in Presenilin-1 (PSEN1) and APP using ESM-2 protein language model, evolutionary features (BLOSUM62), and PolyPhen-2 labels.

## Overview

This project addresses the challenge of predicting pathogenic mutations with limited labeled data. The pipeline combines:

- **ESM-2**: Pre-trained protein language model for mutation likelihood estimation (saturation mutagenesis)
- **BLOSUM62**: Evolutionary substitution matrix
- **PolyPhen-2**: Weak supervision labels for training
- **Logistic Regression**: Lightweight, interpretable risk prediction model

## Project Structure

```
GenAIProject_PSEN1/
├── src/
│   ├── data/              # Data processing modules
│   │   ├── convert_polyphen.py   # Mutation format conversion for PolyPhen-2
│   │   └── tidy.py               # PolyPhen output tidying
│   ├── models/            # Model training and scoring
│   │   └── risk_model.py         # Risk prediction model (Logistic Regression)
│   └── visualization/     # Visualization tools
│       └── visualize_polyphen.py  # PolyPhen distribution plots
├── scripts/               # Main execution scripts
│   ├── run_esm2.py              # ESM-2 saturation mutagenesis
│   └── run_risk_models.py        # Train model and score variants
├── data/
│   ├── raw/               # Raw input data
│   │   ├── psen1_mutations_raw.csv
│   │   ├── app_mutations_raw.csv
│   │   ├── P49768.fasta (PSEN1)
│   │   └── P05067.fasta (APP)
│   ├── processed/         # Processed intermediate data
│   │   ├── polyphen2/     # PolyPhen-2 outputs
│   │   └── tidy/          # Tidy PolyPhen data
│   └── outputs/          # Results and model outputs
│       ├── esm2/          # ESM-2 generated mutations
│       ├── psen1/         # PSEN1 results
│       └── app/           # APP results
└── data_crawler.py        # Alzforum data crawler
```

## Workflow

### 1. Data Collection (Optional)

Crawl mutation data from Alzforum:

```bash
cd GenAIProject_PSEN1
python data/data_crawler.py --protein psen1
python data/data_crawler.py --protein app
```

### 2. Convert Mutations for PolyPhen-2

Convert mutations to PolyPhen-2 batch input format:

```bash
python src/data/convert_polyphen.py \
    --protein psen1 \
    --input data/raw/psen1_mutations_raw.csv \
    --output data/processed/psen1_known_mutations_batch.txt
```

### 3. Generate Mutations with ESM-2

Run ESM-2 saturation mutagenesis (all possible mutations):

```bash
python scripts/run_esm2.py \
    --fasta data/raw/P49768.fasta \
    --protein PSEN1 \
    --output data/outputs/esm2/psen1_esm2_all.csv
```

This generates **all possible mutations** at every position with:
- `delta_loglik`: Change in log-likelihood (mut_logp - wt_logp)
- `entropy`: Position uncertainty
- `wt_logp`, `mut_logp`: Wild-type and mutant log-probabilities

### 4. Train Model and Score Variants

Train risk model and score all variants:

```bash
python scripts/run_risk_models.py
```

This script:
1. Loads ESM-2 generated mutations (PSEN1 and APP)
2. Adds BLOSUM62 scores
3. Loads PolyPhen-2 labels
4. Trains logistic regression model on PSEN1 data
5. Scores all PSEN1 variants
6. Applies PSEN1 model to APP variants (zero-shot transfer)
7. Generates risk distribution plots

**Outputs:**
- `data/outputs/psen1/psen1_risk_scorer.joblib`: Trained model
- `data/outputs/psen1/psen1_scored_variants.csv`: All PSEN1 variants with risk scores
- `data/outputs/app/app_scored_variants_psen1_model.csv`: APP variants scored with PSEN1 model
- `data/outputs/*/psen1_risk_dist.png`: Risk score distributions

### 5. Tidy PolyPhen Data (Optional)

Convert PolyPhen output to tidy format:

```bash
python src/data/tidy.py \
    --input data/processed/polyphen2/psen1_polyphen_scores.txt \
    --output data/processed/tidy/psen1_polyphen_tidy.csv
```

### 6. Visualization

Visualize PolyPhen-2 prediction distribution:

```bash
python src/visualization/visualize_polyphen.py \
    --input data/processed/polyphen2/psen1_polyphen_scores.txt \
    --output data/outputs/psen1/psen1_polyphen_distribution.png \
    --protein PSEN1
```

## Key Features

### Mutation Generation
- **Saturation Mutagenesis**: Generates all 19 possible mutations per position
- Uses ESM-2 (650M parameters) to compute mutation likelihoods
- Extracts position-level entropy for uncertainty quantification

### Risk Scoring
- **Features**: `delta_loglik`, `entropy`, `blosum62`
- **Model**: Logistic Regression (lightweight, interpretable)
- **Training**: Weak supervision from PolyPhen-2 labels
- **Output**: Risk scores (0-1) for all candidate mutations
- **Zero-shot Transfer**: PSEN1-trained model applied to APP

### Validation
- PolyPhen-2 predictions as weak supervision
- Known vs. novel variant separation
- Risk distribution visualization

## Requirements

See `requirements.txt` for full dependencies. Key packages:

- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning
- `biopython`: Sequence analysis and BLOSUM62
- `torch`, `fair-esm`: ESM-2 model
- `matplotlib`, `seaborn`: Visualization
- `requests`, `beautifulsoup4`: Web scraping (for data crawler)

Install with:

```bash
pip install -r requirements.txt
```

## Data Requirements

1. **Raw mutation data**: 
   - `data/raw/psen1_mutations_raw.csv`
   - `data/raw/app_mutations_raw.csv`
2. **Protein sequences**: 
   - `data/raw/P49768.fasta` (PSEN1, UniProt format)
   - `data/raw/P05067.fasta` (APP, UniProt format)
3. **PolyPhen-2 outputs**: 
   - `data/processed/polyphen2/psen1_polyphen_scores.txt`
   - `data/processed/polyphen2/app_polyphen_scores.txt`

## Results

The pipeline generates:
- Risk scores for all candidate mutations (saturation mutagenesis)
- Separation of known vs. novel variants
- High-risk variant prioritization
- Model performance metrics (AUC, accuracy)
- Zero-shot transfer results (PSEN1 model → APP)

## Supported Proteins

- **PSEN1** (Presenilin-1): Primary training target
- **APP** (Amyloid Precursor Protein): Zero-shot transfer target

## Future Extensions

- Support for additional proteins
- AlphaFold structure-based filtering
- Ensemble methods for improved predictions
- Web interface for variant prioritization
- Cross-protein model transfer analysis

## Citation

If you use this pipeline, please cite:
- ESM-2: [Lin et al., 2022](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- PolyPhen-2: [Adzhubei et al., 2010](https://www.nature.com/articles/nmeth0410-248)
