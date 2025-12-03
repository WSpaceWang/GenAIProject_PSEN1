# Project Structure Documentation

## Overview

This document describes the reorganized structure of the PSEN1 mutation risk prediction project, which combines code from `GenAIProject_PSEN1` and `esm2` folders into a unified, well-organized pipeline.

## Directory Structure

```
GenAIProject_PSEN1/
├── config.py                    # Central configuration file
├── requirements.txt             # Python dependencies
├── README.md                   # Main project documentation
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Raw data cleaning (from data_preprocessing.py)
│   │   ├── convert_polyphen.py # Mutation format conversion
│   │   └── polyphen_parser.py  # PolyPhen-2 output parser (from analyze.py)
│   │
│   ├── models/                 # Model training and scoring
│   │   ├── __init__.py
│   │   └── risk_scorer.py     # Logistic regression model (from analyze.py)
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── blosum62.py         # BLOSUM62 scoring (from analyze.py)
│   │
│   └── visualization/          # Visualization tools
│       ├── __init__.py
│       └── visualize_polyphen.py # PolyPhen distribution plots
│
├── scripts/                     # Main execution scripts
│   ├── main.py                 # Main entry point (orchestrates pipeline)
│   ├── generate_mutations.py   # ESM-2 mutation generation (from run_esm2.py)
│   ├── run_analysis.py         # End-to-end analysis (from analyze.py)
│   └── analyze_blosum.py       # BLOSUM62 analysis (from blosum62.py)
│
├── data/                        # Data files
│   ├── raw/                     # Raw input data
│   │   ├── psen1_mutations_raw.csv
│   │   └── P49768.fasta
│   ├── processed/               # Processed intermediate data
│   │   ├── psen1_mutations_tidy.csv
│   │   ├── known_mutations_batch.txt
│   │   └── psen1_esm2_top3.csv
│   └── external/                # External data (PolyPhen outputs)
│       └── known_mutations_score.txt
│
├── outputs/                     # Results and model outputs
│   └── psen1/                   # PSEN1-specific results
│       ├── psen1_training_labeled.csv
│       ├── psen1_risk_scorer.joblib
│       ├── psen1_scored_variants.csv
│       └── known_mutations_prediction_distribution.png
│
└── logs/                        # Log files and model checkpoints
```

## File Migration Summary

### From `GenAIProject_PSEN1/`:
- `data_preprocessing.py` → `src/data/preprocessing.py`
- `convert_mutations_polyphen2.py` → `src/data/convert_polyphen.py`
- `blosum62.py` → `scripts/analyze_blosum.py`
- `visualize_polyphen.py` → `src/visualization/visualize_polyphen.py`
- `dataset/` → `data/` (reorganized into raw/processed/external)

### From `esm2/`:
- `run_esm2.py` → `scripts/generate_mutations.py`
- `analyze.py` → Split into:
  - `src/data/polyphen_parser.py` (PolyPhen parsing)
  - `src/models/risk_scorer.py` (model training)
  - `src/utils/blosum62.py` (BLOSUM62 utilities)
  - `scripts/run_analysis.py` (main analysis pipeline)
- `dataset/P49768.fasta` → `data/raw/P49768.fasta`
- `dataset/known_mutations_score.txt` → `data/external/known_mutations_score.txt`
- `logs/res/psen1_esm2_top3.csv` → `data/processed/psen1_esm2_top3.csv`

## Key Improvements

1. **Modular Structure**: Code is organized into logical modules (data, models, utils, visualization)
2. **Separation of Concerns**: Each module has a clear responsibility
3. **Reusability**: Utility functions can be easily imported and reused
4. **Configuration**: Centralized config file for easy path management
5. **Documentation**: Comprehensive README and structure documentation
6. **Extensibility**: Easy to add support for other proteins (e.g., APP)

## Usage

### Run individual steps:
```bash
python scripts/main.py preprocess
python scripts/main.py convert
python scripts/main.py generate
python scripts/main.py analyze
python scripts/main.py visualize
```

### Run full pipeline:
```bash
python scripts/main.py all
```

### Or run scripts directly:
```bash
python -m src.data.preprocessing
python scripts/generate_mutations.py
python scripts/run_analysis.py
```

## Next Steps

- [ ] Add support for APP protein analysis
- [ ] Integrate AlphaFold structure filtering
- [ ] Add unit tests
- [ ] Create web interface for variant prioritization
- [ ] Add ensemble methods for improved predictions

