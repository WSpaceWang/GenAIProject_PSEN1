"""
Configuration file for PSEN1 mutation risk prediction pipeline.

Edit these paths and parameters as needed for your setup.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Input files
RAW_MUTATIONS_CSV = RAW_DATA_DIR / "psen1_mutations_raw.csv"
PSEN1_FASTA = RAW_DATA_DIR / "P49768.fasta"
POLYPHEN_OUTPUT = EXTERNAL_DATA_DIR / "known_mutations_score.txt"

# Processed files
TIDY_MUTATIONS_CSV = PROCESSED_DATA_DIR / "psen1_mutations_tidy.csv"
POLYPHEN_BATCH_INPUT = PROCESSED_DATA_DIR / "known_mutations_batch.txt"
ESM2_MUTATIONS_CSV = PROCESSED_DATA_DIR / "psen1_esm2_top3.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PSEN1_OUTPUT_DIR = OUTPUT_DIR / "psen1"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model outputs
TRAINING_DATA_CSV = PSEN1_OUTPUT_DIR / "psen1_training_labeled.csv"
MODEL_PATH = PSEN1_OUTPUT_DIR / "psen1_risk_scorer.joblib"
SCORED_VARIANTS_CSV = PSEN1_OUTPUT_DIR / "psen1_scored_variants.csv"
VISUALIZATION_PNG = PSEN1_OUTPUT_DIR / "known_mutations_prediction_distribution.png"

# ESM-2 model configuration
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"  # 650M parameter model

# Model training parameters
FEATURE_COLS = ["delta_loglik", "entropy", "blosum62"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIGH_RISK_THRESHOLD = 0.9

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 OUTPUT_DIR, PSEN1_OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

