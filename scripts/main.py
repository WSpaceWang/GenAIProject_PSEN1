#!/usr/bin/env python
"""
Main entry point for PSEN1 mutation risk prediction pipeline.

This script orchestrates the full workflow:
1. Data preprocessing
2. Mutation generation with ESM-2
3. Risk model training and scoring
4. Results visualization
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.data.preprocessing import main as preprocess_data
from src.data.convert_polyphen import main as convert_polyphen
from scripts.generate_mutations import main as generate_mutations
from scripts.run_analysis import main as run_analysis
from src.visualization.visualize_polyphen import plot_prediction_distribution, read_known_mutations


def main():
    parser = argparse.ArgumentParser(
        description="PSEN1 Mutation Risk Prediction Pipeline"
    )
    parser.add_argument(
        "step",
        choices=["preprocess", "convert", "generate", "analyze", "visualize", "all"],
        help="Pipeline step to execute"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps if output files already exist"
    )
    
    args = parser.parse_args()
    
    if args.step == "preprocess" or args.step == "all":
        print("=" * 60)
        print("Step 1: Data Preprocessing")
        print("=" * 60)
        preprocess_data()
        print()
    
    if args.step == "convert" or args.step == "all":
        print("=" * 60)
        print("Step 2: Convert Mutations for PolyPhen-2")
        print("=" * 60)
        convert_polyphen()
        print()
    
    if args.step == "generate" or args.step == "all":
        print("=" * 60)
        print("Step 3: Generate Mutations with ESM-2")
        print("=" * 60)
        generate_mutations()
        print()
    
    if args.step == "analyze" or args.step == "all":
        print("=" * 60)
        print("Step 4: Train Model and Score Variants")
        print("=" * 60)
        run_analysis()
        print()
    
    if args.step == "visualize" or args.step == "all":
        print("=" * 60)
        print("Step 5: Visualize Results")
        print("=" * 60)
        try:
            df = read_known_mutations("data/external/known_mutations_score.txt")
            plot_prediction_distribution(df, "outputs/psen1/known_mutations_prediction_distribution.png")
        except Exception as e:
            print(f"Visualization error: {e}")
        print()
    
    print("=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

