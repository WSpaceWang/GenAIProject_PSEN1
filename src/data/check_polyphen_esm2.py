import pandas as pd
import argparse
import sys

def check_distribution(filepath):
    print(f" Checking file: {filepath}")
    
    try:
        # FIX: Use sep='\t' because PolyPhen results are Tab-Separated.
        # 'probably damaging' contains a space, using \s+ breaks it!
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='warn')
        
        # Clean header: remove '#' and extra spaces
        df.columns = [c.replace('#', '').strip() for c in df.columns]
        
        # Check columns
        if 'pph2_prob' not in df.columns:
            print(" Error: Column 'pph2_prob' not found.")
            print(f"   Found columns: {df.columns.tolist()}")
            return

    except Exception as e:
        print(f" Error reading file: {e}")
        return

    # Basic Stats
    total_count = len(df)
    print(f"\n--- Basic Info ---")
    print(f"Total Rows: {total_count}")
    
    if total_count == 0:
        print(" File is empty.")
        return

    # Score Statistics
    print(f"\n--- Score Statistics (pph2_prob) ---")
    print(f"Min Score: {df['pph2_prob'].min():.4f}")
    print(f"Max Score: {df['pph2_prob'].max():.4f} (Should be close to 1.0)")
    print(f"Mean Score: {df['pph2_prob'].mean():.4f}")
    print(f"Median Score: {df['pph2_prob'].median():.4f}")

    # Prediction Categories
    print(f"\n--- Prediction Categories ---")
    if 'prediction' in df.columns:
        counts = df['prediction'].value_counts()
        for category, count in counts.items():
            percentage = (count / total_count) * 100
            print(f"{category:<20}: {count:>5} ({percentage:>5.1f}%)")

    # Histogram
    print(f"\n--- Score Distribution (ASCII Histogram) ---")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2 (Benign)', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0 (Damaging)']
    
    binned = pd.cut(df['pph2_prob'], bins=bins, labels=labels, include_lowest=True)
    bin_counts = binned.value_counts().sort_index()
    
    for label, count in bin_counts.items():
        bar_len = int((count / total_count) * 50)
        bar = '█' * bar_len
        print(f"{label:<16} | {count:>4} {bar}")

    print("\n Check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to PolyPhen result file')
    args = parser.parse_args()
    
    check_distribution(args.file)