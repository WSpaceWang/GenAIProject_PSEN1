import sys
import os
import blosum as bl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from src
sys.path.append(os.getcwd())

import src.visualization.visualize_polyphen as vp

INPUT_FILE = "data/external/known_mutations_score.txt"

def main():
    try:
        matrix = bl.BLOSUM(62)  # Load BLOSUM62 matrix
        df = vp.read_known_mutations(INPUT_FILE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate BLOSUM62 scores for each mutation
    def get_blosum62_score(row):
        return matrix[row['aa1']][row['aa2']]

    # Apply the function to each row in the DataFrame
    df['blosum62_score'] = df.apply(get_blosum62_score, axis=1)

    # Print the distribution of BLOSUM62 scores
    print("Distribution of BLOSUM62 scores:")
    print(df['blosum62_score'].value_counts().sort_index(ascending=False))

    # Analyze correlation
    correlation = df['blosum62_score'].corr(df['pph2_prob'])
    print(f"Correlation between BLOSUM62 and PPH2: {correlation}")

    # Visualization
    # sns.barplot(data=df, x='blosum62_score', y='pph2_prob')
    # plt.show(block=True)
    # Since we are likely running on a server without X11, saving might be better or just skipping plot show
    print("Analysis complete.")

if __name__ == "__main__":
    main()
