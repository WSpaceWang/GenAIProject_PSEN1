import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt


file_path = "results/known_mutations_score.txt"


# Read data from PolyPhen2
def read_known_mutations(filepath: str) -> pd.DataFrame:
    # Read all lines from the file
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Remove leading '#' from the first line if present
    if lines and lines[0].lstrip().startswith("#"):
        lines[0] = lines[0].lstrip("#").lstrip()
    buf = StringIO("".join(lines))
    data = pd.read_csv(buf, sep=r"\t", engine="python", comment="#")
    # Strip whitespace from headers and string columns
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    return data


# Visualization
def plot_prediction_distribution(df: pd.DataFrame) -> None:
    """
    Create a figure with two subplots:
    1. Pie chart showing the distribution of predictions
    2. Histogram showing the distribution of PPH2 probabilities by prediction type
    """
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Define colors for different prediction categories
    colors = {
        'probably damaging': '#FF6B6B',  # red
        'possibly damaging': '#FFB347',  # orange
        'benign': '#4CAF50'  # green
    }
    # Left subplot: Pie chart
    prediction_counts = df['prediction'].value_counts()
    patches, texts, autotexts = ax1.pie(
        prediction_counts,
        labels=prediction_counts.index,
        colors=[colors[x] for x in prediction_counts.index],
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title(f'Distribution of Predictions (Total: {len(df)})')
    # Right subplot: Histogram
    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
    for prediction in colors.keys():
        subset = df[df['prediction'] == prediction]
        ax2.hist(
            subset['pph2_prob'],
            bins=bins,
            alpha=0.6,
            label=prediction,
            color=colors[prediction]
        )
    ax2.set_xlabel('PPH2 Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of HumDiv Scores by Prediction Type')
    ax2.legend()
    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/known_mutations_prediction_distribution.png", dpi=300)
    plt.close()


# Read known mutations data
df = read_known_mutations(file_path)
# Generate visualization
plot_prediction_distribution(df)


