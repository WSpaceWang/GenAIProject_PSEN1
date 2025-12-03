import argparse
import os
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_known_mutations(filepath: str) -> pd.DataFrame:
    """
    Read PolyPhen2 output file into a DataFrame.
    The file is expected to be tab-separated and contain at least:
      - prediction
      - pph2_prob
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove leading '#' from the first line if present
    if lines and lines[0].lstrip().startswith("#"):
        lines[0] = lines[0].lstrip("#").lstrip()

    buf = StringIO("".join(lines))
    data = pd.read_csv(buf, sep=r"\t", engine="python", comment="#")

    # Strip whitespace from headers and string columns
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip()

    # Sanity check
    required_cols = {"prediction", "pph2_prob"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {missing}")

    return data


def plot_prediction_distribution(
    df: pd.DataFrame,
    output_path: str | None = None,
    protein_name: str | None = None,
) -> None:
    """
    Create a figure with two subplots:
    1. Pie chart showing the distribution of predictions
    2. Histogram showing the distribution of PPH2 probabilities by prediction type
    """
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for different prediction categories
    colors = {
        "probably damaging": "#FF6B6B",  # red
        "possibly damaging": "#FFB347",  # orange
        "benign": "#4CAF50",             # green
    }

    # Left subplot: Pie chart
    prediction_counts = df["prediction"].value_counts()
    patches, texts, autotexts = ax1.pie(
        prediction_counts,
        labels=prediction_counts.index,
        colors=[colors.get(x, "#808080") for x in prediction_counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )

    protein_prefix = f"{protein_name} – " if protein_name else ""
    ax1.set_title(
        f'{protein_prefix}Distribution of Predictions (Total: {len(df)})'
    )

    # Right subplot: Histogram
    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
    for prediction, color in colors.items():
        subset = df[df["prediction"] == prediction]
        if not subset.empty:
            ax2.hist(
                subset["pph2_prob"],
                bins=bins,
                alpha=0.6,
                label=prediction,
                color=color,
            )

    ax2.set_xlabel("PPH2 Probability")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"{protein_prefix}Distribution of HumDiv Scores by Prediction Type"
    )
    ax2.legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PolyPhen prediction distributions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to PolyPhen score file (tab-separated, with 'prediction' and 'pph2_prob').",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to save the figure. If omitted, show interactively.",
    )
    parser.add_argument(
        "--protein",
        required=False,
        default=None,
        help="Protein name for titles (e.g., PSEN1, APP).",
    )

    args = parser.parse_args()

    df = read_known_mutations(args.input)
    plot_prediction_distribution(df, args.output, args.protein)


if __name__ == "__main__":
    main()
