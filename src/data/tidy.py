# src/data/tidy.py
import argparse
import os
from io import StringIO

import pandas as pd


def convert_polyphen(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if lines and lines[0].lstrip().startswith("#"):
        lines[0] = lines[0].lstrip("#").lstrip()

    buf = StringIO("".join(lines))

    df = pd.read_csv(buf, sep="\t", engine="python")
    df.columns = df.columns.str.strip()

    required = ["o_pos", "o_aa1", "o_aa2", "prediction", "pph2_prob"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column {col} missing in polyphen file. Got: {df.columns.tolist()}"
            )

    pos_numeric = pd.to_numeric(df["o_pos"], errors="coerce")
    valid_mask = pos_numeric.notna()

    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with invalid o_pos")

    df_valid = df.loc[valid_mask].copy()
    pos_numeric = pos_numeric[valid_mask].astype(int)

    tidy = pd.DataFrame(
        {
            "position_1based": pos_numeric,
            "wt_aa": df_valid["o_aa1"].astype(str).str.strip(),
            "mut_aa": df_valid["o_aa2"].astype(str).str.strip(),
            "pph2_prediction": df_valid["prediction"].astype(str).str.strip(),
            "pph2_prob": df_valid["pph2_prob"].astype(float),
        }
    )


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tidy.to_csv(output_path, index=False)
    print(f"[OK] Saved tidy polyphen table to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_polyphen(args.input, args.output)


if __name__ == "__main__":
    main()
