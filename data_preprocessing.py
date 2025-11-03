import pandas as pd


# Read the raw CSV file
df = pd.read_csv("data/raw_data/psen1_mutations_raw.csv", dtype=str)

# Remove rows that are completely empty
df = df.dropna(how="all")
# print(df["Research Models"].notna().sum())  # Check non-empty "Research Models" entries


# Split the "Mutation Type/Codon Change" column into "Mutation Type" and "Codon Change"
def split_mc_column(data: pd.DataFrame) -> pd.DataFrame:
    # Create a new column, called "Codon Change"
    if "Codon Change" not in data.columns:
        data.insert(data.columns.get_loc("Mutation Type/Codon Change") + 1, "Codon Change", "")
    # Find rows where "Mutation Type/Codon Change" contains "to"
    contains_to_mask = data["Mutation Type/Codon Change"].apply(lambda x: "to" in str(x).lower())
    rows_with_to = data.index[contains_to_mask].tolist()
    # Move the "Codon Change" values to the new "Codon Change" column
    for idx in rows_with_to:
        # Check if "Research Models" is empty for this row
        if pd.isna(data.at[idx, "Research Models"]) or str(data.at[idx, "Research Models"]).strip() == "":
            # If "Research Models" is empty, move value to previous row"s "Codon Change"
            prev_idx = idx - 1
            val = str(data.at[idx, "Mutation Type/Codon Change"]).strip()
            data.at[prev_idx, "Codon Change"] = val
        else:
            # If "Research Models" has value, move to same row"s "Codon Change"
            val = str(data.at[idx, "Mutation Type/Codon Change"]).strip()
            data.at[idx, "Codon Change"] = val
            # Clear the original cell
            data.at[idx, "Mutation Type/Codon Change"] = ""
    # Change the column name
    data = data.rename(columns={"Mutation Type/Codon Change": "Mutation Type"})
    return data


# Clean the data after the split (based on "Research Models" column)
df = split_mc_column(df)
df["Research Models"] = df["Research Models"].apply(lambda x: x if str(x).strip() != "" else None)
df = df.dropna(subset=["Research Models"]).reset_index(drop=True)
# print(df["Mutation"].notna().sum())  # Check non-empty "Mutation" entries

# Drop rows where "Mutation" column contains "del" or "ins" or "dup"
df = df[~df["Mutation"].str.contains("del|ins|dup", case=False, na=False)].reset_index(drop=True)

# Keep only rows that contain "Pathogenic" in the "Pathogenicity" column
# print(df["Pathogenicity"].unique())  # Check unique values in "Pathogenicity" column
df = df[df["Pathogenicity"].str.contains("Pathogenic", case=False, na=False)].reset_index(drop=True)

# Drop the "Research Models" and "Primary Papers" columns
df = df.drop(columns=["Research Models", "Primary Papers"])


# Remove content within parentheses from the "Mutation" column and handle duplicates
def remove_parentheses(data: pd.DataFrame) -> pd.DataFrame:
    # Remove content within parentheses from the "Mutation" column
    data["Mutation"] = data["Mutation"].str.replace(r"\s*\([^)]*\)", "", regex=True)
    # Find duplicated mutations
    duplicates = data[data["Mutation"].duplicated(keep=False)].sort_values("Mutation")
    # Group by Mutation
    for mutation in duplicates["Mutation"].unique():
        mask = data["Mutation"] == mutation
        # Keep first row and combine other rows' content
        for column in data.columns:
            if column != "Mutation":
                unique_values = data.loc[mask, column].dropna().unique()
                if len(unique_values) > 1:
                    combined_value = "|".join(unique_values)
                    data.loc[mask.idxmax(), column] = combined_value
        # Drop other duplicate rows
        data = data.drop(index=data[mask].index[1:])
    data = data.reset_index(drop=True)
    return data


df = remove_parentheses(df)
# print(df["Mutation"].nunique())

# Save the cleaned DataFrame to a new CSV file
df.to_csv("data/tidy_data/psen1_mutations_tidy.csv", index=False)
