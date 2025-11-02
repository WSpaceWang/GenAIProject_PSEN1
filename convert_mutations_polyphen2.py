import pandas as pd


# Read CSV file
df = pd.read_csv("psen1_mutations_tidy.csv")


# Function to convert mutation format
def convert_mutation(mutation_str):
    # Check if the format is valid
    if len(mutation_str) < 2:
        return None
    # Extract position and amino acid information
    orig_aa = mutation_str[0]
    new_aa = mutation_str[-1]
    pos = mutation_str[1:-1]
    # Check if position is a number
    if not pos.isdigit():
        return None
    return f"PSN1_HUMAN {pos} {orig_aa} {new_aa}"


# Process first column and convert format
mutations = []
for mut in df["Mutation"]:
    converted = convert_mutation(mut)
    if converted:
        mutations.append(converted)

# Save to file
with open("known_mutations_batch.txt", "w") as f:
    for mut in mutations:
        f.write(mut + "\n")
