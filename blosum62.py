import blosum as bl
import pandas as pd
import visualize_polyphen as vp


matrix = bl.BLOSUM(62)  # Load BLOSUM62 matrix
df = vp.read_known_mutations("results/known_mutations_score.txt")


# Calculate BLOSUM62 scores for each mutation
def get_blosum62_score(row):
    return matrix[row['aa1']][row['aa2']]


# Apply the function to each row in the DataFrame
df['blosum62_score'] = df.apply(get_blosum62_score, axis=1)

# Print the distribution of BLOSUM62 scores
print(df['blosum62_score'].value_counts().sort_index(ascending=False))




# 分析相关性
correlation = df['blosum62_score'].corr(df['pph2_prob'])
print(f"Correlation between BLOSUM62 and PPH2: {correlation}")

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(data=df, x='blosum62_score', y='pph2_prob')
plt.show(block=True)