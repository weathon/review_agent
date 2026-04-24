import pandas as pd
from scipy.stats import spearmanr, pearsonr

df = pd.read_csv("pair_wise.csv")

spearman = spearmanr(df["gt_score"], df["pred_score"])
pearson = pearsonr(df["gt_score"], df["pred_score"])

print(f"N = {len(df)}")
print(f"Spearman: {spearman.statistic:.4f} (p={spearman.pvalue:.4e})")
print(f"Pearson:  {pearson.statistic:.4f} (p={pearson.pvalue:.4e})")