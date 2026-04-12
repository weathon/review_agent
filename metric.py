import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations
import sys

SCALE = [0, 2, 4, 6, 8, 10]

def round_to_scale(x):
    return min(SCALE, key=lambda v: abs(v - x))


def split_half_baseline(df, gt_score_cols):
    """Estimate human reliability via all unique split-half partitions per paper."""
    half_a, half_b = [], []

    for _, row in df.iterrows():
        scores = [float(row[c]) for c in gt_score_cols if pd.notna(row[c])]
        if len(scores) < 2:
            continue

        mid = len(scores) // 2
        indices = range(len(scores))
        for combo in combinations(indices, mid):
            if len(scores) % 2 == 0 and 0 not in combo:
                continue
            left = [scores[i] for i in combo]
            right = [scores[i] for i in indices if i not in combo]
            half_a.append(float(np.mean(left)))
            half_b.append(float(np.mean(right)))

    if len(half_a) < 2:
        return None

    a = np.array(half_a)
    b = np.array(half_b)
    pearson, _ = stats.pearsonr(a, b)
    spearman, _ = stats.spearmanr(a, b)
    mae = float(np.mean(np.abs(a - b)))

    return {
        "n_pairs": len(half_a),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": mae,
        "half_a": half_a,
        "half_b": half_b,
    }


def one_vs_rest_baseline(df, gt_score_cols):
    """Estimate human reliability via leave-one-reviewer-out predictions."""
    rest_means = []
    heldout_scores = []
    paper_pairs = 0

    for _, row in df.iterrows():
        human = [float(row[c]) for c in gt_score_cols if pd.notna(row[c])]
        if len(human) < 2:
            continue
        paper_pairs += len(human)
        for idx, heldout in enumerate(human):
            others = human[:idx] + human[idx + 1:]
            if not others:
                continue
            rest_means.append(float(np.mean(others)))
            heldout_scores.append(float(heldout))

    if len(rest_means) < 2:
        return None

    pearson, _ = stats.pearsonr(rest_means, heldout_scores)
    spearman, _ = stats.spearmanr(rest_means, heldout_scores)
    mae = float(np.mean(np.abs(np.array(rest_means) - np.array(heldout_scores))))

    return {
        "n_pairs": len(rest_means),
        "n_papers": paper_pairs,
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": mae,
        "rest_means": rest_means,
        "heldout_scores": heldout_scores,
    }

def analyze_and_plot(path):
    df = pd.read_csv(path)
    gt_score_cols = [c for c in df.columns if c.startswith("gt_score_")]

    # Filter out rows where pred_score is missing (ERROR / failed papers)
    n_total = len(df)
    df = df.dropna(subset=["pred_score"])
    n_dropped = n_total - len(df)
    if n_dropped > 0:
        print(f"\n  WARNING: Dropped {n_dropped}/{n_total} papers with missing predictions (ERROR rows)")

    pred = df["pred_score"].values
    gt_avg = df["gt_avg_score"].values
    pred_rounded = np.array([round_to_scale(x) for x in pred])

    sp_raw, sp_raw_p = stats.spearmanr(pred, gt_avg)
    pe_raw, pe_raw_p = stats.pearsonr(pred, gt_avg)
    sp_rnd, sp_rnd_p = stats.spearmanr(pred_rounded, gt_avg)
    mae_raw = np.mean(np.abs(pred - gt_avg))
    mae_rounded = np.mean(np.abs(pred_rounded - gt_avg))
    bias_raw = np.mean(pred - gt_avg)
    one_vs_rest = one_vs_rest_baseline(df, gt_score_cols)
    split_half = split_half_baseline(df, gt_score_cols)

    # Weighted MAE: weight by inverse frequency of GT score bins
    # Bins: [0,2), [2,4), [4,6), [6,8), [8,10]
    bin_edges = [0, 2, 4, 6, 8, 10.01]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
    bin_indices = np.digitize(gt_avg, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
    bin_counts = np.bincount(bin_indices, minlength=len(bin_labels))
    # Weight = 1/count for each bin (0 if bin is empty)
    bin_weights = np.where(bin_counts > 0, 1.0 / bin_counts, 0.0)
    sample_weights = bin_weights[bin_indices]
    # Normalize so weights sum to 1
    sample_weights = sample_weights / sample_weights.sum()
    wmae_raw = np.sum(sample_weights * np.abs(pred - gt_avg))
    wmae_rounded = np.sum(sample_weights * np.abs(pred_rounded - gt_avg))

    pred_dec = df["pred_decision"].fillna("N/A").str.strip().str.lower()
    gt_dec = df["gt_binary"].str.strip().str.lower()
    valid_dec_mask = ~pred_dec.isin(["n/a", ""])
    dec_match = ((pred_dec == gt_dec) & valid_dec_mask).sum()

    match_any = 0
    within_1std = 0
    for _, row in df.iterrows():
        r = round_to_scale(row["pred_score"])
        human = [row[c] for c in gt_score_cols if pd.notna(row[c])]
        if r in [int(s) for s in human]:
            match_any += 1
        if len(human) >= 2:
            h_mean = np.mean(human)
            h_std = np.std(human, ddof=1)
            if abs(row["pred_score"] - h_mean) <= h_std:
                within_1std += 1
        elif len(human) == 1:
            # With only 1 reviewer, no std; count as match if exact
            if round_to_scale(row["pred_score"]) == int(human[0]):
                within_1std += 1

    border_mask = (gt_avg >= 4) & (gt_avg <= 6)
    n_border = border_mask.sum()

    # ── CLI Output ──
    print(f"\n  Papers: {len(df)}")
    print(f"  {'─'*45}")
    print(f"  Spearman (raw):        {sp_raw:.4f}  (p={sp_raw_p:.4f})")
    print(f"  Spearman (rounded):    {sp_rnd:.4f}  (p={sp_rnd_p:.4f})")
    print(f"  Pearson (raw):         {pe_raw:.4f}  (p={pe_raw_p:.4f})")
    print(f"  MAE (raw):             {mae_raw:.4f}")
    print(f"  MAE (rounded):         {mae_rounded:.4f}")
    print(f"  Weighted MAE (raw):    {wmae_raw:.4f}")
    print(f"  Weighted MAE (rounded):{wmae_rounded:.4f}")
    print(f"  Bias (pred-gt):        {bias_raw:+.4f}")
    if one_vs_rest is not None:
        print(f"  {'─'*45}")
        print(f"  Human one-vs-rest baseline ({one_vs_rest['n_pairs']} held-out reviews):")
        print(f"    Spearman:            {one_vs_rest['spearman']:.4f}")
        print(f"    Pearson:             {one_vs_rest['pearson']:.4f}")
        print(f"    MAE:                 {one_vs_rest['mae']:.4f}")
    if split_half is not None:
        print(f"  {'─'*45}")
        print(f"  Human split-half baseline ({split_half['n_pairs']} exact split pairs):")
        print(f"    Spearman:            {split_half['spearman']:.4f}")
        print(f"    Pearson:             {split_half['pearson']:.4f}")
        print(f"    MAE:                 {split_half['mae']:.4f}")
    # Show bin breakdown
    print(f"  {'─'*45}")
    print(f"  Score bin weights (inverse freq):")
    for i, label in enumerate(bin_labels):
        if bin_counts[i] > 0:
            bin_mask = bin_indices == i
            bin_mae = np.mean(np.abs(pred[bin_mask] - gt_avg[bin_mask]))
            print(f"    [{label}]: n={bin_counts[i]:>3}, MAE={bin_mae:.4f}")
    print(f"  {'─'*45}")
    if valid_dec_mask.any():
        valid_decisions = int(valid_dec_mask.sum())
        print(f"  Decision accuracy:     {dec_match}/{valid_decisions} = {dec_match/valid_decisions:.1%}")
    else:
        print("  Decision accuracy:     N/A (decision labels disabled)")
    print(f"  Within 1 human std:    {within_1std}/{len(df)} = {within_1std/len(df):.1%}")
    print(f"  Human match (rounded): {match_any}/{len(df)} = {match_any/len(df):.1%}")

    # AUROC: use predicted score to discriminate Accept vs Reject
    gt_binary = (gt_dec == "accept").astype(int)  # 1=Accept, 0=Reject
    n_pos, n_neg = gt_binary.sum(), len(gt_binary) - gt_binary.sum()
    if n_pos > 0 and n_neg > 0:
        auroc = roc_auc_score(gt_binary, pred)
        fpr, tpr, thresholds = roc_curve(gt_binary, pred)
        # Human baseline AUROC: use individual reviewer scores (not the average)
        # Each individual score is an independent prediction of the paper's accept/reject label
        human_indiv_scores = []
        human_indiv_labels = []
        for i, (idx, row) in enumerate(df.iterrows()):
            label = gt_binary[i]
            for c in gt_score_cols:
                if pd.notna(row[c]):
                    human_indiv_scores.append(float(row[c]))
                    human_indiv_labels.append(label)
        human_indiv_scores = np.array(human_indiv_scores)
        human_indiv_labels = np.array(human_indiv_labels)
        n_indiv_pos = human_indiv_labels.sum()
        n_indiv_neg = len(human_indiv_labels) - n_indiv_pos
        if n_indiv_pos > 0 and n_indiv_neg > 0:
            human_auroc = roc_auc_score(human_indiv_labels, human_indiv_scores)
            human_fpr, human_tpr, _ = roc_curve(human_indiv_labels, human_indiv_scores)
        else:
            human_auroc = None
            human_fpr, human_tpr = None, None
        print(f"  AUROC (score→A/R):     {auroc:.4f}")
        if human_auroc is not None:
            print(f"  AUROC (human indiv):   {human_auroc:.4f}  ({len(human_indiv_scores)} individual scores)")
        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        print(f"  Optimal threshold:     {best_thresh:.2f} (TPR={tpr[best_idx]:.2f}, FPR={fpr[best_idx]:.2f})")
        # AUPRC
        precision, recall, _ = precision_recall_curve(gt_binary, pred)
        auprc = auc(recall, precision)
        baseline_rate = n_pos / len(gt_binary)
        print(f"  AUPRC (score→A/R):     {auprc:.4f}  (baseline={baseline_rate:.4f})")
    else:
        auroc = None
        auprc = None
        human_auroc = None
        fpr, tpr = None, None
        human_fpr, human_tpr = None, None
        print(f"  AUROC/AUPRC: N/A (only one class present: {n_pos} Accept, {n_neg} Reject)")

    if n_border > 0:
        b_mae = np.mean(np.abs(pred[border_mask] - gt_avg[border_mask]))
        print(f"  {'─'*45}")
        print(f"  Borderline (gt 4-6):   {n_border} papers")
        border_valid = valid_dec_mask[border_mask]
        if border_valid.any():
            b_dec_acc = ((pred_dec[border_mask] == gt_dec[border_mask]) & border_valid).sum()
            valid_border = int(border_valid.sum())
            print(f"    Decision accuracy:   {b_dec_acc}/{valid_border} = {b_dec_acc/valid_border:.1%}")
        else:
            print("    Decision accuracy:   N/A (decision labels disabled)")
        print(f"    MAE:                 {b_mae:.4f}")

    print(f"\n  {'─'*45}")
    print(f"  {'Paper ID':<20} {'Pred':>5} {'Rnd':>4} {'GT':>5} {'Human':<20} {'Match'}")
    print(f"  {'─'*45}")
    for _, row in df.iterrows():
        r = round_to_scale(row["pred_score"])
        human = [row[c] for c in gt_score_cols if pd.notna(row[c])]
        h_str = ",".join(str(int(s)) for s in human)
        m = "✓" if r in [int(s) for s in human] else "✗"
        print(f"  {row['paper_id']:<20} {row['pred_score']:>5.1f} {r:>4} {row['gt_avg_score']:>5.2f} [{h_str}]{'':<{16-len(h_str)}} {m}")

    # ── Plot ──
    colors = ["#e74c3c" if d.strip().lower() == "reject" else "#2ecc71" for d in df["gt_binary"]]
    legend_dots = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Accept'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Reject'),
    ]

    has_curves = auroc is not None
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    # Top-left: raw
    ax = axes[0, 0]
    ax.scatter(gt_avg, pred, c=colors, s=80, edgecolors="white", linewidth=0.8, zorder=3)
    mn, mx = min(min(pred), min(gt_avg)) - 0.5, max(max(pred), max(gt_avg)) + 0.5
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3)
    m, b = np.polyfit(gt_avg, pred, 1)
    xs = np.linspace(mn, mx, 100)
    ax.plot(xs, m * xs + b, color="#3498db", alpha=0.6)
    ax.set_xlabel("Human Average Score", fontsize=12)
    ax.set_ylabel("Agent Predicted Score", fontsize=12)
    ax.set_title("Raw Scores", fontsize=13)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.text(0.05, 0.95, f"Spearman: {sp_raw:.3f}\nPearson: {pe_raw:.3f}\nMAE: {mae_raw:.3f}\nBias: {bias_raw:+.3f}\nWithin 1 human std: {within_1std}/{len(df)} ({within_1std/len(df):.0%})\nn = {len(df)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax.legend(handles=legend_dots, fontsize=9, loc="lower right")

    # Top-right: human one-vs-rest baseline scatter or ROC baseline if unavailable
    ax2 = axes[0, 1]
    if one_vs_rest is not None and one_vs_rest.get("rest_means") and one_vs_rest.get("heldout_scores"):
        left = np.array(one_vs_rest["rest_means"])
        right = np.array(one_vs_rest["heldout_scores"])
        jitter_rng = np.random.default_rng(42)
        left_jit = left + jitter_rng.uniform(-0.35, 0.35, size=len(left))
        right_jit = right + jitter_rng.uniform(-0.35, 0.35, size=len(right))
        ax2.scatter(left_jit, right_jit, color="#f39c12", s=70, edgecolors="white", linewidth=0.8, alpha=0.9)
        mn2, mx2 = min(left.min(), right.min()) - 0.5, max(left.max(), right.max()) + 0.5
        ax2.plot([mn2, mx2], [mn2, mx2], "k--", alpha=0.3)
        if len(left) >= 2:
            m2, b2 = np.polyfit(left, right, 1)
            xs2 = np.linspace(mn2, mx2, 100)
            ax2.plot(xs2, m2 * xs2 + b2, color="#c0392b", alpha=0.7)
        ax2.set_xlabel("Mean of Other Reviewers", fontsize=12)
        ax2.set_ylabel("Held-Out Reviewer Score", fontsize=12)
        ax2.set_title("Human One-vs-Rest Baseline", fontsize=13)
        ax2.set_xlim(mn2, mx2); ax2.set_ylim(mn2, mx2); ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.2)
        ax2.text(
            0.05, 0.95,
            f"Pearson: {one_vs_rest['pearson']:.3f}\n"
            f"Spearman: {one_vs_rest['spearman']:.3f}\n"
            f"MAE: {one_vs_rest['mae']:.3f}\n"
            f"{one_vs_rest['n_pairs']} held-out reviews",
            transform=ax2.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
        )
    else:
        ax2.axis("off")

    # Bottom-left: ROC curve
    if has_curves:
        ax3 = axes[1, 0]
        ax3.plot(fpr, tpr, color="#3498db", lw=2, label=f"Agent (AUROC={auroc:.3f})")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")
        ax3.scatter([fpr[best_idx]], [tpr[best_idx]], color="#e74c3c", s=100, zorder=5,
                    label=f"Optimal threshold={best_thresh:.2f}")
        ax3.set_xlabel("False Positive Rate", fontsize=12)
        ax3.set_ylabel("True Positive Rate", fontsize=12)
        ax3.set_title("ROC Curve (Score → Accept/Reject)", fontsize=13)
        ax3.set_xlim(-0.02, 1.02); ax3.set_ylim(-0.02, 1.02)
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.2)
        ax3.legend(fontsize=9, loc="lower right")

        # Bottom-right: Precision-Recall curve
        ax4 = axes[1, 1]
        ax4.plot(recall, precision, color="#9b59b6", lw=2, label=f"Agent (AUPRC={auprc:.3f})")
        ax4.axhline(y=baseline_rate, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({baseline_rate:.3f})")
        ax4.set_xlabel("Recall", fontsize=12)
        ax4.set_ylabel("Precision", fontsize=12)
        ax4.set_title("Precision-Recall Curve (Score → Accept/Reject)", fontsize=13)
        ax4.set_xlim(-0.02, 1.02); ax4.set_ylim(-0.02, 1.02)
        ax4.set_aspect("equal")
        ax4.grid(True, alpha=0.2)
        ax4.legend(fontsize=9, loc="lower left")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    # Top-right: Split-half correlation scatter with jitter
    ax5 = axes[0, 2]
    if split_half is not None and split_half.get("half_a"):
        sh_a = np.array(split_half["half_a"])
        sh_b = np.array(split_half["half_b"])
        jitter_rng = np.random.default_rng(99)
        sh_a_jit = sh_a + jitter_rng.uniform(-0.3, 0.3, size=len(sh_a))
        sh_b_jit = sh_b + jitter_rng.uniform(-0.3, 0.3, size=len(sh_b))
        ax5.scatter(sh_a_jit, sh_b_jit, color="#8e44ad", s=80, edgecolors="white", linewidth=0.8, alpha=0.85)
        mn5, mx5 = min(sh_a.min(), sh_b.min()) - 0.5, max(sh_a.max(), sh_b.max()) + 0.5
        ax5.plot([mn5, mx5], [mn5, mx5], "k--", alpha=0.3)
        if len(sh_a) >= 2:
            m5, b5 = np.polyfit(sh_a, sh_b, 1)
            xs5 = np.linspace(mn5, mx5, 100)
            ax5.plot(xs5, m5 * xs5 + b5, color="#2c3e50", alpha=0.7)
        ax5.set_xlabel("Half A Mean Score", fontsize=12)
        ax5.set_ylabel("Half B Mean Score", fontsize=12)
        ax5.set_title("Human Split-Half Correlation", fontsize=13)
        ax5.set_xlim(mn5, mx5); ax5.set_ylim(mn5, mx5); ax5.set_aspect("equal")
        ax5.grid(True, alpha=0.2)
        ax5.text(
            0.05, 0.95,
            f"Pearson: {split_half['pearson']:.3f}\n"
            f"Spearman: {split_half['spearman']:.3f}\n"
            f"MAE: {split_half['mae']:.3f}\n"
            f"{split_half['n_pairs']} exact split pairs",
            transform=ax5.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
        )
    else:
        ax5.axis("off")

    # Bottom-right: Human AUROC curve (individual scores)
    ax6 = axes[1, 2]
    if has_curves and human_auroc is not None:
        ax6.plot(human_fpr, human_tpr, color="#f39c12", lw=2.5,
                 label=f"Human Indiv (AUROC={human_auroc:.3f})")
        ax6.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")
        ax6.set_xlabel("False Positive Rate", fontsize=12)
        ax6.set_ylabel("True Positive Rate", fontsize=12)
        ax6.set_title("ROC Curve (Human Individual Scores)", fontsize=13)
        ax6.set_xlim(-0.02, 1.02); ax6.set_ylim(-0.02, 1.02)
        ax6.set_aspect("equal")
        ax6.grid(True, alpha=0.2)
        ax6.legend(fontsize=9, loc="lower right")
    else:
        ax6.axis("off")

    plt.tight_layout()
    out = path.replace(".csv", "_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {out}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_scores.csv"
    analyze_and_plot(path)
