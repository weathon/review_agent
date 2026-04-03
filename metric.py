import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

SCALE = [1, 3, 5, 6, 8, 10]

def round_to_scale(x):
    return min(SCALE, key=lambda v: abs(v - x))

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

    pred_dec = df["pred_decision"].fillna("N/A").str.strip().str.lower()
    gt_dec = df["gt_binary"].str.strip().str.lower()
    valid_dec_mask = ~pred_dec.isin(["n/a", ""])
    dec_match = ((pred_dec == gt_dec) & valid_dec_mask).sum()

    match_any = 0
    for _, row in df.iterrows():
        r = round_to_scale(row["pred_score"])
        human = [row[c] for c in gt_score_cols if pd.notna(row[c])]
        if r in [int(s) for s in human]:
            match_any += 1

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
    print(f"  Bias (pred-gt):        {bias_raw:+.4f}")
    if valid_dec_mask.any():
        valid_decisions = int(valid_dec_mask.sum())
        print(f"  Decision accuracy:     {dec_match}/{valid_decisions} = {dec_match/valid_decisions:.1%}")
    else:
        print("  Decision accuracy:     N/A (decision labels disabled)")
    print(f"  Human match (rounded): {match_any}/{len(df)} = {match_any/len(df):.1%}")

    # AUROC: use predicted score to discriminate Accept vs Reject
    gt_binary = (gt_dec == "accept").astype(int)  # 1=Accept, 0=Reject
    n_pos, n_neg = gt_binary.sum(), len(gt_binary) - gt_binary.sum()
    if n_pos > 0 and n_neg > 0:
        auroc = roc_auc_score(gt_binary, pred)
        fpr, tpr, thresholds = roc_curve(gt_binary, pred)
        print(f"  AUROC (score→A/R):     {auroc:.4f}")
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
        fpr, tpr = None, None
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

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
    ax.text(0.05, 0.95, f"Spearman: {sp_raw:.3f}\nPearson: {pe_raw:.3f}\nMAE: {mae_raw:.3f}\nBias: {bias_raw:+.3f}\nn = {len(df)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax.legend(handles=legend_dots, fontsize=9, loc="lower right")

    # Top-right: rounded
    ax2 = axes[0, 1]
    rng = np.random.default_rng(42)
    jx, jy = rng.uniform(-0.15, 0.15, len(df)), rng.uniform(-0.15, 0.15, len(df))
    ax2.scatter(gt_avg + jx, pred_rounded + jy, c=colors, s=80, edgecolors="white", linewidth=0.8, zorder=3)
    ax2.plot([0, 11], [0, 11], "k--", alpha=0.3)
    ax2.set_xlabel("Human Average Score", fontsize=12)
    ax2.set_ylabel("Agent Rounded Score", fontsize=12)
    ax2.set_title("Rounded to ICLR Scale {1,3,5,6,8,10}", fontsize=13)
    ax2.set_yticks(SCALE); ax2.set_xlim(0, 11); ax2.set_ylim(0, 11); ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.2)
    ax2.text(0.05, 0.95, f"Spearman: {sp_rnd:.3f}\nMAE: {mae_rounded:.3f}\nHuman match: {match_any}/{len(df)} ({match_any/len(df):.0%})",
             transform=ax2.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax2.legend(handles=legend_dots, fontsize=9, loc="lower right")

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

    plt.tight_layout()
    out = path.replace(".csv", "_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {out}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_scores.csv"
    analyze_and_plot(path)
