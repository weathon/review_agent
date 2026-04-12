import json
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEED = 42


def load_rows(path: Path):
    rows = json.loads(path.read_text())
    out = []
    seen = set()
    for row in rows:
        paper_id = row.get("paper_id")
        scores = row.get("scores")
        avg_score = row.get("avg_score")
        if paper_id in seen or paper_id is None:
            continue
        if not isinstance(scores, list) or len(scores) < 2 or avg_score is None:
            continue
        vals = []
        bad = False
        for score in scores:
            try:
                vals.append(float(score))
            except Exception:
                bad = True
                break
        if bad or len(vals) < 2:
            continue
        seen.add(paper_id)
        out.append(
            {
                "paper_id": paper_id,
                "scores": vals,
                "avg_score": float(avg_score),
            }
        )
    return out


def split_half(rows):
    half_a = []
    half_b = []

    for row in rows:
        scores = row["scores"]
        mid = len(scores) // 2
        indices = range(len(scores))
        for combo in combinations(indices, mid):
            if len(scores) % 2 == 0 and 0 not in combo:
                continue
            left = [scores[i] for i in combo]
            right = [scores[i] for i in indices if i not in combo]
            half_a.append(float(np.mean(left)))
            half_b.append(float(np.mean(right)))

    half_a = np.array(half_a)
    half_b = np.array(half_b)

    return {
        "half_a": half_a,
        "half_b": half_b,
        "pearson": float(stats.pearsonr(half_a, half_b).statistic),
        "spearman": float(stats.spearmanr(half_a, half_b).statistic),
        "mae": float(np.mean(np.abs(half_a - half_b))),
        "n_pairs": len(half_a),
    }


def one_vs_rest(rows):
    rest_means = []
    heldout_scores = []
    for row in rows:
        scores = row["scores"]
        for idx, heldout in enumerate(scores):
            others = scores[:idx] + scores[idx + 1 :]
            rest_means.append(float(np.mean(others)))
            heldout_scores.append(float(heldout))

    rest_means = np.array(rest_means)
    heldout_scores = np.array(heldout_scores)

    return {
        "rest_means": rest_means,
        "heldout_scores": heldout_scores,
        "pearson": float(stats.pearsonr(rest_means, heldout_scores).statistic),
        "spearman": float(stats.spearmanr(rest_means, heldout_scores).statistic),
        "mae": float(np.mean(np.abs(rest_means - heldout_scores))),
        "n_pairs": len(rest_means),
    }


def summarize(rows, split_half_metrics, one_vs_rest_metrics):
    avg_scores = np.array([row["avg_score"] for row in rows])
    paper_stds = np.array([np.std(row["scores"], ddof=1) for row in rows])
    return {
        "n_papers": len(rows),
        "avg_score_mean": float(np.mean(avg_scores)),
        "avg_score_sd": float(np.std(avg_scores)),
        "paper_std_mean": float(np.mean(paper_stds)),
        "split_half_sb": float(
            (2 * split_half_metrics["pearson"])
            / (1 + split_half_metrics["pearson"])
        ),
        "one_vs_rest_pearson": one_vs_rest_metrics["pearson"],
    }


def add_scatter(ax, x, y, title, x_label, y_label, color, metrics_text):
    jitter_rng = np.random.default_rng(SEED)
    x_jit = x + jitter_rng.uniform(-0.18, 0.18, size=len(x))
    y_jit = y + jitter_rng.uniform(-0.18, 0.18, size=len(y))
    ax.scatter(x_jit, y_jit, s=16, alpha=0.12, color=color, edgecolors="none")

    mn = min(float(np.min(x)), float(np.min(y))) - 0.5
    mx = max(float(np.max(x)), float(np.max(y))) + 0.5
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.25)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(mn, mx, 100)
    ax.plot(xs, slope * xs + intercept, color="#2c3e50", alpha=0.75, linewidth=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
    )


def main():
    data = {
        "ICLR 2025": load_rows(Path("iclr2025_data/all_notes.json")),
        "ICLR 2026": load_rows(Path("iclr2026_unbalanced/all_notes.json")),
    }

    metrics = {}
    for name, rows in data.items():
        sh = split_half(rows)
        ovr = one_vs_rest(rows)
        summary = summarize(rows, sh, ovr)
        metrics[name] = {
            "split_half": sh,
            "one_vs_rest": ovr,
            "summary": summary,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    colors = {
        "ICLR 2025": "#1f77b4",
        "ICLR 2026": "#d62728",
    }

    for col, name in enumerate(["ICLR 2025", "ICLR 2026"]):
        sh = metrics[name]["split_half"]
        ovr = metrics[name]["one_vs_rest"]

        add_scatter(
            axes[0, col],
            sh["half_a"],
            sh["half_b"],
            f"{name} Human Split-Half",
            "Half A Mean Score",
            "Half B Mean Score",
            colors[name],
            "\n".join(
                [
                    f"Pearson: {sh['pearson']:.3f}",
                    f"Spearman: {sh['spearman']:.3f}",
                    f"MAE: {sh['mae']:.3f}",
                    f"Spearman-Brown: {(2 * sh['pearson']) / (1 + sh['pearson']):.3f}",
                    f"{sh['n_pairs']:,} exact split pairs",
                ]
            ),
        )

        add_scatter(
            axes[1, col],
            ovr["rest_means"],
            ovr["heldout_scores"],
            f"{name} Human One-vs-Rest",
            "Mean of Other Reviewers",
            "Held-Out Reviewer Score",
            colors[name],
            "\n".join(
                [
                    f"Pearson: {ovr['pearson']:.3f}",
                    f"Spearman: {ovr['spearman']:.3f}",
                    f"MAE: {ovr['mae']:.3f}",
                    f"{ovr['n_pairs']:,} held-out reviews",
                ]
            ),
        )

    header_lines = []
    for name in ["ICLR 2025", "ICLR 2026"]:
        summary = metrics[name]["summary"]
        header_lines.append(
            f"{name}: n={summary['n_papers']:,} | avg SD={summary['avg_score_sd']:.3f} | "
            f"paper disagreement={summary['paper_std_mean']:.3f} | "
            f"split-half Pearson={metrics[name]['split_half']['pearson']:.3f} | "
            f"SB={summary['split_half_sb']:.3f} | "
            f"one-vs-rest Pearson={summary['one_vs_rest_pearson']:.3f}"
        )

    fig.suptitle("Human Reviewer Agreement: ICLR 2025 vs ICLR 2026", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.945,
        "\n".join(header_lines),
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = Path("human_corr_2025_vs_2026.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
