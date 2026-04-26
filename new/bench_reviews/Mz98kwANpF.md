Now I have sufficient context. Let me finalize and write the review.

## Summary

The paper challenges the prevailing multi-component LoRA paradigm for multi-task learning by showing that (1) a simplified multi-head variant (M-LoRA) without routing outperforms more complex designs, (2) simply increasing the rank of a standard single-adapter LoRA matches multi-component architectures at comparable parameter budgets, and (3) a proposed Align-LoRA method that adds a KL-divergence alignment loss on LoRA's down-projection representations consistently improves performance across model scales and families while introducing zero inference overhead.

## Strengths

- **Compelling empirical challenge to multi-component LoRA**: Tables 2–3 demonstrate that a standard single-adapter LoRA at comparable parameter budgets matches or exceeds HydraLoRA, R-LoRA, and LoRA MoE. This is a clean, important finding that questions an architectural trend in the community. For example, LoRA rank-10 at 0.25% params achieves 49.51 on Qwen2.5-7B BBH, tying R-LoRA and beating HydraLoRA (49.12).

- **Strong experimental coverage**: Evaluations span Qwen2.5 (3B/7B/14B), LLaMA2 (7B/13B), and LLaMA3 (8B) on both out-of-domain (BBH) and in-domain (8-task) benchmarks, with multiple alignment objectives (KL, MMD) and multiple baseline families. This breadth makes the empirical challenge to multi-component LoRA difficult to dismiss.

- **Zero inference overhead by design**: Unlike MoE-routed variants, Align-LoRA's weights merge into the base model after training. The paper explicitly quantifies this advantage (Appendix C and D), demonstrating lower FLOPs and faster training.

- **Clear narrative progression**: The paper logically builds from observation (M-LoRA paradox) → hypothesis testing (rank scaling) → principled method (Align-LoRA), making the motivation feel earned.

## Weaknesses

### Fatal

None.

### Major

- **Missing same-rank LoRA baseline in Tables 4–5 creates a confound**: Align-LoRA uses rank 8 (0.20% params) while the standard LoRA baseline in Tables 4–5 uses rank 10 (0.25% params). The paper attributes A-LoRA-K's improvements over LoRA-rank-10 to the alignment loss, but without a LoRA-rank-8 baseline in the same experimental setup, we cannot fully isolate the alignment effect from possible confounds (e.g., different optimization dynamics at rank 8, hyperparameter search advantages). Table 3 does show LoRA-rank-8 scoring 46.66 vs. A-LoRA-K's 50.28, but these are in different experimental contexts (different training data subsets). The 1.92-point gap between A-LoRA-K (50.28) and LoRA-rank-10 (48.36) on Qwen2.5-7B BBH is substantial enough that the result would likely survive this ablation, but the absence of a direct same-rank comparison leaves a gap in validating the core contribution. The parameter budget difference (0.20% vs 0.25%) means Align-LoRA actually uses *fewer* parameters, which partly mitigates the concern—but a same-rank baseline would cleanly isolate the alignment effect.

- **The "task-shared representation" mechanism is not distinguished from regularization**: The alignment loss pushes per-task batch statistics (mean/variance) toward a common value. This could function as a regularizer—constraining the function class and preventing overfitting to individual tasks—rather than genuinely promoting cross-task knowledge transfer. The paper provides no probe experiments testing whether aligned representations actually transfer knowledge, nor comparisons against other regularizers (e.g., increased weight decay, dropout) at matched parameter budgets. The A-LoRA-M (MMD) result partially addresses this by showing a different distance metric also helps, but A-LoRA-M often underperforms A-LoRA-K substantially (e.g., 82.31 vs 83.95 on Qwen2.5-7B in Table 5)—which itself raises questions about the generality claim. Distinguishing alignment from regularization would significantly strengthen the paper's interpretive claims.

### Minor

- **The theoretical bound (Section 5.3) does not specifically validate Align-LoRA**: Equation 7 states that generalization error is bounded by training risk plus a cross-task distribution discrepancy term plus a standard complexity term. This is a well-known domain-adaptation principle (Ben-David et al., 2006). The paper does not establish that minimizing KL divergence between Gaussian-batch statistics from A's output actually reduces the Δ(D_i, D_j) term, nor does it analyze Align-LoRA's hypothesis class capacity. The section asserts rather than demonstrates the connection, making the theory decorative rather than substantiative. This is a minor issue because the paper's core contribution is empirical.

- **A-LoRA-M underperformance undermines the "broadly applicable principle" claim**: The paper states "the principle of aligning representations is broadly applicable and not contingent on a single metric." Yet A-LoRA-M consistently underperforms A-LoRA-K, sometimes failing to beat even M-LoRA (Table 5, 3B: A-LoRA-M at 78.35 vs M-LoRA at 78.51). If alignment is the key principle, the dependence on the specific metric choice needs better explanation.

- **No variance or statistical significance reported**: Table results show 1–3 point improvements without standard deviations or confidence intervals across seeds, making it hard to assess whether differences are meaningful for the smaller margins.

### Trivial

None.

## Nice-to-Haves

- A same-rank LoRA baseline (rank 8) in Tables 4–5 would cleanly isolate the alignment contribution.
- Probe experiments to test whether aligned representations exhibit genuine cross-task transfer vs. mere regularization.
- Per-task performance disaggregation to reveal whether alignment helps all tasks uniformly or disproportionately benefits structurally similar ones.

## Removed Points

- **"Abstract contributions are redundant"** (Harsh Critic): This is a minor presentation nitpick. While contributions 1–3 are related, they describe distinct findings (empirical challenge, rank-scaling result, hypothesis framing). Not a substantive weakness.

- **"Inter-head cosine similarity on B_i doesn't capture functional similarity"** (Harsh Critic): The paper uses the metric to show that M-LoRA has high redundancy while R-LoRA has low redundancy—a contrast metric, not an absolute measure. The correlation with performance differences validates its use as a rough indicator. The claim is not that cosine similarity perfectly captures functional redundancy, but that the relative ordering matters.

- **"HydraLoRA w/o Router ablation is not a clean factorial design"** (Harsh Critic): While true that a full factorial (±router × ±dropout) would be ideal, the paper does show that M-LoRA (router+dropout removal + dropout retention) outperforms HydraLoRA-w/o-Router (router removal only, no dropout), which provides evidence for the dropout contribution. A cleaner design would help, but the presented ablation is informative.

- **"The A/B role distinction relies on the multi-head paradigm the paper argues against"** (Harsh Critic): The paper uses the A/B role observation instrumentally (to motivate where to apply alignment) while arguing against the multi-head architecture. These are not contradictory: the observation that A learns general features is orthogonal to the architectural claim about whether heads should be separated. One can accept the empirical A/B role finding while questioning multi-head routing.

- **"The paper overclaims that architectural complexity is unnecessary"** (Harsh Critic): The paper says "may not be a prerequisite" and "suggests it is a less effective path"—language that is appropriately hedged for the scope of the experiments.

## Novel Insights

The paper's most novel insight is the "paradox of diversity" in multi-head LoRA: methods designed to encourage head diversity (R-LoRA) actually perform worse than methods with highly redundant heads (M-LoRA). This directly challenges the MoE-inspired design philosophy in the multi-task LoRA literature. However, the insight that alignment may be acting primarily as regularizer—rather than truly facilitating cross-task representation sharing—is an important distinction the paper leaves unresolved.

## Suggestions

- Add a LoRA-rank-8 baseline in Tables 4–5 under identical experimental conditions to directly isolate the alignment loss's contribution from rank/hyperparameter confounds.
- Compare Align-LoRA against a simple regularization baseline (e.g., LoRA rank-8 + increased weight decay or dropout) to distinguish alignment from regularization effects.
- Add per-task score breakdowns for Align-LoRA to reveal task-level effects.

## Score and Decision

**Calibration anchors considered:**

1. **High score anchors (≥6)**: MeteoRA (6.20) — multi-task LoRA MoE, incremental but solid; ComLoRA (6.00) — competitive LoRA training, limited baselines; RaSA (7.00) — questions LoRA's low-rank bottleneck with theoretical grounding and strong results; FLoRA (8.00) — efficient multi-task batching, practical and clean; HiRA (8.00) — challenges LoRA paradigm with Hadamard high-rank, strong empirical gains; SD-LoRA (7.50) — LoRA for continual learning with decoupled learning.

2. **Medium score anchors (~5)**: GatedMTL (5.25) — proposes gating for MTL sharing with regularization; Dual-Balancing (5.4) — loss/gradient balancing for MTL.

3. **Low score anchors (≤4)**: UnoLoRA (3.00) — single shared adapter for multi-task, very limited evaluation and questionable claims; MORE/MoRE LoRA (4.00) — marginal improvements, missing ablations, limited baselines; ME-LoRA (3.75) — Bayesian LoRA, questionable practicality.

The paper under review is stronger than low-scoring LoRA papers (UnoLoRA, MoRE) which had marginal improvements and missing ablations. It has a genuinely important empirical finding (multi-component LoRA is unnecessary) and a clean method (Align-LoRA) with consistent improvements. However, the missing same-rank baseline is a real gap for validating Align-LoRA, and the regularization vs. alignment distinction is unresolved. Compared to mid-score papers (GatedMTL ~5.25), this paper has broader experimental coverage and a clearer narrative, but the same type of unresolved mechanistic question. Compared to RaSA (7.00), this paper has less theoretical depth and a missing key ablation, but comparable empirical breadth and a similarly important challenge to the status quo.

The core empirical finding (multi-component LoRA is unnecessary) is well-supported. The Align-LoRA contribution is strong but not fully validated due to the missing rank-controlled ablation. The theoretical section adds little. Overall, this lands in the 5-6 range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>