Now let me search for calibration papers.Now I have enough information to write the final review. Let me synthesize.

---

## Summary
This paper introduces the problem of attributing **global properties** of diffusion models (e.g., image quality, demographic diversity, aesthetic score) to **groups of training data contributors**, distinguishing itself from prior work that attributes local, per-sample properties. The authors propose a computationally efficient Shapley value framework using sparsified fine-tuning (magnitude pruning + fine-tuning) to approximate retraining, enabling feasible Shapley estimation for diffusion models. Experiments across three datasets (CIFAR-20, CelebA-HQ, ArtBench) and three model architectures (DDPM, LDM, SD-LoRA) show strong improvements over existing attribution baselines on the LDS metric.

---

## Strengths

- **Genuinely novel problem formulation** (Section 1, Definition 1): The paper clearly distinguishes *global* property attribution to *contributors* (groups) from prior local per-datum methods like D-TRAK and Journey-TRAK. This is a clean conceptual contribution with concrete practical motivation (fair crediting and compensation).

- **Strong empirical results on two of three datasets** (Table 1): Sparsified-FT Shapley achieves 61.48% and 61.44% LDS on CIFAR-20 and ArtBench respectively, far above all baselines including LOO (30.66% on CIFAR-20) and TRAK variants (many negative). The margins are large enough to be practically meaningful.

- **Principled budget-controlled comparison within Shapley variants** (Figure 2): Under identical computational budgets, sparsified fine-tuning consistently outperforms full fine-tuning and full retraining for Shapley estimation across all three datasets. The speedup factors (5.3×, 10.4×, 18.6×) are concrete and reproducible.

- **Multi-faceted evaluation** (Section 4.4–4.5): The combination of LDS (quantitative ranking correlation) and counterfactual analysis (actual model retraining after removing top-K contributors) provides corroborating evidence that identified contributors are genuinely important, not just artifacts of the metric.

- **Diverse experimental settings** (Section 4.1–4.2): Three substantially different architectures (35.7M DDPM, 274M LDM, 5.1M SD-LoRA) and three qualitatively different global properties (Inception Score, diversity entropy, 90th-percentile aesthetic score) make the results harder to dismiss as dataset-specific artifacts.

- **Qualitative interpretability validation** (Section 4.5): Top CIFAR-20 contributors have lower average classifier entropy (5.18 vs. 6.03), top CelebA-HQ contributors belong to non-majority demographic clusters, and top ArtBench contributors produce more vivid, colorful images — all consistent with the claimed global properties.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 1 does not control for computational cost, weakening the "outperforms existing methods" claim.** TRAK, D-TRAK, Journey-TRAK, and influence functions operate on a single already-trained model with gradient computation; sparsified-FT Shapley runs M fine-tuning jobs (each 10–23 minutes). Figure 2 controls for compute only *within* Shapley variants; it does not place TRAK, LOO, or influence functions on the same budget axis. While the paper's strongest argument is that Shapley with marginal subset contributions is fundamentally better-suited for global attribution (supported by LOO at 30.66% vs 61.48% with similar retraining cost), the claim in Contribution 3 that the method "outperforms existing attribution methods" cannot be taken at face value without a budget-matched comparison. The authors should add TRAK/IF/LOO to Figure 2's x-axis with equivalent compute units.

### Minor

- **CelebA-HQ LDS (26.34%) is substantially lower than the other two datasets (61%+) with no analysis.** While the method still outperforms all baselines on CelebA-HQ, the large gap raises unanswered questions: is the diversity entropy metric intrinsically harder to attribute? Is the 274M LDM less amenable to sparsification? Is 50 contributors with KernelSHAP coverage suboptimal? A brief ablation or explanation would meaningfully strengthen the paper's generality claim.

- **The number of KernelSHAP samples M is unreported in the main text.** M directly governs the estimation variance and the total compute cost of the method. Its absence makes it impossible for readers to independently assess estimation reliability or fairly compare the method's compute to baselines. This should be stated alongside the fine-tuning step counts in Section 4.2.

- **Theoretical propositions are asymptotic with unquantified constants.** Propositions 1 and 2 require convexity and Lipschitz-continuity assumptions that do not hold for diffusion training objectives — the paper acknowledges this is a "standard setting for analysis." More importantly, the bounds converge to constants B and C that are never quantified. For ArtBench with n=258, the Shapley error bound of 2√258·C ≈ 32C is uninformative without knowing C. The theory is motivational rather than rigorous; the empirical validation in Figure 2 and Appendix D does the real work. The paper should be more upfront about this limitation rather than presenting the propositions as theoretical justification.

### Trivial
None beyond the unquantified constants issue above.

---

## Nice-to-Haves

- A convergence plot showing how Shapley value estimates stabilize as M (KernelSHAP samples) increases, particularly for ArtBench with n=258 contributors where coverage is sparsest.
- Explicitly stating the total compute cost of the LDS evaluation (the 300 ground-truth models per dataset) to help readers understand the full experimental cost, separate from the method's cost.
- A brief ablation on pruning ratio vs. attribution quality tradeoff for CelebA-HQ, to understand whether the 26% LDS gap is addressable by tuning sparsification depth.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic: LDS Ground Truth Circularity (fatal claim)]** — The critic speculates that the 300–900 LDS ground-truth models "may have been" computed via sparsified fine-tuning (which would be circular), but provides no evidence for this. Definition 2 of the paper explicitly states "where θ*_{S_b} denotes a model **retrained from scratch** with the contributor subset S_b." Absent evidence of deviation from this stated protocol, this is speculation, not a documented flaw. Retained only as a nice-to-have (authors should report total evaluation compute for transparency). **Removed as a fatal/major weakness.**

- **[Strength Finder: Theoretical grounding (Propositions 1–2) as a key strength]** — Conflicted with the verified weakness that the bounds are asymptotic, the convexity assumption is violated, and the constants B and C are unquantified. The theory is present but provides no actionable guarantees. **Removed as a substantive strength.**

---

## Novel Insights

The paper's most genuinely novel observation — that aggregating per-datum TRAK/D-TRAK/influence scores across a contributor's data is systematically insufficient for global property attribution, often producing negative correlations — is an important negative result in its own right. The distinction between local and global attribution is not just terminological: the experiments show that methods optimized for local attribution actively fail at global attribution, while Shapley values with marginal subset contributions succeed. This suggests that global property attribution requires fundamentally different methodology, not just aggregation of existing local scores. The ArtBench setting (LoRA fine-tuning on a medium-sized dataset with application-relevant metrics) is a useful and underexplored testbed for future attribution work.

---

## Suggestions

1. Add TRAK, LOO, and influence functions to Figure 2's budget axis (converting gradient computation time to the same units as fine-tuning time). This is the single most impactful revision for supporting the paper's main empirical claim.
2. Add M (KernelSHAP sample count) to the experimental setup table and report total compute for LDS evaluation.
3. Provide at least a paragraph analyzing the CelebA-HQ underperformance — a brief ablation on pruning ratio or KernelSHAP samples for that dataset would suffice.
4. Reframe Propositions 1–2 as motivation/intuition rather than formal guarantees, noting explicitly that constants B and C are not quantified.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison to Paper Under Review |
|------|----------------|----------------------------------|
| `vKViCoKGcB.md` — D-TRAK (Intriguing Properties) | **6.0** (Accept, poster) | Most topically similar; empirically strong on same benchmarks but no new problem framing, theoretically unjustified heuristics. Paper under review has stronger novelty. |
| `kuutidLf6R.md` — Diffusion Attribution Score (DAS) | **7.5** (Accept, spotlight) | Similar topic; stronger theoretical grounding + state-of-art LDS, but also focuses on local per-datum attribution. Paper under review novelty is complementary rather than superior. |
| `HD6bWcj87Y.md` — Data Shapley in One Training Run | **7.5** (Accept, oral) | Efficient Shapley for large models; strong conceptual + algorithmic contribution. Paper under review has comparable empirical strength but weaker theoretical formalism. |
| `esYrEndGsr.md` — Influence Functions for Diffusion Models | **8.0** (Accept, oral) | Very strong: rigorous theory + scalable method + strong empirical results. Paper under review has weaker theory but different (and arguably more practically motivated) problem framing. |
| `uVMZgtw2pf.md` — CHG Shapley | **4.67** (Reject) | Rejected Shapley paper with poorly justified approximation and weak experiments. Paper under review is clearly superior: well-motivated, clean evaluation, strong margins. |
| `bVzLZr0S8s.md` — Action Shapley | **3.0** (Reject) | Rejected low-quality Shapley paper. Paper under review is substantially better in every dimension. |

**Positioning:** The paper has genuine novelty in problem formulation (global vs. local attribution), strong empirical results on 2/3 datasets, and practical motivation. Its main weakness — the compute-uncontrolled Table 1 — is a real concern but does not undermine the core contribution; the LOO vs. Sparsified-FT comparison already implies methodological superiority beyond raw compute. Compared to D-TRAK (6.0), this paper has stronger novelty; compared to DAS (7.5), it has weaker theory but a different (complementary) problem setting. The CelebA-HQ gap is unexplained but the method still leads all baselines there. Overall, this sits comfortably in the 6–7 range, closer to the D-TRAK anchor (poster) with a slight uplift for genuinely new problem framing. I score this at **6.5**.

**Originality:** High — first to address global property attribution for contributor groups in diffusion models.
**Importance:** High — addresses real-world data compensation and fairness questions.
**Claims vs. Support:** Mostly well-supported; the compute-fairness issue in Table 1 is the one claim that needs strengthening.
**Experimental Soundness:** Good; multi-metric, multi-dataset, with both quantitative (LDS) and counterfactual evaluation.
**Clarity:** Good; the problem is clearly defined and the experiments are described concisely.
**Value to Community:** Solid — introduces a new evaluation paradigm and demonstrates that existing attribution methods are inadequate for global property attribution.

**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>