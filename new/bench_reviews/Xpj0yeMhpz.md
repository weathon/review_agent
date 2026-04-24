Good, I have the full paper. Now let me do calibration searches in parallel.Now let me read a couple of the most relevant anchor papers for calibration detail.Now I have all the information I need to compose the final review.

---

## Summary

This paper decouples the class label from the target concept in machine unlearning, identifying three new label-domain-mismatch scenarios beyond the conventional "all-matched" setting: *target mismatch* (LD = LM ≺ LT), *model mismatch* (LD = LT ≺ LM), and *data mismatch* (LD ≺ LT = LM). The authors propose TARget-aware Forgetting (TARF), a three-phase framework combining annealed gradient ascent with target-aware gradient descent, grounded in a theoretical "representation gravity" argument. Experiments span CIFAR-10/100, ImageNet-1k, TOFU (LLMs), and Stable Diffusion, showing TARF outperforms unadapted baselines across all four settings.

---

## Strengths

- **Novel and well-formalized problem setting** (Table 1, Figure 1, Section 3.1): The paper introduces a clean taxonomy of label-domain relations (L1 = L2, L1 ≺ L2) that yields four mutually-exclusive unlearning scenarios, filling a genuine gap in the literature. The observation that real-world unlearning requests (copyright, fairness, safety) may not align with the training taxonomy is practically important.

- **Theoretically motivated framework** (Theorem 3.2, Figure 3): Representation gravity is formally connected to forgetting dynamics via an upper-bound on inter-subset loss divergence, and empirically validated across scenarios (Figure 3). While the bound is qualitative, it directly motivates the three-phase design of TARF.

- **Strong empirical performance in mismatch settings** (Table 3, Table 2): TARF achieves the lowest Gap in target mismatch CIFAR-10 (1.23 vs. next best 20.80) and significantly outperforms all baselines in mismatch scenarios. Table 2 provides fine-grained evaluation of affected vs. target-concept data, confirming genuine disentanglement.

- **Multi-scale and cross-domain validation** (Tables 4–5, Figure 6): Results on ImageNet-1k, TOFU (LLaMA 3.2 1B and 8B), and Stable Diffusion concept removal show TARF can generalize beyond image classification, and the TARF-NPO variant successfully restores retain-set performance that vanilla NPO destroys (e.g., QA Prob on R: 0.00 → 0.47 in representation mismatch).

- **Ablations characterize key design choices** (Figure 7): The paper ablates annealing schedule, gradient operation type on Dfr, model architecture, and forgetting strength k, providing actionable design guidance.

---

## Weaknesses

### Fatal
None.

### Major

- **No adapted baselines for the mismatch settings.** TARF's core comparisons (Table 3) pit it against baselines applied in their original, all-matched form. However, the paper explicitly states in Section 2 that "the number of classes in Dun belonging to the target concept is known in target mismatch forgetting." TARF uses this oracle cardinality to set the β threshold in Phase I — information that is not shared with FT, GA, SCRUB, or SalUn. A minimally adapted baseline (e.g., GA applied directly to all identified target-concept classes, using the same oracle count) would separate TARF's algorithmic contribution from its information advantage. Without such a comparison, the large performance gaps in Table 3 cannot be cleanly attributed to TARF's three-phase design rather than to the asymmetric use of oracle information. This is the paper's most significant experimental gap.

- **Oracle assumption for β is only partially analyzed.** The threshold β is said to be estimated from "the number of classes in Dun belonging to the target concept" (Section 2) and set "as the lowest value of top-10% data" (Section 3.3). The paper shows sensitivity to the *number of given forgetting classes* Df in Figure 5(a)-right, but this is distinct from the cardinality of false-retaining classes. The paper does not systematically study what happens when the class-count oracle is noisy or incorrect — a practically critical case since real unlearning requests rarely come with precise target-concept membership counts. This leaves the core of Phase I's robustness unevaluated.

### Minor

- **TARF underperforms SCRUB in the conventional all-matched setting on CIFAR-100.** The paper characterizes TARF as a "general framework" for all four tasks, but in Table 3 SCRUB achieves a lower Gap than TARF in all-matched CIFAR-100 (0.71 vs. 1.11). The paper does not analyze why SCRUB's knowledge-distillation retaining objective outperforms TARF's annealed approach in the canonical scenario, nor does it discuss the implied trade-offs. This inconsistency with the "general framework" framing should be addressed.

- **MIA metric saturates in most scenarios.** In Table 3, MIA = 100.00 for both the Retrained reference and TARF in nearly every row with good forgetting. This contributes no discriminative information and inflates the apparent quality of the Gap aggregate. A more calibrated privacy metric (e.g., likelihood ratio test) would better reveal method differences.

- **Gap metric aggregates incommensurable quantities with equal weight.** The Gap is defined as the mean of |UA, RA, TA, MIA differences|. A 5-point UA gap (unlearning failure) and a 5-point RA gap (utility loss) are weighted identically, though they have qualitatively different implications. The uniform averaging is not justified, and could reverse method rankings under any principled weighting.

- **LLM and diffusion results are thin.** Table 5 covers only QA probability across four scenario-model pairs without ablations isolating TARF's contribution over the base forgetting objective (GA, NPO). Figure 6 shows four images for one concept with the full quantitative results deferred to the appendix. These are promising but insufficient to support strong claims of LLM/diffusion generalization.

- **Theorem 3.2 contains uncomputed terms.** The bound depends on λmax(Jθt), Cℓ, and Edh, none of which are estimated in practice. The bridge to the operational metric Icon(x, y, θ) is informal and justified only by "empirically supported gravity effects." The theory functions as qualitative motivation rather than a mechanistic design justification — which is fine but should be stated honestly.

### Trivial

- τ's time parameters (t0, t1) are deferred entirely to Appendix E without even a qualitative sensitivity summary in the main text. A single line noting typical ranges would help readers calibrate practical use.

---

## Nice-to-Haves

- Include at least one adapted baseline per mismatch scenario (e.g., GA/FT using the oracle class count) to disentangle algorithmic design from information advantage.
- Provide a systematic sensitivity analysis of Phase I's performance as the assumed cardinality of false-retaining classes is varied by ±25–50%.
- Add t-SNE/UMAP trajectories across the three phases to verify that Phase II achieves representation deconstruction and Phase III achieves reconstruction, as the narrative claims.
- Explore combined mismatch scenarios (e.g., simultaneous model and data mismatch), which arise naturally when generative models receive fine-grained unlearning requests.
- Analyze why SCRUB outperforms TARF in the all-matched CIFAR-100 setting; this would clarify TARF's design trade-offs.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

1. **"Baselines are deliberately operating blind"** (full framing of Harsh Critic point 1 as a "structural flaw"): While the lack of adapted baselines is a real concern (kept as Major above), the framing that the comparison is fundamentally unfair overstates the case. In papers introducing new problem settings, applying existing methods in their original form is standard practice to demonstrate the motivation for the new setting. The framing that TARF is compared "to baselines that are deliberately operating blind" is too strong — it is reduced to a request for adapted baselines as an additional experiment.

2. **"Combining mismatch scenarios as an obvious next step"**: This is scope creep. The paper introduces four scenarios; asking it to also study all pairwise combinations is outside the stated scope. Moved to nice-to-haves.

3. **"Feature-space trajectories across phases" as missing validation**: The paper's Figure 3 and Figure 9 (appendix) provide representation-distance and class-accuracy trends. The request for animated t-SNE sequences is a nice-to-have, not a weakness.

4. **"Beyond classification: quantitative LLM results" as missing ablations isolating TARF's contribution over runtime and data selection**: The LLM experiments show clear improvements in retain QA probability where baselines catastrophically forget; the claim is not that TARF dominates on LLMs but that it extends there. The request for granular ablations controlling for runtime is outside the paper's scope for these case-study sections.

---

## Novel Insights

The paper's most genuinely novel analytical insight is the representation-gravity framing: by showing that the magnitude of forgetting-induced loss change between two data subsets is upper-bounded by their representation distance (Theorem 3.2), the paper provides a unified explanation for why *all* existing unlearning methods fail in mismatch settings — methods optimized for aligned labels inadvertently either under-affect or over-affect data in proportion to its representational proximity to the forgetting set. This framing reframes the mismatch problem not as a labeling artifact but as a geometry-of-representations problem, opening a path to representation-aware unlearning that existing influence-function or gradient-ascent methods do not exploit. The three-phase TARF design — identification → separation → reconstruction — then maps cleanly onto three qualitatively different regimes of representation entanglement, which is a clean conceptual contribution that could stimulate further work on representation-geometry-driven unlearning.

---

## Suggestions

1. **Include one adapted baseline per mismatch scenario** using the same oracle information available to TARF — this single change would substantially strengthen the paper's central experimental claim.
2. **Report β sensitivity** (varying the assumed false-retaining class count ±25–50%) in the main paper, since this assumption drives Phase I and its robustness is core to the practical claim.
3. **Qualify the "general framework" characterization** by acknowledging the all-matched CIFAR-100 result where SCRUB dominates; discuss when TARF's design trade-offs are worth the mild conventional-setting cost.
4. **Replace or supplement MIA** with a more discriminative privacy metric (e.g., likelihood ratio–based membership inference) for scenarios where MIA saturates at 100.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Human Score | Comparison |
|---|---|---|
| gn0mIhQGNM (SalUn) | 7.50 | Broader baselines, more rigorous diffusion evals, modular — notably stronger empirically than the paper under review |
| HVFMooKrHX (Unlearning utility-complexity) | 6.60 | Rigorous theory accepted; theory is much tighter but experiments are weaker; the paper under review has stronger empirical coverage |
| iQIQT88prm (Adversarial MU, Stackelberg) | 5.33 | Rejected borderline — similar ambition but narrower scope than paper under review |
| TLBPjECC5D (Zero-shot sparse unlearning) | 5.25 | Rejected — limited novelty, model-specific; paper under review is clearly more novel and general |
| bIoWuzFm6r (Streaming forgetting) | 4.75 | Rejected — limited experiments, theoretical gaps; paper under review is stronger across all dimensions |
| ZyMXxpBfct (Catastrophic forgetting explanation) | 1.50 | Strong reject — trivial results; paper under review is clearly superior |

**Positioning:** The paper under review is comfortably above the 4.75–5.33 rejected borderline papers: it introduces a genuinely new problem formulation, has multi-scale experiments, and a theoretically motivated framework. However, it falls short of the 6.60+ tier due to (a) the lack of adapted baselines that disentangle algorithmic contribution from oracle-information advantage, and (b) the unanalyzed β sensitivity. SalUn (7.50) is the most topically similar strong paper; TARF's problem formulation is arguably more novel, but its experimental rigor is lower (fewer baselines, no adapted comparisons, thin LLM/diffusion ablations). The paper sits in the 5.5–6.0 range, which I assign a score of **5.5**.

**Evaluation axes:**
- *Originality*: High — new problem formulation with clean taxonomy is a genuine contribution.
- *Importance of research question*: High — label-domain mismatch in unlearning is practically relevant.
- *Claims well-supported*: Partially — strong in mismatch settings, weakened by unadapted baseline comparisons and oracle assumption.
- *Soundness of experiments*: Moderate — multi-scale, but comparison fairness is a real methodological gap.
- *Clarity of writing*: Good — problem setup is clear; hyperparameter guidance is underdeveloped in the main text.
- *Value to research community*: Moderate-high — the new settings and theoretical framing are useful; the method needs cleaner validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>