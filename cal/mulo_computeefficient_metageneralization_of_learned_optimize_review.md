=== CALIBRATION EXAMPLE 33 ===

# Final Consolidated Review
## Summary

The paper derives Maximal Update Parametrization (µP) for two learned optimizer architectures (VeLO and small_fc_lopt), proposes a multi-width meta-training recipe for µ-parameterized learned optimizers (µLOs), and demonstrates empirically that µLOs substantially improve meta-generalization to wider unseen networks over standard parametrization baselines and per-task tuned hand-designed optimizers, while also showing unexpected empirical generalization to deeper networks and longer training horizons.

## Strengths

- **First principled derivation of µP for learned optimizer architectures.** Adapting µP to LOs is technically non-trivial because the optimizer's own output structure (magnitude, direction, tensor-level learning rate) must be correctly scaled. Propositions 4.1 and 4.2 provide architecture-specific derivations rather than a generic scaling recipe, and the update scaling in Eq. 3 is a concrete, verifiable contribution.

- **Strong empirical results on the primary axis (width).** Table 1 shows µLO_M and µVeLO_M consistently ranking 1st or 2nd across all OOD width categories (Large, XL, XXL) and at all evaluation checkpoints (1k, 3k, 5k steps). SP LOs rank 5th–6th (worst), confirming the parametrization matters dramatically. These results are across 5 task domains including ViTs and LMs.

- **Dramatic meta-training efficiency gains.** µLO_M is meta-trained for 100 GPU hours and generalizes to tasks where VeLO—trained with 4000 TPU-months—still struggles (Figures 6 and 9 of Metz et al. 2022b, cited in the introduction). This is a compelling cost-to-capability ratio that could lower barriers to entry for learned optimization research.

- **Surprising empirical findings on depth and horizon generalization.** Figure 5 shows µLOs remain stable on 16-layer networks despite being meta-trained on 3-layer MLPs; Figure 6 shows stable training for 25× longer than the meta-training unroll. These findings, while empirical only, could stimulate theoretical follow-up work on why parametrization-induced stability propagates across these axes.

## Weaknesses

### Major:

- **The contributions of µP parameterization vs. multi-width meta-training recipe are not fully isolated.** The paper's method combines two changes: (a) µP scaling and (b) meta-training on multiple widths {128, 512, 1024}. The ablation in Section 5.2.1 compares µLO_S vs. µLO_M and LO_S vs. LO_M, showing both that multi-width helps and that µP helps. However, the critical missing comparison is: *how does an SP LO meta-trained on the same multi-width distribution with carefully tuned per-width learning rates perform?* Figure 3 shows LO_M diverges at large widths, but it is unclear whether this is because SP fundamentally cannot work at large widths, or because the SP LO's own learning rate needs per-width tuning that wasn't provided. This conflation makes it hard to attribute improvements solely to µP. The paper should more clearly state that the contribution is the *combination* of µP + recipe, and discuss whether µP alone (with single-width training) already provides meaningful gains over SP with multi-width training.

- **Depth and horizon generalization claims lack both theoretical grounding and empirical validation of the proposed hypothesis.** The paper explicitly acknowledges that µP theory covers width only (Section 5 intro), yet features depth and horizon generalization prominently (abstract, Figure 5, Figure 6). The hypothesized mechanism is pre-activation stability (Section 5.2.2), but Figure 2 only validates this for *width* variations. There is no equivalent activation stability plot for deep networks or long-horizon training. Without this, the hypothesis remains speculative. A simple extension—plotting activation statistics for the 16-layer networks or the 25k-step runs—would substantially strengthen or revise the claim.

### Minor:

- **Meta-training exclusively on MLPs while claiming architecture generalization to ViTs and Transformers.** While the results on ViTs (Figure 4e) and LMs (Figure 4d) are impressive, the distribution shift is not just in width but in *operation type* (attention softmax interactions vs. pure matrix multiplication). The paper acknowledges this (Section 6), but the abstract's claim of improved "meta-generalization" could be misread as covering architecture transfer when it is an incidental empirical finding. The paper should more clearly distinguish the theoretically supported axis (width within architecture) from the empirical architecture-transfer finding.

- **Only training loss is reported throughout.** All figures and Table 1 report training loss. An optimizer that aggressively minimizes training loss but harms test performance would be problematic. While the meta-objective (Eq. 1) optimizes training loss by design, reporting test accuracy or validation loss on at least a subset of tasks would strengthen practical confidence in µLOs.

- **"Compute-Efficient" in the title refers only to meta-training cost, not inference cost.** Learned optimizers inherently have higher per-step overhead than Adam due to the LO network's forward pass per parameter. The paper does not quantify this overhead. For practitioners, the total cost of using µLOs = meta-training cost + (inference overhead × steps × tasks). Clarifying this distinction in the title or early in the introduction would prevent misinterpretation.

- **LLN alignment assumption during meta-training is not discussed.** Propositions 4.1 and 4.2 assume "optimizee's parameters and input data become aligned, leading to Law of Large Numbers (LLN) scaling." In standard µP theory, this holds at infinite width. During meta-training, the optimizer weights ω are constantly changing between meta-steps. The paper should briefly discuss whether the alignment assumption is expected to hold throughout the inner unroll at each meta-step, or whether early inner-loop steps (before alignment develops) might produce biased meta-gradients.

### Trivial:

- **Missing oracle SP AdamW baseline tuned at each test width** — The authors explicitly acknowledge this (Section 6). Given that AdamW and µAdam are tuned on width=1024 versions and µLOs still outperform them at larger widths, this is a minor gap rather than a critical flaw.

## Nice-to-Haves

- Ablation of individual µP components (initialization, pre-activation multiplier, update scaling) to identify which drives the most improvement.
- Meta-training on at least one non-MLP architecture (e.g., small Transformer) to disentangle the architecture-transfer benefit from the µP benefit.
- Comparison with CompleteP (Dey et al., 2025), which the paper mentions as concurrent work addressing depth+width transfer, to assess whether µP is the optimal parameterization for LO meta-learning.
- Evaluation on a standard L2O benchmark (e.g., Chen et al., 2022) to enable direct comparison with prior LO work beyond the custom evaluation suite.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Unfair comparison with hand-designed optimizers because AdamW/µAdam are per-task tuned."** This asymmetry *favors the baselines*, not the authors' method. If a single meta-trained µLO outperforms individually-tuned-per-task AdamW, that is a stronger result for µLOs, not a weaker one. Per hard rules, this is removed.

- **Weakness: "Scalability limited to moderate-scale networks (width 8192, transformer hidden 3072)."** The paper acknowledges this in Section 6 and the 8× width increase from meta-training to testing is substantial. Demanding evaluation at 1B+ parameter scales is scope creep for an academic paper with stated compute constraints.

- **Weakness: "Missing related works."** Per hard rules, no missing related works criticisms are included without external verification.

- **Weakness: "Formatting/style nitpicks."** Removed per hard rules.

- **Weakness: "Reproducibility concerns about undisclosed hyperparameters."** The paper provides substantial detail on meta-training configurations (Table 5), hyperparameter grids (Sections B.1, B.2), and open-sources code. Removed per hard rules.

## Novel Insights

The most striking insight emerging from the reviews is that µP's benefits may extend far beyond its theoretical guarantees. The paper shows µP stabilizes not just width transfer (theoretically expected) but also depth transfer and long-horizon training (theoretically unexpected). This suggests that µP's core mechanism—stabilizing activation magnitudes across scales—may be a more general "training stabilizer" than previously appreciated, functioning almost like an implicit regularizer for meta-learned systems. If this hypothesis is correct, it raises the tantalizing possibility that parametrization choices in meta-learning could be as impactful as architectural choices, and that the community may be underinvesting in parametrization theory relative to architecture search. However, this remains speculative until the pre-activation stability hypothesis is validated for depth and horizon axes, not just width.

## Suggestions

1. **Add activation stability plots for deep networks and long-horizon runs** (analogous to Figure 2) to validate or refute the pre-activation stability hypothesis for the empirically observed depth and horizon generalization. This is the single most impactful addition the authors could make.

2. **Report validation loss or test accuracy on at least 2–3 tasks** (e.g., ImageNet ViT, LM1B) to confirm that µLOs' aggressive training loss minimization does not harm generalization.

3. **Clearly separate the contributions of µP vs. the multi-width recipe** in the abstract and introduction. State upfront that the method combines both, and clarify that the multi-width recipe is necessary (per the ablation) but insufficient without µP.

4. **Add a brief discussion of inference overhead**, even approximate (e.g., "the LO forward pass adds ~X% per-step overhead compared to Adam"), to make the "compute-efficient" claim precise.

5. **Discuss the validity of the LLN alignment assumption during early inner-loop steps** of meta-training, when the optimizee has not yet developed feature learning. A one-paragraph discussion would suffice.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
