Now I have a thorough understanding of the paper. Let me synthesize the review.

## Summary

This paper develops a theory of compositional generalization in kernel models with compositionally structured representations, proving that such models are limited to "conjunction-wise additive" computations (Theorem 4.2). This characterization yields an impossibility result for transitive equivalence, identifies two failure modes—memorization leak and shortcut bias—on solvable tasks, and derives a closed-form formula (Proposition 5.1) for how representational salience and training set size jointly distort generalization on symbolic addition. The theory is validated qualitatively in ConvNets, ResNets, and ViTs on MNIST and CIFAR-10 versions of symbolic addition and context dependence tasks.

## Strengths

- **Theorem 4.2 provides a clean, exact characterization** of what compositionally structured kernel models compute at test time. The conjunction-wise additivity decomposition is precise (not an approximation or bound), and the partition of the compositional task space into solvable and unsolvable tasks (Fig. 2d) is a real structural insight that goes beyond prior work analyzing specific tasks (Lippel et al., 2024) or infinite-component limits (Abbe et al., 2023).

- **The impossibility result for transitive equivalence** is a concrete, consequential negative result. It identifies a fundamental relational reasoning task that compositionally structured kernel models provably cannot solve, while showing that transitive *ordering* can be solved—highlighting that superficially similar tasks have fundamentally different computational requirements (Section 4.3).

- **The memorization leak characterization (Proposition 5.1)** yields a precise, parameterized formula $m = \frac{p \cdot S(1;2)}{1 + (p-2)S(1;2)}$ showing how representational geometry and training set size jointly distort compositional generalization. This goes beyond qualitative failure-mode identification to quantitative prediction.

- **Representational salience $S(k;C)$** effectively reduces $\binom{C}{k}$ possible conjunction terms to a single interpretable scalar per order $k$ (Section 5.1, Appendix B), providing a parsimonious descriptor of compositional geometry that enables concrete predictions and cross-architecture comparisons (Fig. 3).

- **Empirical validation bridges kernel theory and deep networks**: The finding that a conjunction-wise additive model is highly predictive of ConvNet/ResNet/ViT outputs (Appendix D.3), and that qualitative trends (slope increasing with $S(1;2)$ in Fig. 5c and training set size $p$ in Fig. 5d; shortcut-driven failure on CD-3 across architectures in Fig. 5e) match theoretical predictions, provides meaningful evidence that the theory captures real network behavior.

## Weaknesses

### Fatal
None.

### Major

- **No empirical test of the theory's strongest prediction (transitive equivalence failure in deep networks)**. The impossibility result for transitive equivalence (Section 4.3) is the paper's most consequential theoretical claim—it partitions the task space and motivates the entire framework. Yet the experiments only validate the theory on conjunction-wise additive tasks (symbolic addition and context dependence) where the theory predicts success (modulo memorization leak/shortcut bias). Without testing whether deep networks also fail on transitive equivalence, the paper validates only the "positive" side of the theory, leaving its strongest structural claim untested. The authors acknowledge in Section 7 that the theory is limited to kernel models, but testing the impossibility prediction in deep networks would be the most direct and important empirical validation of the theory.

- **The claim that the theory "captures the behavior of deep neural networks" (Abstract) overstates what the experiments show.** The experiments demonstrate qualitative trend agreement on two additive tasks across three architectures, but: (a) the compositional structure assumption (Definition 3.1) is unverified for deep networks using real images, (b) no quantitative comparison of predicted vs. observed slopes is provided (the paper measures effective $S(1;2)$ in Fig. 5a and could in principle compare predicted vs. observed values from Proposition 5.1), and (c) the spatial-distance manipulation in Fig. 5a may confound salience changes with feature quality changes. The Discussion appropriately acknowledges the absence of quantitative bounds, but the Abstract's framing is stronger than the evidence warrants. The gap between "the theory predicts correct qualitative trends" and "the theory captures deep network behavior" is significant.

### Minor

- **Proposition 5.1 assumes a specific training set structure** (rows/columns containing anchor set $\mathcal{W}$) and zero-mean values. The generality of the memorization leak failure mode beyond this setup is asserted but not formally established for arbitrary training set configurations.

- **The compositional structure assumption (Definition 3.1)** requires that all components and all same-component values are treated symmetrically—ruling out representations where, e.g., shape is more salient than color. The paper notes multi-hot representations satisfy this and Appendix A.5 discusses "compositionally structured in expectation," but the main theoretical results all rely on the exact version. The paper would be strengthened by even informal analysis of how deviations from exact compositional structure affect Theorem 4.2.

### Trivial
None.

## Nice-to-Haves

- Test whether feature learning (escaping the kernel regime) can overcome conjunction-wise additivity or the transitive equivalence impossibility. The paper mentions this in passing (Appendix F) but does not pursue it, despite it being the most direct path from the theory to practical implications.
- Visualize the learned conjunction functions $f_J$ in deep networks to make the conjunction-wise additive claim more concrete.
- Derive the 0%/100% accuracy cliff in context dependence theoretically rather than observing it empirically.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "No quantitative validation" claim (Issue 1)**: While the absence of quantitative prediction matching is a valid concern (moved to Major above), the harsh critic's framing is overstated. The paper does provide Proposition 5.1 with an exact formula and validates qualitative trends (slope increases with $S(1;2)$ and $p$). The issue is that quantitative slope comparisons are absent, not that there's no validation at all. Additionally, the paper itself acknowledges in the Discussion that it "does not provide any quantitative bounds," so the author awareness mitigates the severity somewhat.

- **Harsh Critic's claim that spatial distance manipulation confounds salience with feature quality (Section 6 notes)**: This is a plausible concern, but the paper controls for in-distribution performance and shows the effect is specific to compositional generalization, not general task difficulty ("in-distribution generalization is not systematically affected by distance," Section 6). The confound exists but is partially addressed.

- **Strength Finder's claim that "Proposition 4.1 grounds the theory in random networks" as a core strength**: While accurate, this is more of a supporting point than a core strength—it ensures the theory applies to at least one class of networks but doesn't directly validate the main empirical claims about trained deep networks.

## Novel Insights

The paper's most novel insight is the conjunction-wise additivity decomposition itself: the idea that compositional structure constrains kernel model generalization not through what the model can learn on training data (it can learn anything, thanks to the $f_{12}$ term), but through what conjunctions are *available at test time*. This reconceptualizes the compositional generalization problem as fundamentally a problem of data coverage—what subset of conjunctions the training set makes available—rather than one of representational capacity. The partition of the task space into conjunction-wise additive vs. non-additive tasks, and the sharp distinction between transitive ordering (solvable) and transitive equivalence (unsolvable), follows uniquely from this perspective and provides a concrete, constructive method for designing benchmark tasks that requiregoing beyond the kernel regime.

## Suggestions

- Test transitive equivalence in deep networks—even a simple experiment showing that deep networks fail on transitive equivalence as predicted would dramatically strengthen the paper's core claim.
- Compare predicted vs. observed slopes quantitatively using Proposition 5.1 with the measured $S(1;2)$ values, directly testing whether the theory is quantitatively accurate, not just directionally correct.
- Moderate the Abstract's claim from "captures the behavior of deep neural networks" to "captures qualitative trends in deep neural networks" to better match the empirical evidence.

## Calibration and Score

**Calibration anchors:**

1. **DhdqML3FdM** (avg 7.0, Accept poster): Proves impossibility results for SSMs/Transformers on function composition, with theoretical and empirical validation. This paper has a similar profile—a theory that identifies limitations (conjunction-wise additivity, impossibility for transitive equivalence) plus empirical validation—but the empirical validation here is narrower (two additive tasks only) and lacks testing of the impossibility prediction. The present paper is somewhat weaker empirically.

2. **STUGfUz8ob** (avg 7.6, Accept poster): Proves transformers but not MLPs can learn abstract relations, with both theoretical and empirical results spanning multiple task types. The theory-empire bridge is stronger there. The present paper has a more comprehensive theoretical framework but weaker empirical validation of the theory-empire gap.

3. **vt5mnLVIVo** (avg 6.0, Accept poster): Studies grokking as lazy-to-rich transition with limited theoretical results and mostly simple empirical validation. The present paper has stronger theoretical contributions but similar limitations in empirical bridge.

4. **hKMPz3wkPV** (avg 6.75, Reject): Formal theory of compositionality with good conceptual contribution but concerns about practical relevance and limited empirical support. The present paper is stronger (has concrete theorems and real experiments) but shares the theory-empire gap concern.

5. **3SJE1WLB4M** (avg 8.0, Spotlight): Rigorous kernel method generalization theory with precise formulas. Stronger theoretical precision than the present paper, but no experiments.

6. **2NwHLAffZZ** (avg 2.33, Reject): Weak correlations/NTK paper with overclaimed novelty and no experiments. Much weaker than the present paper.

7. **Njx1NjHIx4** (avg 7.5, Spotlight): Representation formation theory with theoretical and empirical contributions. Similar profile but with broader empirical validation.

**Score reasoning**: The paper makes a genuinely novel theoretical contribution—a clean characterization of what kernel models can compute on compositional tasks (Theorem 4.2), with meaningful corollaries (transitive equivalence impossibility, memorization leak, shortcut bias). The theory goes significantly beyond prior work. The empirical validation, while meaningful, has two notable gaps: no test of the impossibility prediction and only qualitative (not quantitative) prediction matching. These are significant but do not invalidate the theoretical contribution. Relative to the calibration anchors, the paper sits between the DhdqML3FdM profile (7.0, impossibility result + empirical validation) and the hKMPz3wkPV profile (6.75, formal theory with limited empirical support). The theoretical contribution here is cleaner than both, but the empirical bridge is weaker. I'd place this around 6.5.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>