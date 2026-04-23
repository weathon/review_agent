Now I have all the information needed. Let me compose the final review.

## Summary

TTVD proposes a geometric framework for test-time adaptation (TTA) that reformulates neighbor-based TTA methods through the lens of Voronoi Diagrams and extends this foundation via two generalizations: Cluster-induced Voronoi Diagram (CIVD), which incorporates rotation-augmented prototypes through a multi-source influence mechanism, and Power Diagram (PD), which introduces classifier-derived weights to shift partition boundaries for noisy sample filtering. Evaluated under the standardized TTAB toolkit across CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R, TTVD achieves consistent error reductions (0.7–1.6%) and notable calibration improvements (ECE reductions of 1.8–4.3%) over prior TTA methods.

## Strengths

- **Standardized, rigorous evaluation**: The use of the peer-reviewed TTAB toolkit (Section 4) with grid-searched hyperparameters for all methods ensures credible comparisons. This addresses a real reproducibility problem in the TTA literature.

- **Consistent empirical improvements**: TTVD achieves the lowest error and ECE across all four datasets (Table 1). The ECE reductions (3.4%, 1.8%, 4.1%, 4.3%) are substantial and address an underexplored dimension in TTA.

- **Progressive ablation validates each component**: Table 2 shows meaningful gains at each step: VD (28.4%) → CIVD (22.7%, −5.7%) → CIPD (20.5%, −2.2%) on CIFAR-10-C, confirming that both geometric generalizations contribute.

- **Interpretability through geometric visualization**: Figures 1 and 2 provide qualitative insight into how different diagram structures affect feature-space partitioning and entropy landscapes, which most TTA papers lack.

- **Practical robustness**: Table 4 demonstrates that TTVD's performance is virtually unchanged (59.8% → 59.9%) when using only 1% vs. 10% of ImageNet training data for class mean computation, making the method practical for large-scale datasets.

- **PD–VD subtraction as a noise-filtering principle**: The idea of using the difference between weighted (PD) and unweighted (VD) partition boundaries to identify boundary-adjacent noisy samples (Section 3.3, Figure 2b) is a genuine geometric insight that could inspire further work.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed novelty of the VD connection**: The paper frames the alignment between neighbor-based TTA and Voronoi Diagrams as a key "revelation" (Abstract: "we first reveal that the underlying structure of neighbor-based methods aligns with the Voronoi Diagram"). However, this alignment is a definitional equivalence: VD partitions space so each point belongs to its nearest site, which is exactly what nearest-prototype classification does. The paper's genuine algorithmic contributions (CIVD's multi-source influence from augmented views; PD's classifier-derived weights for boundary filtering) are reasonable extensions, but they are motivated and derived through well-known geometric concepts rather than unlocked by a new discovery. The overclaim weakens the paper's novelty narrative without undermining the empirical contributions.

- **Overclaimed "unification" of self-supervision and entropy minimization via CIVD**: The introduction motivates CIVD by claiming it "unifies" self-supervision and entropy minimization to resolve conflicting gradients (Challenge II). However, CIVD merely provides augmented reference points (rotation-augmented prototypes) within a single entropy-on-distance loss (Equation 3 applied to CIVD's influence function). There is no multi-objective optimization and no mechanism for reconciling conflicting gradient directions. The claim that this "avoids negative transfer" (Section 3.2) is unsupported—no gradient conflict analysis or comparison to jointly-optimized self-supervision + entropy is provided. CIVD should be described more accurately as incorporating augmented views as additional reference points for distance-based entropy minimization, not as a "unification."

- **Ablation limited to CIFAR-10-C**: Table 2 validates the VD → CIVD → CIPD progression only on CIFAR-10-C. The headline claims of 5.7% CIVD gain and 2.2% CIPD gain are not validated on CIFAR-100-C, ImageNet-C, or ImageNet-R—the datasets where the absolute error differences matter most. Without this, it is unclear whether the component-level contributions generalize beyond the smallest benchmark.

### Minor

- **Parameter update scope unclear**: The paper states that "only the channel-wise affine parameters in normalization layers are updated during TTA" (line 81), but Algorithm 1 writes σ_{t+1} = σ_t − λ∇L_VD, suggesting the entire feature extractor may be updated. The paper does not clarify whether TTVD updates only BN parameters (like Tent/SAR) or more. If more parameters are updated than in baselines, the comparison could be unfair. Clarifying this in the main text or algorithm specification would strengthen the paper.

- **PD noise filtering supported only by 2D MNIST visualization**: Figure 2a illustrates the entropy landscape on 2D MNIST data, and the claim that "noisy samples are only identifiable if they are near boundaries" is derived from this low-dimensional visualization. In high-dimensional feature spaces, boundary geometry is fundamentally different. The paper would benefit from quantitative filtering statistics or analysis in the actual feature space used for experiments.

- **Underspecified transition from CIVD influence to soft labels**: The paper states (Section 3.2) that "the soft label given by CIVD can be calculated from the influence function" but does not provide an explicit equation analogous to Equation 3 for CIVD. The reader must infer how the influence function F(z, C_k) from Equation 4 maps to soft labels and loss computation, making the CIVD contribution harder to verify.

### Trivial
None.

## Nice-to-Haves

- **Ablation on larger datasets**: Validating VD → CIVD → CIPD on ImageNet-C or ImageNet-R would substantiate the component-level claims on the more challenging benchmarks.

- **Analysis of why distance-based entropy outperforms classifier-based entropy**: The paper replaces classifier logits with distance-to-prototype logits but never analyzes why this helps. Does it provide more stable reference under distribution shift? A brief discussion or experiment would strengthen the method's rationale.

- **Sensitivity analysis on τ and γ**: These hyperparameters control the loss landscape shape. Understanding their sensitivity would help distinguish the contribution of the geometric formulation from hyperparameter tuning.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Criticisms about missing appendix content** (Algorithm 3 in Appendix H, joint label ȳ_k^(α) specification, PD-VD filtering algorithm): The parser strips appendix sections from all papers; these details exist in the original submission. Removed per rules about missing appendix criticisms.

- **Criticisms about v_k being "never specified"**: Lemma 3.1 explicitly derives v_k from classifier parameters (W, b), providing v_k² = b^k + ¼‖W^{k×ℓ}‖² and μ_k = ½W^{k×ℓ}. The claim that v_k is unspecified is incorrect.

- **SHOT comparison fairness concern**: The paper is transparent that SHOT is a repurposed DA method. Since the comparison asymmetry (if any) favors SHOT (which has access to the full target domain), this does not constitute a weakness for TTVD per the rule about unfair comparisons that favor baselines.

- **Claims that TTVD might overfit to the test stream based on adaptation curves**: The adaptation curves (Figure 4) show TTVD consistently reducing error, which the paper interprets positively. Without evidence of actual overfitting, this is speculative.

- **Missing recent TTA baselines (EATA, EcoTTA)**: Per rules, I should not flag missing related work since I cannot verify their existence or relevance.

- **Reproducibility concerns about undisclosed hyperparameters (τ, γ)**: The implementation section states hyperparameters are grid-searched following TTAB guidelines, which is standard practice in this community. Flagging unspecified hyperparameters as a reproducibility concern is a minor nitpick given the standardized evaluation protocol.

## Novel Insights

The PD–VD subtraction concept for identifying boundary-adjacent noisy samples is a genuinely novel geometric insight that goes beyond the definitional VD connection. This idea—using the structural difference between classifier-weighted and unweighted partitions to detect samples in uncertain regions—could be a useful general principle for noise filtering in other settings. However, this insight remains largely conceptual in the current paper, supported only by a 2D visualization rather than systematic high-dimensional analysis.

## Suggestions

- Describe CIVD more accurately as incorporating augmented reference points into distance-based entropy minimization, rather than "unifying" self-supervision and entropy minimization. Acknowledge that this is a single-objective formulation.

- Explicitly state in Algorithm 1 (or the main text) whether TTVD updates only BN affine parameters or the full feature extractor. If BN-only, adjust the notation σ to reflect this; if full, provide a baseline that also updates the full extractor with standard entropy loss to isolate the geometric formulation's contribution.

- Extend the ablation (Table 2) to at least one additional dataset (CIFAR-100-C or ImageNet-C) to validate the VD → CIVD → CIPD progression beyond CIFAR-10-C.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison to TTVD |
|-------|-----------|----------|-------------------|
| AEA (sEMJ1PLSZR) | 6.25 | Accept Poster | Stronger novelty in decomposition insight; comparable empirical scope |
| NGTTA (bdHjLCcMSP) | 5.50 | Reject | Similar geometry + TTA idea; TTVD has better empirical results and standardized eval |
| DART (xqxG5WogN6) | 5.67 | Reject | Comparable scope (TTA with distribution shift); TTVD has more consistent gains but similar novelty concerns |
| Intransigent Teacher (Chq4OQ3p18) | 5.25 | Reject | Simple effective idea; TTVD has broader evaluation but overclaims novelty |
| Active TTA (pdzHpQbGrn) | 2.50 | Reject | Marginal novelty, weak evaluation; TTVD is clearly stronger |
| Neural Collapse/Voronoi (TmcH09s6pT) | 7.00 | Reject (borderline) | Stronger theoretical contribution; TTVD has weaker theory but empirical novelty |
| Voronoi Compression (Tv36j85SqR) | 7.20 | Accept Spotlight | Much stronger theoretical grounding for geometric idea |

TTVD is stronger than the clearly rejected TTA papers (Active TTA at 2.5, NGTTA at 5.5) in empirical contribution and evaluation rigor. It is weaker than AEA (6.25, Accept) and the accepted geometric papers (7.0+) in novelty depth and theoretical grounding. The overclaimed novelty (VD "revelation" and CIVD "unification") and the limited ablation scope place it below the borderline accept threshold. The genuine contributions (PD-VD filtering, ECE improvements, standardized evaluation) are meaningful but insufficient to overcome the overclaimed novelty narrative and evaluation gaps.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>