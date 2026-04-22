Now I have a thorough understanding of the paper and the calibration anchors. Let me produce the final review.

## Summary

The paper proposes TTVD, a test-time adaptation framework that reformulates prototype-based TTA using Voronoi Diagram terminology. Starting from a basic VD loss (softmax over negative distances to class means, then entropy minimization), it extends to Cluster-Induced Voronoi Diagrams (CIVD, using rotation-augmented prototypes with a multi-site influence function) and Cluster-Induced Power Diagrams (CIPD, adding per-class weights and a boundary-based noisy sample filtering mechanism). Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R using the TTAB benchmark show consistent error reductions and notably large ECE improvements over prior methods.

## Strengths

- **Consistent empirical improvements across all four benchmarks**: Table 1 shows TTVD achieves the lowest error and ECE on all datasets, with error reductions of 0.8%, 0.7%, 1.6%, and 0.7% over the next-best method. The cross-dataset consistency is notable and not a given in TTA research.

- **Substantial ECE improvements**: On ImageNet-C, TTVD achieves 21.0% ECE vs. 38.4% for the next-best entropy method (SAR), a near-halving of calibration error. This is a practically important finding—prototype-anchored predictions are better calibrated—and deserves deeper analysis.

- **Principled stepwise ablation (Table 2)**: VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) on CIFAR-10-C shows progressive improvement across all 15 corruption types, validating that each component adds value. The 5.7% gain from CIVD over VD is substantial.

- **Robustness to prototype precision (Table 4)**: Using 1% vs. 10% of ImageNet for computing class means yields nearly identical performance (59.9% vs. 59.8%), making the approach practical for large-scale settings.

- **Evaluation under batch size and label shift variations** (Appendix B) addresses two practically important deployment scenarios that are often neglected.

## Weaknesses

### Fatal
None.

### Major

- **The "geometric framework" framing provides algorithmic content no different from prototype-based classification**: The VD loss (Eq. 3) is softmax over negative distances to class means followed by entropy minimization—functionally identical to nearest-centroid classification plus Tent. CIVD (Eq. 4) computes a sum of powered distances to augmented prototypes—this is prototype augmentation, not a Voronoi-specific algorithm. PD (Eq. 5) is weighted nearest-centroid matching, which Lemma 3.1 itself shows is equivalent to what a linear classifier already performs. No Voronoi-specific algorithm (boundary computation, Delaunay duality, explicit cell intersection) is ever used. The paper's central claimed contribution—"we revisit the TTA problem from geometric view and formulate it using Voronoi Diagram"—adds no operations or insights that would not arise from standard prototype-based reasoning. This disconnect between the claimed contribution level (geometric framework) and the actual method (augmented-prototype entropy minimization) inflates the paper's perceived novelty. The method works, but the VD/CIVD/PD framing is a relabeling, not a foundation.

- **The claim that CIVD "unifies self-supervision and entropy minimization" is misleading**: The paper repeatedly states (§1, §3.2, contributions list) that CIVD provides a "joint influence mechanism" that "unifies multiple objectives" and enables "seamless integration of self-supervision and entropy minimization." In reality, rotation augmentation creates four prototype sites per class at precomputation time, and only one objective—entropy minimization on prototype-based predictions—is optimized at test time. There is no rotation-prediction head, no self-supervised loss signal, and no optimization of multiple objectives. The rotation-augmented prototypes improve robustness via data augmentation, not via self-supervision. The paper states (§3.2): "the joint label $\tilde{y}_k^{(\alpha)}$ avoids the negative transfer since the objective is now unified"—but no unification occurs because only one objective was ever optimized. This is not presentation; it is a substantive overclaim about the method's mechanism.

- **The PD-based noisy sample filtering mechanism is under-specified and un-isolated**: The paper proposes filtering "by subtracting the PD from the VD" (§3.3) but never formally defines this subtraction operation, specifies a threshold for exclusion, or provides a concrete algorithm. The ablation in Table 2 compares CIVD vs. CIPD, but this confounds two changes: (a) introducing per-cell weights (which changes predictions for all samples) and (b) boundary-based filtering (which excludes some samples). Without isolating the filtering mechanism (e.g., CIPD with and without sample exclusion), the claim that PD-based filtering improves robustness is unsupported by evidence.

- **Potential unfair backbone comparison due to label augmentation during training**: The paper states (§4.1) that TTVD's backbones are trained "using label augmentation (Lee et al., 2020)," which creates rotation-consistent features during training. It is unclear whether baseline methods use the same augmented backbone or the standard TTAB-provided models. If TTVD's backbone is already partially rotation-invariant, this systematically advantages its rotation-augmented prototypes—a confound the paper does not address. This could be clarified in rebuttal, but is currently a gap.

### Minor

- **The $\gamma$ hyperparameter in the influence function (Eq. 4, Eq. 6) is unexplored**: The paper introduces $\text{sign}(\gamma)$ and $(d)^\gamma$ in the influence function but never analyzes what values of $\gamma$ are used, how performance depends on them, or why this aggregation is preferred over alternatives (e.g., mean distance or minimum distance). This is a design choice presented without justification.

- **Ablation only on CIFAR-10-C**: Table 2 provides stepwise ablation only on CIFAR-10-C. The contribution of CIVD and CIPD may differ qualitatively on higher-dimensional feature spaces (ImageNet-C, CIFAR-100-C), where prototype augmentation and weighted filtering could have different effects.

- **Figure 1 uses a potentially misleading toy setting**: The MNIST-C visualization (Augmented VD: 98.54% vs. T3A: 58.43%) is a 3-class example in $\mathbb{R}^2$. These dramatic numbers will not generalize to actual benchmarks and create a misleading first impression of the method's advantage.

- **No analysis of why ECE improves so substantially**: The near-halving of calibration error is the most striking result but receives no mechanistic analysis. Whether it arises because distance-to-prototype predictions are inherently better calibrated than softmax-over-logits, or because the adaptation mechanism preserves calibration better, is unexplored.

### Trivial
None.

## Nice-to-Haves

- Ablation on ImageNet-C or CIFAR-100-C to validate that geometric extensions help at scale
- Isolating the PD-based filtering mechanism (CIPD with/without exclusion) to substantiate the filtering claim
- Running at least one strong baseline with the same label-augmented backbone to isolate the contribution of the adaptation method from the training procedure
- Ablation over $\gamma$ values with justification for the chosen setting
- Analysis of whether continuously decreasing adaptation curves (Figure 4) reflect genuine adaptation or exploitation of the stationary corruption distribution within a test sequence

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"VD loss is functionally identical to T3A + Tent"** (from Harsh Critic): While the VD loss shares the prototype-based prediction structure of T3A, T3A updates prototypes online from test data while TTVD uses fixed training-time class means. The anchoring to fixed means is a legitimate design difference, not merely "T3A + Tent." The harsh critic's phrasing oversimplifies this distinction. However, the point that VD adds nothing *algorithmically* beyond nearest-centroid reasoning remains valid as a Major weakness.

- **"Figure 1 numbers are cherry-picked"** (from Harsh Critic): Retained as a Minor weakness (misleading toy setting) but the term "cherry-picked" implies deliberate deception. The figure caption does note it is a 3-class example in $\mathbb{R}^2$.

- **"Lemma 3.1 is never used and is theoretical decoration"** (from Harsh Critic): The lemma establishes a link between linear classifiers and PD that provides context for why PD weights might help, even if no methodological consequence is directly derived from it. Downgraded from the critic's framing.

- **"Adaptation curves could indicate overfitting"** (from Harsh Critic): An alternative explanation worth raising but not as a Major weakness—the paper does note that TENT/SAR don't overfit under these settings, and the continuous improvement under a single corruption is expected behavior for prototype alignment. Listed as a nice-to-have.

- **"Missing comparison with augmented backbone"** (from Harsh Critic): This is a valid concern but could be addressed in rebuttal, so it stays as Major not Fatal. The critic treated it more definitively than warranted.

- **Strength Finder's "unified treatment of self-supervision and entropy minimization"**: Removed as a strength because this claim is itself a verified weakness. A strength and weakness disagree—the weakness wins.

- **Strength Finder's "insightful noisy sample filtering via diagram subtraction"**: Removed as a strength because the filtering mechanism is under-specified and un-isolated experimentally, making this claim unsupported rather than an established strength.

## Novel Insights

The most interesting observation that emerges from reviewing this paper is that the ECE improvements are arguably a more significant contribution than the geometric framework itself. Prototype-anchored TTA produces dramatically better-calibrated predictions than logit-based entropy minimization (21.0% vs. 38.4% ECE on ImageNet-C). This suggests that constraining the adaptation loss to operate in a distance-to-prototype space—rather than directly on softmax logits—has a regularization effect on confidence estimation. This insight might generalize beyond VD/CIVD/PD: any method that grounds its loss in a geometry relative to fixed reference points (rather than the model's own output distribution) may inherit this calibration benefit. The paper does not explore this, but it could be the more consequential finding.

## Suggestions

- Reframe the contribution around the actual algorithmic innovations (augmented prototypes with multi-site influence, weighted prototype matching with geometric filtering) rather than claiming the Voronoi Diagram framework itself as the primary contribution. The empirical results are strong enough to stand on their own.
- Provide a concrete algorithmic specification of the "diagram subtraction" filtering procedure, including how to compute the excluded region and what threshold determines exclusion, and add an ablation isolating filtering from weighting.
- Clarify whether baseline methods use the same label-augmented backbone; if not, add at least one comparison with a label-augmented baseline to isolate the contribution of the adaptation method.

## Score and Decision

**Calibration anchors used:**

- **High-band (>7):**
  - `/home/wg25r/review_agent/human_reviews/9w3iw8wDuE.md`: "Entropy is not Enough for TTA" (DeYO), avg 7.0, Accept spotlight. Directly on TTA with entropy minimization; had missing baselines and some overclaiming but a clearer conceptual contribution (PLPD metric for sample selection). TTVD has weaker conceptual novelty but comparable empirical strength.
  - `/home/wg25r/review_agent/human_reviews/TPZRq4FALB.md`: Multi-modal TTA, avg 8.0, Accept poster. Stronger novelty and complete evaluation. TTVD is below this.

- **Medium-band (4-6):**
  - `/home/wg25r/review_agent/human_reviews/eXrUdcxfCw.md`: Prototype-based CTA with EMA, avg 4.80, Reject. Limited technical novelty (EMA on prototypes), minor empirical gains (<0.5%). TTVD has notably stronger empirical results (1.6% on ImageNet-C) and a more developed method, placing it above this anchor.
  - `/home/wg25r/review_agent/human_reviews/KNtcoAM5Gy.md`: BaFTA (backprop-free TTA for VLMs), avg 5.50, Reject. Simple prototype-centric method with limited novelty. TTVD has more components but similar novelty concerns; ECE results are a distinguishing factor.
  - `/home/wg25r/review_agent/human_reviews/jzzEHTBFOT.md`: Calibrated TTA for CLIP, avg 6.00, Accept poster. Addresses calibration in TTA like TTVD but with clearer novelty. TTVD is below this.

- **Low-band (<3):**
  - `/home/wg25r/review_agent/human_reviews/a9xZqOqzEW.md`: Spider diagram fairness framework, avg 2.50, Reject. Decorative theoretical framework with simple ideas beneath. TTVD shares the "decorative framework" concern but has genuinely strong empirical results, placing it well above this.
  - `/home/wg25r/review_agent/human_reviews/OXIIFZqiiN.md`: IGCP dual-modal framework, avg 1.50, Reject. Nonsensical theory wrapping a weak method. TTVD is far above this.

TTVD sits above the prototype-based CTA paper (4.8) because of much stronger empirical results and the ECE contribution, but below BaFTA/Calibrated TTA (5.5-6.0) because those papers had cleaner claims and less overclaiming. The overclaiming about the "geometric framework" and "unification" pulls it down, while the empirical results pull it up. The net position is in the borderline range.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>