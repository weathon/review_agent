Now I have all the information I need. Let me write the consolidated review.

## Summary

The paper proposes TTVD, a geometric framework for test-time adaptation (TTA) that reformulates neighbor-based TTA through Voronoi Diagrams (VD), extends it to Cluster-induced Voronoi Diagrams (CIVD) that unify self-supervision and entropy minimization via a multi-site influence mechanism, and further to Power Diagrams (PD) that enable geometric noisy-sample filtering. The progressive VD→CIVD→CIPD pipeline yields monotonic error reductions across all 15 corruption types on CIFAR-10-C, and TTVD achieves the best error and ECE across four benchmark datasets under standardized TTAB evaluation.

## Strengths

- **Progressive ablation validates each geometric extension**: Table 2 shows VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) with consistent improvement on every single corruption type (all 15). This is the strongest piece of evidence supporting the core claim that geometric structure upgrades systematically improve TTA.

- **Substantial ECE improvements**: TTVD achieves 21.0% ECE on ImageNet-C vs. SAR's 38.4% and 11.8% on CIFAR-10-C vs. TTT's 15.2% (Table 1). These calibration improvements are practically significant for deployment trustworthiness and are not trivially obtained—centroid-distance soft labels inherently regularize confidence, but the magnitude of the gap across all four datasets is notable.

- **Rigorous evaluation protocol**: Using TTAB (a peer-reviewed, open-source toolkit) for fair hyperparameter tuning and comparison, and reporting both error and ECE across batch size and label shift scenarios, provides confidence in the fairness of the comparisons.

- **Adaptation curves show genuine continued learning**: Figure 4 demonstrates that TTVD continues decreasing error across 750 online batches while Tent and SAR plateau early, particularly on impulse noise and defocus blur, suggesting the method avoids premature convergence to suboptimal points.

- **Large gains over neighbor-based baselines**: Table 3 shows TTVD achieves 53.2% on zoom blur vs. AdaNPC's 60.6%, and dominates across all four blur types on ImageNet-C, validating that the geometric mechanism significantly improves over the nearest-neighbor paradigm.

## Weaknesses

### Fatal
None.

### Major

- **Core method formulations underspecified in main text**: The CIVD loss is never explicitly written out. After introducing the influence function F(z, C_k) in Equation 4, the paper states only that "Similar to Equation 3, the soft label given by CIVD can be calculated from the influence function" (Section 3.2). Equation 3 applies to single-site VD—how the multi-site influence translates into a concrete soft-label formula and loss is left implicit. Similarly, the CIPD algorithm is deferred entirely to "Algorithm 3 in Appendix H" and the PD-based filtering criterion ("subtracting the PD from the VD") is described only in prose without a formal decision rule (Section 3.3). Without these specifications, the paper's central mechanism cannot be fully evaluated from the main text, and readers must trust that the appendix fills these gaps correctly.

- **Unjustified claim that CIVD avoids negative transfer**: Section 3.2 asserts that CIVD's "joint label avoids the negative transfer since the objective is now unified." This is a strong claim, especially given that Gandelsman et al. (2022)—cited by the authors—demonstrated that jointly training self-supervision and entropy minimization can degrade accuracy due to conflicting gradients. The CIVD mechanism aggregates multi-site distances through an influence function, but it is never established why this particular aggregation would resolve gradient conflict rather than merely average opposing signals. No gradient analysis, theoretical argument, or experimental evidence (e.g., gradient cosine similarity statistics) is provided to support this claim.

- **PD-based filtering lacks formal specification and direct empirical validation**: The filtering mechanism (Section 3.3) identifies noisy samples based on VD–PD boundary disagreement, but the precise criterion—e.g., a threshold on region membership differences or a geometric condition—is never formalized. Moreover, there is no ablation directly comparing TTVD with PD filtering disabled vs. TTVD with entropy-based filtering (as in SAR). Without this isolation, it is unclear whether the PD subtraction adds value over the simpler entropy threshold it claims to improve upon.

### Minor

- **Voronoi Diagram observation is trivially true**: Section 3.1 frames the observation that neighbor-based TTA "aligns with the Voronoi Diagram" as a revelation, but nearest-centroid classification partitioning space into Voronoi cells is the definition of a Voronoi Diagram. The VD loss (Eq. 3) is softmax entropy minimization on negative L2 distances to class means—standard prototype classification with temperature-scaled entropy. The insight value lies in the subsequent extensions (CIVD, PD), not in this foundational observation itself.

- **Voronoi site precision barely affects performance**: Table 4 shows TTVD error remains at 59.8–59.9% whether class means are computed from 10%, 5%, or 1% of ImageNet training data. While the authors present this as robustness, it also suggests that the geometric precision of Voronoi sites is not driving the method's performance—raising the question of what aspect of the geometric framework is actually essential.

- **Overclaimed "remarkable" improvements**: The abstract claims "remarkable improvements," but error rate gains are 0.7–1.6% over the strongest baselines. The ECE gains are substantial, but the paper centers its narrative on error rate. More measured language would better reflect the actual contribution profile.

### Trivial
None.

## Nice-to-Haves

- An ablation replacing L2-distance-to-centroid soft labels (Eq. 3) with the model's own logits under the same entropy minimization would clarify whether the geometric framework or simply prototype-based soft labels drive the gains.
- A visualization of which samples PD filtering removes and how they differ from what entropy-based filtering would remove, to concretely demonstrate the geometric advantage.
- Gradient analysis (e.g., cosine similarity between self-supervision and entropy loss gradients under CIVD vs. separate training) to substantiate the negative-transfer-avoidance claim.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "The paper does not discuss whether centroids are fixed or updated during adaptation."** The paper is clear that only the feature extractor σ is updated via gradient descent (Eq. 1, Algorithm 1), while the Voronoi sites μ are precomputed from training data. This is stated in Section 4 Implementation Details.

- **Harsh critic: "TTT is essentially embedded in geometric language" regarding rotation augmentation.** This conflates the mechanism—rotation augmentation in CIVD creates multi-site clusters that produce a *joint* soft label combining rotation-based information with geometric partition assignment, which is structurally different from TTT's separate auxiliary classifier loss. Whether this is a meaningful distinction is debatable, but calling it "essentially TTT" dismisses the joint-label unification mechanism that is the core claim.

- **Harsh critic: "Baselines T3A and TAST have high error rates (>74%)" implying unfair comparison.** These are neighbor-based methods included for completeness; the stronger baselines (SAR, Tent, SHOT) are the relevant comparisons, where gaps are smaller. Including weak baselines is standard practice to show the full landscape.

- **Harsh critic: "Figure 2a shows entropy is high near boundaries, validating entropy-based filtering rather than undermining it."** The paper actually argues in Section 3.3 that entropy is high near boundaries (confirming entropy can identify boundary samples) but that "noisy samples are only identifiable if they are near the boundaries, leaving many noisy samples undetected." The claim is that PD-VD disagreement captures a *broader* region of noisy samples than just boundary-adjacent ones. Whether this actually works is the legitimate concern (no ablation), but the critic mischaracterizes the paper's argument.

- **Strength finder: "Consistent state-of-the-art across all benchmarks"**: The error rate improvements over the strongest baselines are 0.7–1.6%, which is consistent but modest—overstating this as "meaningful margins" is not supported. ECE improvements are more substantial, but the strength finder treats both metrics equivalently.

## Novel Insights

The paper's most interesting observation is that the PD structure derived from the model's linear classifier (Lemma 3.1) and the VD structure from training-data centroids produce *different* partition boundaries, and the regions of disagreement between these two structures may identify unreliable samples for adaptation. This dual-diagram comparison is a genuinely novel way to think about sample filtering—instead of relying on prediction confidence alone, it compares two geometrically principled partitionings (one from the model's parameters, one from the data distribution) to identify unstable regions. However, the potential of this insight remains unrealized without formal analysis proving that PD-VD disagreement identifies *different* noisy samples than entropy thresholds.

## Suggestions

- Provide explicit loss functions for CIVD and CIPD directly in the main text (even a single equation each), rather than deferring to "Similar to Equation 3" and an appendix algorithm.
- Add the PD-filtering vs. entropy-filtering ablation to substantiate the core claim about the geometric approach's superiority for noisy sample identification.
- Report gradient statistics (e.g., cosine similarity) between the self-supervision and entropy objectives under CIVD to support the negative-transfer-avoidance claim, or soften the claim accordingly.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| TPZRq4FALB (Multi-modal TTA with reliability bias) | 8.0 | Higher: novel problem + new benchmarks + effective method, clear contributions beyond reframing |
| 9w3iw8wDuE (DeYO: entropy not enough for TTA) | 7.0 | Higher: genuine new insight (entropy unreliable due to spurious features) + practical new metric, well-justified |
| bdHjLCcMSP (NGTTA: geometry-driven TTA for 3D) | 5.5 | Comparable: also uses geometric features for TTA, also has underspecification issues, but weaker experiments |
| G4D6jClNFl (Curved-space contrastive for deepfakes) | 4.75 | Comparable: also wraps existing methods in geometric framing with modest novelty; TTVD is somewhat stronger due to progressive ablation |
| Chq4OQ3p18 (Intransigent teacher for TTA) | 5.25 | Comparable: simple but principled idea, similar tier of contribution |
| Oi6BhzIu7R (REAL: max-min entropy TTA defense) | 4.67 | Comparable: challenges entropy minimization, also has novelty concerns |
| pdzHpQbGrn (Active test-time prompt learning) | 2.5 | Lower: fundamentally flawed methodology, TTVD is clearly above this |

TTVD sits in the 4.5–5.5 range. It outperforms the geometric-reframing-with-modest-novelty anchor (G4D6jClNFl, 4.75) because its progressive ablation is genuinely strong, ECE gains are substantial, and CIVD's multi-site mechanism adds more than just a relabeling. However, it falls short of the DeYO anchor (7.0) because that paper identified a genuine, well-justified limitation with clear experimental support, whereas TTVD's core claims about negative transfer avoidance and PD-filtering superiority remain unsupported. The underspecification of core algorithms in the main text is a presentation barrier that prevents full evaluation. The strongest comparison is with NGTTA (5.5), which similarly uses geometric features for TTA with partial underspecification—TTVD's evaluation is more thorough, but its core claims are less well-justified.

MY FINAL SCORE: 5.0
MY FINAL DECISION: <orange>Reject</orange>