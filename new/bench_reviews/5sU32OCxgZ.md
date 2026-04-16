Now I have sufficient information to write the final consolidated review. Let me carefully synthesize all the reviewer inputs, verify claims against the paper, and determine which criticisms are valid.

Key things to verify:

1. **Is TTVD just re-packaged entropy minimization?** The harsh critic claims the VD/CIVD/CIPD framework is just entropy minimization with prototype distances. Let me think about this carefully. Equation 3 defines: soft labels from distance to prototypes, then entropy minimization on those soft labels. The CIVD adds multi-site influence via rotation augmentations, and CIPD adds weights. This IS fundamentally different from standard Tent (which minimizes entropy of the model's own predictions). Here, the entropy is computed over a distance-based soft label rather than the model's direct output. This creates a different loss landscape. The geometric framing is not just window dressing—it changes the actual adaptation objective. However, the core innovation IS relatively incremental in computational terms.

2. **Is the "diagram subtraction" filtering specified?** Looking at Sec 3.3, the paper says "by subtracting the PD from the VD, we can extract a larger region from the resulting differences" and Fig 2(b) shows visual regions. But there's no explicit mathematical definition of what "subtracting diagrams" means in high dimensions, nor a clear algorithm for how samples in these difference regions are identified and filtered. The algorithm is deferred to "Appendix H" (Algorithm 3). This IS a valid criticism.

3. **Power of 7 in Eq 4**: This appears in the paper as $(d(\mu_k^{(\alpha)}, z))^7$ with the exponent being the value "7". However, looking more carefully, the paper says "$\gamma$ is a hyperparameter that controls the scale of the influence" and the expression is $-\text{sign}(\gamma) \sum_\alpha (d(\mu_k^{(\alpha)}, z))^7$. Wait - is 7 actually γ? Let me look more carefully... The equation shows exponent "7" not "γ". This seems like either a typo or an unusual design choice. But actually, γ is defined as a scale parameter with sign(γ), not as the exponent. So exponent being fixed at 7 IS a specific hyperparameter choice that is unexplained.

4. **Source data access**: TTVD requires pre-computed class means from training data. This is a meaningful design constraint that limits applicability compared to methods like Tent/SAR that only need the pre-trained model.

5. **Algorithm notation**: Algorithm 1 updates σ (feature extractor), but the paper says "only the channel-wise affine parameters in normalization layers are updated during TTA" while rest of the model remains unchanged. This IS an inconsistency.

Let me now write the consolidated review.</think>Now I have enough context to write the final review. Let me carefully synthesize everything.

## Summary

The paper proposes TTVD, a geometric framework for Test-Time Adaptation that reframes neighbor-based TTA methods through the lens of Voronoi Diagrams. It extends the basic VD in two directions: (1) Cluster-induced Voronoi Diagram (CIVD), which creates multi-site influences via rotation augmentations to unify self-supervision and entropy minimization; and (2) Power Diagram (CIPD), which introduces flexible weighted boundaries to identify and filter noisy samples near decision boundaries. Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R show improvements in classification error and ECE over state-of-the-art methods.

## Strengths

- **Novel geometric perspective on TTA**: The paper establishes a principled connection between neighbor-based TTA methods and Voronoi Diagrams, providing a unified geometric interpretation for prototype-based adaptation. This is a fresh conceptual lens that enables structural insights (e.g., identifying boundary samples for filtering) that heuristic methods lack.

- **Progressive and well-motivated framework**: The construction VD → CIVD → CIPD is logically motivated, with each extension addressing a specific limitation—CIVD addresses single-site insufficiency by incorporating multi-source influence through self-supervision, and CIPD addresses noisy sample filtering through flexible boundaries. The ablation study (Table 2) validates each component's contribution (5.7% from CIVD, 2.2% from CIPD over VD on CIFAR-10-C).

- **Strong empirical performance**: TTVD achieves consistent improvements across four benchmarks in both error rate and ECE, including under challenging settings (small batch sizes and label shift in the appendix). The improvements on ECE (3.4–4.3% reductions) suggest meaningful calibration gains, not just accuracy improvements.

- **Standardized evaluation**: The use of TTAB for fair comparison addresses a common concern in TTA literature about inconsistent experimental protocols.

## Weaknesses

### Major

- **The geometric framing, while novel, partially repackages established mechanisms**: At its core, the VD loss (Eq. 3) computes soft labels via a distance-based softmax to class prototypes and minimizes their entropy. This is conceptually close to prototype-based self-training with entropy minimization—the Voronoi framing provides an interpretation but the resulting adaptation objective is a modified distance-based entropy. The CIVD extension adds rotation augmentations to create multi-site prototypes, which is fundamentally an augmentation-based multi-view averaging scheme (not unlike prior multi-view TTA methods), re-described through the influence function lens. While the geometric perspective offers interpretability advantages, the paper's central claim of a "novel framework" based on computational geometry somewhat overstates the methodological departure from existing techniques. The actual TTA algorithm does not exploit geometric properties unique to Voronoi diagrams (e.g., efficient nearest-neighbor queries, boundary computation algorithms, or Delaunay triangulations) in its implementation.

- **Access to labeled source data for prototype computation limits applicability**: TTVD explicitly computes class means from training data ("We use the full training set of CIFAR-10, CIFAR-100 to compute the class means for Voronoi sites and 10% of ImageNet for similar calculation"), which requires access to the labeled source dataset. Many TTA methods (Tent, SAR, NOTE) only require the pre-trained model, making TTVD less broadly applicable. While the paper shows robustness to subsampling (Table 4: 59.8%, 59.8%, 59.9% at 10%, 5%, 1%), these results still require some labeled source data and do not include a 0% (no source data) baseline—this would be equivalent to random initialization of prototypes, which is an important reference point.

- **Noisy sample filtering via Power Diagram is under-specified**: Section 3.3 introduces PD-based filtering via "subtracting the PD from the VD" to identify boundary regions, but this operation is never formally defined for high-dimensional feature spaces. Figure 2(b) provides a 2D visualization on MNIST, but no algorithmic or mathematical specification is given for how this subtraction translates to an actual sample selection rule in $\mathbb{R}^\ell$ (where $\ell$ is 512–2048 in the actual experiments). The paper references "Algorithm 3 in Appendix H" for the complete CIPD procedure, but this critical operational detail is deferred without any explanation in the main text. There is no ablation isolating the PD filtering contribution from the weight-based re-partitioning, so it is unclear whether the 2.2% gain from CIPD over CIVD comes from filtering or from the weighted distance alone.

- **Arbitrary hyperparameter choices without justification or sensitivity analysis**: The influence function in CIVD (Eq. 4) uses an exponent of 7: $F(z,C_k) = -\text{sign}(\gamma)\sum_\alpha d(\mu_k^{(\alpha)}, z)^7$, but no justification is provided for why 7 rather than 2 or another value. Similarly, the temperature τ in Eq. 3 and the CIPD exponent γ in Eq. 6 are critical hyperparameters with no sensitivity analysis. The $\text{sign}(\gamma)$ term in Eq. 4 qualitatively changes the behavior of the influence function (switching between maximization and minimization), but its role is never discussed.

### Minor

- **No error bars or statistical significance**: Tables 1–3 report single numbers without standard deviations or confidence intervals. Given that some improvements are modest (e.g., 0.7% on CIFAR-100-C error), statistical significance is important to establish.

- **Ablation limited to a single dataset**: Table 2 provides the VD → CIVD → CIPD ablation only on CIFAR-10-C. Without corresponding ablations on CIFAR-100-C, ImageNet-C, or ImageNet-R, it is unclear whether the progressive improvements generalize.

- **Algorithm notation inconsistency**: Algorithm 1 updates the full feature extractor σ via gradient descent, while the paper text states that "only the channel-wise affine parameters in normalization layers are updated during TTA." This discrepancy could confuse readers about the adaptation scope.

- **Limited baseline comparison among neighbor-based methods**: Table 3 compares TTVD against AdaNPC on just four blur corruption types on ImageNet-C. A broader comparison with recent methods combining self-supervision and entropy minimization (e.g., methods that also unify these objectives) would strengthen the empirical case.

### Trivial

- The visualization in Figure 1 (on MNIST in 2D) is useful for intuition, but the geometric properties of Voronoi diagrams (boundary interpretability, cell structure) are known to degrade in high dimensions. No discussion addresses whether the geometric claims hold in the 512–2048 dimensional spaces where experiments actually run.

## Nice-to-Haves

- A simple baseline that applies Tent/SAR directly on prototype distances (without the Voronoi/CIVD/CIPD machinery) would help isolate the geometric contribution from the prototype-distance-based formulation.

- An ablation that compares PD-based filtering vs. entropy-based filtering (all else equal) to directly validate the noisy sample identification claim.

- Computational cost (wall-clock time, memory overhead) comparison against baselines like Tent and SAR, since TTA methods are often deployed in resource-constrained settings.

- Experiments on natural distribution shifts (beyond synthetic corruption benchmarks) and on architectures without BatchNorm (e.g., ViT).

## Removed Points

- **"Geometric framing is just re-interpretation" (Harsh Critic Issue #1, full version)**: While the geometric framing does re-describe some established concepts, stating it as "not a substantive new algorithm" is too reductive. The distance-to-prototype soft labels produce a different loss landscape than standard entropy minimization on model predictions, and CIPD introduces genuinely novel mechanisms (weighted partitioning, diagram-based filtering). The contribution is in the geometric lens enabling new designs, even if some components have simpler descriptions. **Kept a moderated version under Major Weakness #1.**

- **"Questionable exponent in Equation 4 is a typo" (Neutral Reviewer #2)**: The exponent "7" may or may not be a typo, but calling it suspicious without evidence goes beyond what the text supports. It IS unexplained, which is a valid criticism. **Kept as part of Major Weakness #4 (no justification/sensitivity analysis).**

- **"Missing baselines like DeYO, ViDA, CFA" (Spark)**: The paper does include a reasonable set of baselines across four categories (neighbor-based, repurposed DA, self-training, entropy-based). Requesting additional recent methods is reasonable but the paper's baseline set is adequate for its scope. **Moved to Nice-to-Haves.**

- **"No experiment measuring negative transfer avoidance" (Spark)**: The paper makes claims about CIVD unifying self-supervision and entropy to avoid negative transfer but does not directly measure this. This IS a gap in evidence but does not invalidate the method. **Kept as a minor implicit point under Major Weakness #1.**

- **"Reproducibility: Algorithm 3 deferred to appendix" (Spark)**: The CIPD algorithm being in the appendix is a reasonable editorial choice for space; the main text provides the key equations. This is a formatting concern. **Removed.**

- **"Only CNN-based experiments, missing ViT" (Human Finder)**: This is a valid but standard scope extension request; the paper evaluates on the standard TTA benchmarks with the standard architectures used in TTAB. **Moved to Nice-to-Haves.**

## Novel Insights

The geometric connection between prototype-based TTA classification and Voronoi diagrams is genuinely insightful—it reveals that the decision boundaries in TTA are Voronoi cell boundaries, and that misclassified test samples lying near these boundaries are precisely the ones with high loss and unstable gradients. This provides a structural explanation for why entropy-based filtering (which removes such samples) works, while also showing its limitations: only samples near boundaries can be geometrically identified, and the Power Diagram's weighted boundaries can capture a broader set of potentially noisy samples. The idea of diagram subtraction (even if under-specified) as a concrete geometric procedure for sample selection is a creative direction that could inspire more principled approaches to sample filtering in TTA.

## Suggestions

- Provide explicit mathematical specification of the PD-VD subtraction operation for sample filtering in high-dimensional spaces, and add an ablation comparing PD-based filtering alone vs. entropy-based filtering to validate this key claim.
- Add sensitivity analysis for the key hyperparameters (exponent in Eq. 4, γ in Eq. 6, temperature τ), and consider whether the exponent of 7 has any principled justification or if a standard choice (2) would suffice.
- Include ablations on at least one additional dataset beyond CIFAR-10-C to validate the VD → CIVD → CIPD progression.
- Clarify the ambiguity in Algorithm 1 regarding which parameters are updated (full feature extractor vs. BN parameters only).

## Score and Decision

**Calibration**: I compared against several TTA papers in the human review corpus:

- **COME** (Accept Poster, scores 6/6/5/8): Conceptual contribution (conservative entropy minimization) with solid but not breakthrough empirical results. TTVD has comparable empirical gains but a less rigorously validated conceptual contribution.
- **PROGRAM** (Accept Poster, scores 6/5/6/6/8): Prototype-graph-based pseudo-labeling for TTA with gains over baselines. Similar novelty level (prototype-based approach) and similar empirical scope. TTVD's geometric lens is more novel than PROGRAM's graph-based pseudo-labeling.
- **DeYO** (Accept Spotlight, scores 8/6/6/8): Strong conceptual contribution (entropy is not enough), empirical improvement on multiple settings, and clear evidence for each claim. TTVD is weaker on this axis.
- **CTA by Source Prototypes** (Reject, scores 3/5/5/6/5): Very similar approach (using source prototypes for TTA alignment), but with limited novelty, weak empirical gains, and similar concerns about source data access. TTVD is stronger than this but shares the source-data-access limitation.
- **NGTTA** (Reject, scores 6/5/6/5): Geometry-driven TTA for point clouds. Similar concept of using geometric priors, but with writing/clarity issues and limited validation. TTVD is somewhat stronger empirically but has similar concerns about conceptual overselling.

TTVD sits between these anchors: stronger than the rejected prototype CTA method and NGTTA, with a genuinely novel geometric perspective, but with notable gaps in specification (PD filtering), evidence (ablations, statistical significance), and scope (limited baselines, synthetic benchmarks only). The conceptual novelty is real but partially oversold, and the empirical contribution is solid but not overwhelming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>