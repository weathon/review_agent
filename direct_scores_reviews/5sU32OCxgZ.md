## Summary
TTVD proposes a geometric framework for Test-Time Adaptation (TTA) by formalizing neighbor-based methods as Voronoi Diagrams (VD) and then extending this to two more expressive structures: Cluster-induced Voronoi Diagrams (CIVD), which incorporates rotation-augmented multi-site influence to unify self-supervision and entropy minimization, and Power Diagrams (PD), which introduces site weights to shift decision boundaries for improved noisy-sample filtering. The complete method (CIPD = CIVD + PD) is benchmarked against a wide array of TTA baselines on CIFAR-10/100-C, ImageNet-C, and ImageNet-R using the standardized TTAB toolkit.

---

## Strengths

- **Geometric reframing enables principled extensions.** While the connection between nearest-prototype methods and VD is definitionally straightforward, articulating it explicitly unlocks access to a well-developed literature on CIVD and Power Diagrams. The progression VD → CIVD → CIPD is coherent, and the multi-site influence mechanism of CIVD (Eq. 4, 6) is a non-trivial extension that subsumes both rotation self-supervision and entropy minimization into a single geometric objective rather than treating them as competing losses.

- **Substantial ECE improvements alongside classification error.** TTVD achieves ECE reductions of 3.4%, 1.8%, 4.1%, and 4.3% across the four benchmarks—materially larger than the error improvements and practically important for deployment trustworthiness. Most competing TTA papers focus only on error; this dual-metric reporting is a genuine added value.

- **Rigorous evaluation via TTAB.** Using a peer-reviewed, open-source standardized toolkit for all comparisons, with full grid search and consistent hyperparameter selection across methods, is a high standard that many TTA papers do not meet. This substantially increases confidence in the reported numbers.

- **Favorable adaptation dynamics.** Figure 4 shows TTVD continuously improving over ~750 online batches without plateauing, contrasting with Tent and SAR which stagnate or converge early. This empirically supports the claim that the geometric structure provides longer-lasting guidance than pure entropy minimization.

- **Robustness under class imbalance and small batch size.** The paper goes beyond standard benchmark evaluation to test these practically relevant scenarios (Appendix B), which is a meaningful additional contribution for real-world deployment analysis.

---

## Weaknesses

- **Core claim of negative-transfer avoidance is unverified.** Section 3.2 asserts that CIVD "avoids negative transfer since the objective is now unified," but this claim—which is central to the motivation for CIVD—is never empirically quantified. There is no measurement of gradient cosine similarity between the VD loss and entropy minimization loss, no ablation that directly contrasts CIVD against a naïve joint loss combining VD + rotation-self-supervision, and no experiment that reproduces the conflicting-gradient scenario from Gandelsman et al. (2022) with and without CIVD. Without this evidence, the "unified objective" narrative is asserted rather than demonstrated.

- **CIVD benefit is not isolated from multi-augmentation.** The ablation in Table 2 compares VD vs. CIVD vs. CIPD, showing a 5.7% jump from VD to CIVD. However, CIVD simultaneously adds (a) the geometric multi-site influence structure and (b) rotation augmentation (four new sites from Rot ∈ {0°, 90°, 180°, 270°}). There is no baseline that adds rotation augmentation *without* the CIVD influence function (i.e., simply averaging predictions from 4-rotation TTT). Without this control, it is impossible to attribute the gain to the geometric structure rather than to the additional augmentation signal. This undermines the paper's geometric narrative.

- **Suspicious insensitivity in Table 4 puts the geometric mechanism in question.** Using 10%, 5%, or 1% of ImageNet training data to compute Voronoi sites yields identical classification error (59.8, 59.8, 59.9). If the Voronoi sites—the central structural ingredient—barely matter for performance, this raises the question of whether the sites are actually guiding adaptation or whether the performance comes from the entropy-minimization component of the loss operating largely independently of site quality. The paper presents this as a robustness positive but does not address the mechanistic implication.

- **Missing important contemporaneous baselines.** Methods with strong reported results on the exact same benchmarks—CoTTA (Wang et al., 2022), EATA (Niu et al., 2022), RoTTA (Yuan et al., 2023), and ROID (Döbler et al., 2023)—are absent from Table 1. Given that the paper's headline claim is "state-of-the-art," these omissions weaken that claim.

- **Online computational overhead is unreported.** CIVD requires 4 forward passes per test image (for the four rotation augmentations), yielding ~4× inference overhead relative to Tent or SAR. This cost is never reported or discussed. For a method targeting real-time deployment, inference latency and FLOPs per batch are essential and missing.

- **Full adaptation algorithm deferred to an unavailable appendix.** Algorithm 3 in Appendix H contains the complete CIPD adaptation loop, which is one of the two core contributions. The main text only verbally describes the PD-VD subtraction filtering and does not provide a self-contained pseudocode. This materially hinders reproducibility assessment from the main paper.

- **Ablation study limited to a single dataset.** Table 2 ablates VD → CIVD → CIPD only on CIFAR-10-C. It is unclear whether the 5.7% gain from CIVD and additional 2.2% from CIPD hold on ImageNet-C (K=1000, higher-dimensional features, larger distribution shift). Given that the paper claims four-dataset state-of-the-art, the ablation should cover at least one large-scale benchmark.

- **Rotation augmentation limitation not acknowledged.** The CIVD construction relies on 90°-rotation invariance, which is well-suited for natural image classification but inappropriate for medical imaging, satellite imagery, or text recognition, where rotation carries semantic content. The paper does not acknowledge this scope limitation anywhere, despite it being a meaningful constraint on generalizability.

---

## Nice-to-Haves

- **Online prototype update mechanism.** Adding a moving average update of Voronoi sites μ_k during test time would address potential feature drift over long test streams and strengthen the theoretical coherence of the method. Currently, sites computed from training data are fixed while the feature extractor σ is updated.

- **Gradient conflict visualization.** Plotting gradient cosine similarity between the VD objective and the rotation self-supervision objective before and after CIVD unification (e.g., using the technique from Gandelsman et al.) would directly validate the "negative transfer avoidance" claim.

- **Adaptation curve comparison including T3A and TAST.** Figure 4 shows adaptation curves only for Tent and SAR. Including the neighbor-based baselines (T3A, TAST) that are most architecturally related to TTVD would provide a more informative comparison.

- **Feature-space visualization of site alignment.** A t-SNE plot comparing training-set Voronoi sites against test features before and after adaptation would visually validate the mechanism and substantiate the geometric narrative.

- **Ablation of γ sensitivity.** The influence function exponent γ=7 (Eq. 4) is taken from the CIVD literature but is not validated for the TTA setting. A brief sensitivity table would support its robustness.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"VD revelation is near-trivial" (Harsh Critic).** While the equivalence of nearest-prototype classification and VD is textbook knowledge, the paper's contribution is not the equivalence itself but the use of this framing to motivate CIVD and PD extensions. The paper explicitly calls it "building on this observation" rather than presenting the equivalence as a discovery in isolation. The framing is pedagogically motivated and does not constitute a false novelty claim. **Removed.**

- **VD loss is "indistinguishable" from Tent on prototype distances (Harsh Critic).** Equation 3 is entropy minimization applied to distance-to-prototype softmax logits, which is structurally similar to T3A, but the paper presents this as the *baseline* VD component, not the contribution. The contributions are CIVD and CIPD, built on top. Characterizing the paper's contribution as "re-expression of existing methods" ignores that VD is the foundation, not the claim. **Removed.**

- **Fairness of parameter updates (Spark Finder).** The paper explicitly states "commonly, only the channel-wise affine parameters in normalization layers are updated during TTA, while the rest of the model remains unchanged" and Algorithm 1 uses σ to denote the feature extractor (which includes BN layers under standard TTA). There is no evidence that TTVD updates more parameters than baselines. **Removed (misread).**

- **High-dimensional VD approximation error (Spark Finder).** The method does not actually construct a Voronoi diagram in high-dimensional space via a geometric algorithm. Inference is performed by computing K nearest-prototype distances via Eq. 3/6, which is standard ML with no approximation error beyond floating-point precision. The "VD" is a conceptual framing. **Removed (based on misunderstanding of implementation).**

- **Memory overhead of storing cluster sites (Spark Finder).** Storing 4K class mean vectors at 2048 dimensions for ImageNet-K=1000 amounts to ~32MB, which is negligible. **Removed.**

- **Statistical significance / confidence intervals (Harsh Critic).** Single-run evaluation without confidence intervals is the universal standard for large-scale TTA benchmarks (TTAB itself reports single runs). Demanding multiple seeds for ImageNet-scale TTA is not standard practice in this community. **Removed (non-standard requirement for this field).**

- **Oracle asymmetry analysis (Harsh Critic).** TTVD showing minimal gap between oracle and non-oracle results actually indicates consistent performance and robustness rather than a flaw. The asymmetry with other methods (e.g., SHOT gaining from oracle selection) reflects their sensitivity, not TTVD's weakness. **Removed.**

---

## Novel Insights

The most genuinely novel observation—which none of the reviews fully develop—is the "diagram subtraction" mechanism for noisy sample filtering (Section 3.3, Figure 2). The paper shows that entropy-based filtering (e.g., SAR) only captures samples near high-entropy boundaries, while PD boundary-shifting can recover a geometrically larger "noisy region" by taking the set difference of PD and VD cells. This is a qualitatively different characterization of test-time noise from prior work: rather than asking "is this sample's entropy above a threshold?", the method asks "does the PD and VD disagree on which region this sample belongs to?" If this mechanism is empirically validated (the current evidence is limited to 2D MNIST visualizations), it would represent a conceptually distinct approach to robust TTA with potential applicability beyond this specific method.

---

## Suggestions

1. **Add a CIVD-without-geometry ablation baseline**: Train a model with standard VD loss + rotation self-supervision as a separate auxiliary loss (equivalent to VD + TTT) and compare against CIVD. This is the critical control for isolating the geometric contribution of CIVD from the augmentation contribution.

2. **Report per-batch inference latency for CIFAR-C and ImageNet-C**, broken down by model component (4 forward passes for CIVD, PD computation, etc.), alongside Tent and SAR.

3. **Move Algorithm 3 (CIPD) or its key pseudocode into the main text**; the full CIPD loop is a core contribution and readers should not need an appendix to understand the method.

4. **Extend ablation (Table 2) to ImageNet-C** to confirm that the VD→CIVD→CIPD gains generalize to the large-scale setting.

5. **Add a measurement of gradient conflict**: before and after applying CIVD, measure cosine similarity between the entropy minimization gradient and the rotation self-supervision gradient to substantiate the "unified objective avoids negative transfer" claim.

6. **Acknowledge scope limitations** (rotation augmentation assumption, source data needed for class means) in a limitations section; the conclusion currently has none.

---

**Novelty:** Moderate-to-high. The CIVD and PD application to TTA is a meaningful departure from prior art, though the building blocks are imported from the computational geometry literature.

**Technical soundness:** Moderate. The geometric framework is mathematically coherent but the key empirical claims (negative transfer avoidance, Table 4 insensitivity implications) are inadequately supported.

**Empirical support:** Moderate. The TTAB evaluation is rigorous and ECE results are strong, but the missing baselines, single-dataset ablation, and lack of runtime analysis weaken the empirical case.

**Significance:** Moderate. The geometric lens on TTA is interpretable and extensible, but the practical gains in error rate are small (0.7–1.6%); the ECE gains are more impactful.

**Clarity:** Moderate. The VD→CIVD→CIPD progression is logically structured, but the core filtering algorithm is deferred to an appendix and the high-dimensional vs. 2D gap is not addressed.

MY FINAL SCORE: <pineapple>5.5</pineapple>