Now I have sufficient context. Let me synthesize the final review.

## Summary

The paper proposes TTVD, a framework for test-time adaptation (TTA) that reformulates neighbor-based TTA through the lens of Voronoi Diagrams. The basic TTA mechanism uses pre-computed class means as Voronoi sites and minimizes an entropy-like loss over softmax of negative distances. The framework is then extended in two directions: (1) Cluster-induced Voronoi Diagram (CIVD), which expands each site to a cluster of augmented sites (via rotation self-supervision) aggregated through a multi-source influence function, and (2) Power Diagram (PD), which introduces weighted cells to identify noisy samples near decision boundaries via "diagram subtraction" from VD cells. Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R under the TTAB benchmark show consistent improvements over baselines.

## Strengths

- **Systematic empirical evaluation under standardized settings:** The use of TTAB (a peer-reviewed, open-source TTA benchmark) with grid-searched hyperparameters for all methods lends credibility to the comparisons. Results are reported on four datasets with both error and ECE metrics, and TTVD achieves the best results across all settings with notable ECE reductions (3.4%, 1.8%, 4.1%, 4.3%).
- **Clear ablation within the method family:** The progressive VD → CIVD → CIPD ablation on CIFAR-10-C (Table 2) shows meaningful contributions from each geometric component: +5.7% from CIVD and +2.2% from CIPD, indicating that the added structures do provide quantifiable benefits within the proposed framework.
- **Helpful interpretability via geometric visualization:** Figure 1 and Figure 2 provide intuitive visualizations of how different diagram variants partition feature space and how they relate to adaptation performance, making the geometric perspective accessible even if its novelty is debatable.
- **Practical robustness analyses:** The paper tests robustness to class mean precision (Table 4: 1%–10% data gives near-identical results), batch size effects, and label shift, which are important for practical TTA deployment.

## Weaknesses

### Major:

- **Core conceptual novelty is overstated — the geometric framing may be largely descriptive rather than operationally distinct.** The key observation that neighbor-based methods correspond to Voronoi Diagrams is a well-known textbook result in computational geometry and pattern recognition. The "VD loss" (Eq. 3) — softmax over negative distances plus entropy — is essentially equivalent to prototype-based pseudo-label entropy minimization, which is already used by methods like SHOT. CIVD with rotation-augmented sites aggregated via the influence function F(z, C_k) structurally resembles combining self-supervision with prototypes (as in TTT + prototype alignment). The PD-based filtering resembles margin-based or confidence-based sample selection. The paper does not provide any ablation or comparison against non-geometric proxies that implement the same functional ingredients (prototype distances + augmentation ensembling + boundary-region filtering) without the CIVD/PD machinery. Without this, the claim that the geometric structures themselves (rather than the engineering recipe they encode) drive the improvements is unsubstantiated. The paper repeatedly frames this as a "breakthrough" and "novel framework," but the evidence supports TTVD as a competently engineered combination rather than a conceptual advance.

- **The "diagram subtraction" mechanism for noisy sample filtering is under-specified and unvalidated.** Section 3.3 states "By subtracting the PD from the VD, we can extract a larger region from the resulting differences," but no precise algorithmic rule is given for how this subtraction operates in high-dimensional feature space. Does the method exclude samples whose PD and VD class assignments differ? Does it threshold boundary distances? The visual in Figure 2b is only in 2D on MNIST, yet all real experiments operate in high-dimensional embeddings. Furthermore, the claimed superiority of PD-based filtering over standard entropy-based or confidence-based filtering is never empirically tested — there is no ablation comparing CIVD+entropy-filter vs. CIPD, nor any quantitative analysis of how many truly noisy samples are removed by the PD-VD subtraction. The +2.2% gain from CIVD to CIPD could come from any number of implementation details rather than the geometrically motivated filtering.

- **TTVD requires source data (class means), blurring the boundary with SFDA and creating an uneven comparison with some TTA baselines.** The Voronoi sites μ_k are computed from the training set (requiring access to source data). While the paper shows robustness to site precision (Table 4), this fundamentally makes TTVD a source-dependent method, unlike methods such as T3A, Tent, or SAR which operate in a strictly source-free setting. The paper positions itself within TTA but does not explicitly acknowledge or discuss this asymmetry — some baselines (e.g., T3A) do not use source data at all, while TTVD does. This affects the fairness of comparisons.

### Minor:

- **Limited ablation scope:** The VD → CIVD → CIPD ablation is only on CIFAR-10-C (Table 2). There is no corresponding ablation on CIFAR-100-C or ImageNet-C/R, where the absolute improvements over baselines are comparable or larger. Additionally, there is no ablation testing whether CIVD's specific influence function (with exponent γ) provides gains over simply averaging features from augmented views, nor whether the temperature τ in Eq. 3 plays a critical role vs. standard softmax temperature.

- **Ambiguity between the classifier's PD geometry and the method's PD weights:** Lemma 3.1 connects a logistic regression classifier to a Power Diagram, and the paper uses PD for sample filtering. However, it is unclear whether the PD weights v_k used for filtering are derived from the frozen classifier parameters (via Lemma 3.1) or are separately tuned. If from the frozen classifier, the PD boundaries are static during adaptation; if separately set, this adds tunable parameters. The paper does not clearly specify this, making it difficult to map the formal definitions to the actual implementation.

- **Experiments limited to ResNet architectures:** All experiments use ResNet-26 or ResNet-50. Whether the geometric assumptions hold for architectures without batch normalization (e.g., ViTs) is not tested, and the method's reliance on BN updates for adaptation may limit applicability.

### Trivial:

- The influence function exponent γ (appearing as 2 in Eq. 6 and 7 in Eq. 4) is never justified or analyzed for sensitivity, though this is a relatively standard hyperparameter concern.

## Nice-to-Haves

- Ablation on all four datasets (not just CIFAR-10-C) and comparison with non-geometric proxies that implement the same functional components without CIVD/PD machinery — e.g., "prototype distances + rotation-augmented feature averaging + entropy-based filtering." This would most directly test whether the geometric perspective adds value beyond a competent engineering combination.
- Comparison with more recent TTA methods (e.g., MEMO for augmentation-based consistency, EATA, CoTTA), particularly those that also use augmentation or source prototypes.
- Computational cost analysis: per-batch wall-clock time for TTVD vs. baselines, especially given that CIVD requires computing influences from K×4 sites per sample.
- Analysis of PD-VD subtraction in high-dimensional space: a quantitative evaluation of what fraction of mislabeled or high-gradient samples are captured in the PD-VD difference region vs. standard entropy thresholding.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The VD-nearest-neighbor connection is not formally proven"** (from neutral reviewer): The correspondence between nearest-neighbor classification and Voronoi Diagrams IS a well-established textbook result. Requesting a formal proof of this is unnecessary; the real issue is whether the geometric framing provides new operational insights, not whether the correspondence holds.

- **"Unclear generalization beyond rotation self-supervision"** (from neutral reviewer): This is speculative — the paper demonstrates one choice (rotation) and testing other augmentations is a reasonable extension, not a requirement for acceptance.

- **"Insufficient theoretical justification for CIVD's influence function"** (partially from neutral reviewer): The influence function is a well-defined construct from prior computational geometry work (Chen et al., 2013; 2017; Huang et al., 2021). While sensitivity to γ is fair game, demanding a theoretical justification for why this specific form works in deep feature space is an unreasonable standard for an empirical TTA paper.

- **"Missing domain generalization benchmarks like DomainNet, Office-Home, WILDS"** (from spark reviewer): The paper uses four standard corruption/rendition benchmarks that are the normative evaluation for TTA methods. Demanding additional benchmarks with different characteristics is scope creep.

- **Formatting/style nitpicks from various reviewers** (not included above but noted for removal).

## Novel Insights

The paper's most genuinely novel contribution is the identification that the Power Diagram (induced by the pretrained classifier's weight matrix and bias, per Lemma 3.1) and the Voronoi Diagram (induced by class means in feature space) produce *different* space partitions, and that the disagreement region between them corresponds to samples near decision boundaries that are likely to generate unstable gradients. Even though the operational details of exploiting this disagreement are under-specified, the conceptual observation that a neural network's classifier and its feature-space prototypes define two distinct partitionings, and that their divergence is informative, is a genuinely interesting geometric insight that could motivate future work on sample selection for TTA.

## Suggestions

- **Add a critical ablation:** Replace CIVD's influence function with simple averaging of rotated features (same augmentation, no γ exponent), and replace PD-VD subtraction with standard entropy-threshold filtering. If the gap between these and TTVD is small, the geometric framing is mainly descriptive; if it is large, the geometric structures genuinely matter. This single experiment would resolve the most significant criticism.
- **Explicitly acknowledge source-data dependence:** Discuss that TTVD operates in a setting closer to SFDA than strictly source-free TTA, and clarify which baselines share this characteristic.
- **Provide Algorithm 3 (referenced in main text as being in Appendix H) in the main paper** if space permits, or at minimum specify precisely how PD-VD subtraction is implemented for the CIPD noisy sample filtering step.

## Score and Decision

**Calibration anchors:**

- **PIF (Prototypical Influence Function for TTA)** — Reject, scores 5,5,3,5,5 (avg ~4.6). Similar to TTVD: requires source prototypes, moderate novelty over existing methods, concerns about hyperparameters and computational cost, but solid empirical results. TTVD is somewhat stronger than PIF due to more benchmarks, ECE improvements, and TTAB standardization.

- **Continual TTA with Source Prototypes** — Reject, scores 3,5,5,6,5 (avg ~4.8). Very similar to TTVD in using source prototypes for TTA. Rejected for lack of technical innovation, marginal gains, and source-data reliance. TTVD has more substantial gains and a clearer framework, placing it above this paper.

- **PROGRAM (Prototype Graph for Pseudo-Label TTA)** — Accept poster, scores 6,5,6,6,8 (avg ~6.2). Stronger novelty with graph-based label propagation, thorough ablations, and clear algorithmic contribution. TTVD is weaker due to less clear algorithmic novelty and missing discriminative ablations.

- **Adaptive Energy Alignment** — Accept poster, scores 8,6,8,3 (avg ~6.25). Genuine theoretical insight (entropy decomposition into conflicting energy terms) with clear experiments. TTVD's geometric framing provides less operational novelty.

TTVD sits between the rejected prototype-based TTA papers (~4.8) and the accepted TTA papers (~6.2). Its strengths over rejected papers include: (1) more comprehensive evaluation across four benchmarks with both error and ECE, (2) use of TTAB for fair comparison, and (3) the PD-VD disagreement observation. Its weaknesses preventing it from reaching accepted-paper territory include: (1) the core novelty is largely a geometric re-description of existing mechanisms, (2) the PD-based filtering is under-specified, and (3) missing ablations against simpler non-geometric proxies.

**Score: 5.0** — marginally below acceptance. The paper presents a competent, well-evaluated combination of existing ideas with geometric terminology, but the conceptual contribution is primarily descriptive rather than operationally novel, and the key mechanism (PD-VD diagram subtraction) lacks sufficient specification or empirical validation against simpler alternatives.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>