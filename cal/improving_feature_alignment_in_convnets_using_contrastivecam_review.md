=== CALIBRATION EXAMPLE 23 ===

# Final Consolidated Review
## Summary

The paper identifies a theoretical limitation of HiResCAM—its explanations are not uniquely determined from probability predictions due to softmax shift invariance, admitting an arbitrary spurious matrix $M$—and proposes ContrastiveCAMs, which are invariant to this shift while providing pairwise class-versus-class explanations. Leveraging the observation that networks often rely on non-core regions, the paper further introduces Core-Focused Cross-Entropy (CFCE), a training loss that suppresses non-core region contributions in ContrastiveCAMs to improve feature alignment. Experiments on Hard-ImageNet, Oxford-IIIT Pets, and PASCAL VOC demonstrate improved alignment (IoU) and reduced reliance on spurious features.

## Strengths

- **Theoretical identification of HiResCAM non-uniqueness (Theorem 3.2) and elegant resolution via ContrastiveCAMs (Theorem 3.5):** The proof that all HiResCAMs sharing a common additive spatial matrix $M$ yield identical probability predictions, and that subtracting class-wise maps eliminates $M$, is clean and provides a principled foundation for more faithful explanations. This is a specific, concrete theoretical contribution that most CAM-family papers lack.
- **Well-controlled experimental isolation of loss vs. architecture effects:** The "CE w/ Arch" baseline in Table 2, which applies the same architectural modifications (removed bias, BN, ReLU, downsampling) but uses standard cross-entropy, cleanly isolates the contribution of the CFCE loss. The IoU gap (16.25% → 51.52%) cannot be attributed to architecture changes alone.
- **Practical adaptability with approximate masks:** Section 5.2 demonstrates that SAM-generated and bounding-box masks yield competitive alignment to ground-truth masks on Oxford-IIIT Pets (e.g., CFCE+KL with SAM: 83.54% valid IoU vs. GT: 92.72%), suggesting the method is not limited to datasets with pixel-perfect segmentation.

## Weaknesses

### Major:

- **Unreported computational overhead of training with spatial CAM maps in the loss:** CFCE (Eq. 15) requires materializing ContrastiveCAM spatial maps for all $C{-}1$ class pairs per sample during each training step. While for single-layer classifiers this avoids second-order gradients (since HiResCAM = $W_c \odot A$), the overhead of computing, storing, and backpropagating through $C$ spatial maps of size $d_1 \times d_2$ per sample is non-trivial. The paper reports no training time, memory footprint, or FLOPs comparison against standard CE. For a proposed training method, this omission undermines practical viability assessment. The hyperparameter section (Appendix C) states a batch size of 768, which is unusually large and may itself be an artifact of the overhead—this should be clarified.

- **No empirical validation that the spurious shift $M$ is practically significant in trained models:** Theorem 3.2 proves $M$ *can* exist, but the paper does not quantify its magnitude in actual trained networks. Table 1 reports a "redundancy ratio" $\gamma$ (e.g., 0.201 for Hard-ImageNet, 0.367 for Pets), suggesting the removed component is 20–37% of the Frobenius norm—but this is presented without discussion of whether $M$ actually produces misleading explanations in practice. A direct comparison of HiResCAM vs. ContrastiveCAM visualizations on the same model, with quantitative evaluation of explanation faithfulness (e.g., insertion/deletion metrics), would establish whether this theoretical contribution addresses a real practical problem. The paper's Figure 1 is a synthetic example, not an empirical demonstration.

- **Incomplete ablation of loss components and hyperparameters:** The paper introduces three distinct innovations: (1) ContrastiveCAMs, (2) CFCE loss (non-core suppression via absolute value), and (3) KL divergence regularization. Tables 2–4 report "CFCE" and "CFCE+KL" but do not isolate: (a) whether using standard HiResCAM instead of ContrastiveCAM in the CFCE loss degrades alignment, (b) the individual effect of the absolute-value suppression term vs. the core-region retention term, and (c) sensitivity to $\lambda = \{50, 10^3, 10\}$—the three distinct regularization coefficients are never independently varied or analyzed.

### Minor:

- **Architectural constraints buried in appendix:** The method requires removing the final bias, BatchNorm, and ReLU from the last convolutional block (Appendix C.1) to maintain theoretical faithfulness. This limits applicability to off-the-shelf pre-trained models. While "CE w/ Arch" partially controls for this, the constraint should be discussed in the main text as a scope limitation, not deferred to an appendix.
- **Accuracy–alignment tradeoff under-quantified:** On Hard-ImageNet (Table 2), CFCE drops clean accuracy from 93.69% to 90.53% while dramatically improving IoU. The paper states this is "at the cost of some un-ablated performance" but does not analyze whether this tradeoff is Pareto-optimal or whether intermediate $\lambda$ values offer better tradeoffs. A Pareto frontier plot would clarify practical utility.
- **Theorem 4.6 relies on a strong realizability assumption:** The consistency proof for CFCE as a surrogate for Core-Constrained Risk Minimization assumes the function class $\mathcal{F}$ is sufficiently expressive to realize the optimal $R^*_{CCRM}$. In over-parameterized deep networks, this may not hold in the constrained sense (the model may be expressive enough for CE-optimal but not CCRM-optimal solutions). The practical implications of this gap are unexplored.

### Trivial:

- The class-versus-class explanation granularity (Definition 3.3) produces $C{-}1$ maps per prediction, which scales poorly for large $C$. This is a known limitation of pairwise approaches and not a deep flaw, but deserves a brief note.

## Nice-to-Haves

- Comparison with mainstream robustness methods (e.g., IRM, Group DRO) on Hard-ImageNet to contextualize CFCE within the broader shortcut-mitigation landscape.
- Pareto frontier plots (Accuracy vs. Core-IoU) across methods and $\lambda$ values.
- Analysis of whether suppressing non-core regions causes the model to discover *new* spurious features within core regions (a common failure mode of mask-guided training).
- Extension to architectures beyond ResNet-50 (e.g., ConvNeXt, which the paper cites in Section 2 as having a single-layer classifier).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Second-order gradient overhead"** (from harsh critic): The critic claims computing CAM-in-loss requires second-order gradients (gradients of gradients). This is factually incorrect for the paper's setting. For a single-layer classifier $h(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$, HiResCAM simplifies to $W_c \odot \mathbf{A}$ (as shown in Draelos & Carin 2020 and used throughout the paper). Backpropagating through this requires only standard first-order gradients. The computational overhead concern is valid but the claimed mechanism is wrong.
- **Weakness: "Missing state-of-the-art saliency-guided training baselines like Ismail et al. (2021) and Gao et al. (2024)"** (from spark finder): Per review rules, I cannot confirm these works exist or are appropriate baselines, so this is removed. The more general point about limited baseline breadth is retained in Minor weaknesses.
- **Weakness: "Standard deviation vs. standard error" and "number of runs not stated"** (from harsh critic): The paper reports ± values consistently (e.g., $90.53 \pm 0.69$). This is a reproducibility nitpick. The paper also states code and weights will be released.
- **Weakness: "PASCAL VOC 2011 is outdated"** (from spark finder): The dataset is used specifically because it provides segmentation annotations paired with classification labels, which is the core requirement of the method. The dataset choice is fit for purpose.
- **Weakness: Demanding confidence intervals for benchmarks**: Not standard in this field for the reported metrics; removed as a nice-to-have at best.
- **Formatting complaints about parser artifacts**: Removed per instructions.

## Novel Insights

The redundancy ratio $\gamma$ reported in Table 1 (20–37% of HiResCAM Frobenius norm is the spurious component $M$) is a surprisingly concrete quantification of the gap between logit-level and probability-level explanations. This suggests that in practice, roughly a quarter to a third of what HiResCAM attributes to each class is information invisible to the model's probability output—a finding that, if validated with faithfulness metrics beyond norm ratios, would significantly strengthen the paper's practical motivation. Additionally, the paper's theoretical decomposition of cross-entropy into core and non-core ContrastiveCAM contributions (Proposition 4.2 / Eq. 12) provides a *causal* lens on why CE permits shortcut learning: it is not merely an absence of regularization, but a structural property that CE is indifferent to which regions drive the logit differences. This is sharper than the typical "CE doesn't penalize spurious features" argument.

## Suggestions

1. **Report training time and memory overhead** for CFCE vs. CE on each dataset, including the spatial map computation cost. This is essential for a training-method paper.
2. **Directly compare HiResCAM vs. ContrastiveCAM faithfulness** on the same trained model using insertion/deletion or pointing game metrics, to empirically validate that the removed component $M$ is not just norm but actually misleading.
3. **Add ablation rows** for CFCE with standard HiResCAM (not ContrastiveCAM) in the loss, and for individual $\lambda$ sensitivity, to isolate which components drive which improvements.
4. **Move architectural constraints discussion** from Appendix C.1 to the main text (Section 4 or 5), clearly stating that the faithfulness guarantees require these modifications and that applicability to unmodified off-the-shelf architectures is out of scope.
5. **Plot Accuracy vs. Core-IoU** for varying $\lambda$ values to show whether CFCE offers a controllable and Pareto-dominant tradeoff.

---

**Evaluation by axis:**

- **Novelty:** Moderate-to-high. The $M$-shift identification and ContrastiveCAM resolution are genuine theoretical contributions. CFCE is a natural but well-motivated extension.
- **Technical soundness:** Generally sound theoretical proofs, but the gap between Theorem 3.2's existence proof and empirical significance, plus the unverified realizability assumption in Theorem 4.6, leave open questions.
- **Empirical support:** Adequate for the alignment claim, but the absence of computational overhead reporting, incomplete ablations, and no direct faithfulness comparison between HiResCAM and ContrastiveCAM limit the strength of the empirical case.
- **Significance:** Moderate. Bridging interpretability guarantees and feature alignment training is valuable, but mask dependency and architectural constraints limit immediate practical scope.
- **Clarity:** Good overall; notation is consistent, though dense. The burying of architectural constraints in the appendix is the main clarity issue.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
