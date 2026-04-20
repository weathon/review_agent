Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

This paper introduces "principal distortions," a framework for comparing the local geometry of N > 2 image representations simultaneously. The authors define a pseudo-metric over FIM-induced sensitivities and find a pair of distortions that maximize variance of log sensitivity ratios across all models — a natural generalization of pairwise generalized eigen-distortions (Zhou et al., 2023) to the multi-model case. The method is demonstrated on a nested family of early visual system models and on deep neural networks (AlexNet, ResNet50) under standard, stylized, and adversarial training. The key empirical finding is that adversarial training, unlike stylized-ImageNet training, fundamentally reorients the principal axis of inter-model local geometry variation.

---

## Strengths

- **Clean, principled mathematical framework** (Section 3.1–3.2): The pseudo-metric m_{u,v} is well-defined, with desirable properties established formally — scale invariance (Eq. 3 is invariant to positive FIM scaling), permutation invariance, and connection to the Fisher-Rao geometry. The extension from pairwise to N-model via variance maximization is natural and analogous to PCA in a principled way.

- **Backward compatibility with prior methods**: The paper shows that eigen-distortions (N=1) and generalized eigen-distortions (N=2) are recovered as special cases of the framework (Section 3.1), ensuring mathematical consistency with the literature.

- **Novel empirical dissociation between adversarial and stylized training** (Fig. 4–5): The finding that SIN training does not alter the principal axis of variation (distortions still separate by architecture), while adversarial training shifts the axis so models separate by training type rather than architecture, is non-obvious, well-controlled across 100 base images, and a genuinely new observation in representational similarity analysis.

- **Robustness of results** (Figs. 3E, 4A, 5A): Results are reported with error bars across 100 ImageNet base images and replicated across 6 randomly initialized networks (3 AlexNet, 3 ResNet50), giving confidence that findings are not image- or seed-specific.

- **Interpretability of distortions** (Fig. 3B, 5B): The principal distortions are visually meaningful — ε₁ concentrates on high-contrast, textured regions while ε₂ concentrates on smooth regions — enabling hypothesis generation about mechanistic differences between models.

- **Effective pedagogical presentation**: Figure 1B, contrasting the three methods (eigen-distortions, generalized eigen-distortions, principal distortions) using FIM level sets and log-ratio visualizations, makes the conceptual advance immediately accessible.

---

## Weaknesses

### Fatal
None.

### Major

- **No quantitative evaluation of the method's discriminability or advantage over the direct baseline**: All main results are qualitative (visual inspection of distortion patterns, scatter plots of log sensitivity ratios). Critically, there is no quantitative comparison to applying Zhou et al. (2023) pairwise. For N=4 models (the early visual system case), Zhou et al. would produce up to 12 generalized eigen-distortions; the proposed method yields 2. Whether those 2 capture as much or more discriminating information — e.g., measured by silhouette coefficient, downstream classification accuracy from log sensitivity ratios, or variance explained — is never shown. A methods paper proposing an advantage in efficiency should demonstrate that the efficient solution does not sacrifice meaningful information relative to the more expensive alternative.

- **No human psychophysical data despite the primary motivating application**: The paper frames principal distortions in Section 4.1 as enabling more efficient psychophysical comparison of models to human observers. Yet no human threshold measurements are reported. The conclusion that "LGN and LN models are closest to human distortion thresholds" is drawn entirely from the reader's visual inspection of scaled distortions. While the paper notes this is "consistent with" Berardino et al. (2017) — which did include human data — making an empirical claim about proximity to human perception based on informal visual inspection without measured thresholds is a genuine gap. The direct predecessors (Berardino et al., 2017; Zhou et al., 2023) both include human data. Given that the paper's primary stated motivation is psychophysical model comparison, the complete absence of human measurements limits how much confidence one can place in the method's practical utility for that purpose.

### Minor

- **The 2 log₂(N) efficiency claim is speculative and conditional**: Section 4.1 states the claim explicitly as conditional ("If one could reduce the number of models by, say, a factor of two on each iteration…"), so the paper is honest about this. However, the "if" is doing significant work: whether a single pair of principal distortions reliably halves the model set in practice depends on model clustering, psychophysical noise, and FIM geometry — none of which is examined. Presenting this as a concrete, quantifiable advantage of the method (vs. the direct competitors' 2N and N(N+1)) risks overstating the gain. Even a simulation using existing log sensitivity ratios as proxy measurements would substantially strengthen this claim.

- **Fisher-Rao connection holds only in the pairwise special case**: Section 3.1 states the metric approximates the Fisher-Rao distance, but this approximation holds when ε₁, ε₂ are the generalized eigen-distortions of a specific pair of models. In the N-model principal distortion setting, ε₁, ε₂ are optimized for variance across all N models and are not, in general, the generalized eigen-distortions of any pair. The theoretical grounding through Fisher-Rao geometry is therefore pair-specific and approximate in the general case. This limitation is not clearly flagged.

- **Gaussian noise assumption not stress-tested**: The framework assumes additive isotropic Gaussian output noise. The paper acknowledges this simplification (Section 5) but provides no analysis of robustness to alternative noise models (e.g., Poisson, or noise profiles fit to actual neural systems). For a methods paper, some sensitivity analysis would be valuable.

### Trivial

- The choice of p=2 (variance) over p=1 or a minimax objective is justified by the PCA analogy but not by any analysis of when variance maximization could underperform (e.g., skewed distributions with one outlier model). A brief discussion would improve completeness.

---

## Nice-to-Haves

- Extending the framework to more than 2 principal distortions (a scree plot of cumulative variance explained would strengthen the PCA analogy and show how many distortions are needed to summarize inter-model variation).
- Showing optimization convergence plots or multiple initialization comparisons for at least one representative image to verify the non-convex optimization is reliable.
- Exploring the sensitivity of principal distortions to the choice of base image (e.g., naturalistic vs. uniform fields, high vs. low frequency content) to bound how image-dependent the conclusions are.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Optimization details absent from main text** (Harsh Critic point 3): The critic cites absence of convergence details from the main text, but the paper explicitly defers to Appendix B. Under the established rules, criticisms about missing appendix content are removed because the parser strips appendix sections from the submission text provided to reviewers.

- **Computational cost discussion** (Harsh Critic note): Concern about Jacobian computation cost for large architectures. Likely addressed in appendix; removed per same rule.

- **Section 4.2 choice of AlexNet/ResNet50** (Harsh Critic note): The critic acknowledges the architectural gap is well-documented and the paper already addresses this with ViT/EfficientNet supplementary results. Not a weakness at the level it was framed.

- **"Closest to human distortion sensitivity" claim is overclaimed** (Harsh Critic Section 4.1 note): Partially valid but weakened because the paper frames this explicitly as "consistent with Berardino et al.," not as a novel standalone claim. The weakness is already captured in the Major section under absence of human data.

---

## Novel Insights

The most genuinely novel observation — enabled specifically by the multi-model framework — is that adversarial training fundamentally reorients the principal axis of local geometry variation so that the dominant source of inter-model variation becomes *training type* rather than *architecture*, while stylized ImageNet training leaves the architecture-determined structure intact. This dissociation between two prominent training paradigms is not predictable from prior representational similarity results and suggests that adversarial training produces a qualitatively distinct local sensory geometry — a hypothesis that could be further tested psychophysically using the proposed method's distortions as probe stimuli.

---

## Suggestions

1. **Add a quantitative comparison against Zhou et al. (2023) pairwise application**: For the N=4 early visual model case, compute Zhou et al.'s 12 generalized eigen-distortions, then measure how much model-discriminating variance the 2 principal distortions capture relative to the full set. Report a scalar (e.g., fraction of total inter-model variance, or clustering accuracy) rather than visual inspection.
2. **Either include a small pilot psychophysics experiment or reframe the primary motivation**: The claim that the method enables efficient psychophysical comparison is central to Section 4.1. Either demonstrate this with threshold measurements for at least one pair of distortions, or explicitly reframe the paper as a computational tool for DNN analysis with psychophysics as a suggested future application.
3. **Label the 2 log₂(N) claim explicitly as a conjecture**: Replace "the total number of stimuli scales as 2 log₂(N)" with "we conjecture the total number of stimuli scales as 2 log₂(N), provided each round of principal distortions successfully partitions the model set in half."
4. **Add a brief robustness analysis of the Fisher-Rao approximation**: For at least one model pair, compare the actual Fisher-Rao distance against the metric m_{u,v} when using principal distortions (vs. when using the generalized eigen-distortions of that pair) to quantify how much the approximation degrades in the N-model setting.

---

## Score and Decision

**Calibration:**

- **kvByNnMERu** (Estimating Shape Distances on Neural Representations, scores: 8, 10, 6, 6, avg ~7.5, Accept poster): This paper has strong theoretical bounds and novel estimators; the paper under review is similarly principled but has weaker validation (qualitative vs. quantitative results). Placing below this anchor.

- **k9t8dQ30kU** (Task structure and representational geometry, scores: 5, 6, 8, 8, avg ~6.75, Accept poster): Also mostly empirical with limitations to specific architectures; paper under review is comparable but with a cleaner mathematical framework. Roughly peer.

- **vWRwdmA3wU** (Similarity metrics for neural representations, scores: 6, 6, 8, 5, avg ~6.25, Accept poster): Analyzes existing metrics; paper under review proposes a new method, giving it slightly more novelty.

- **D6pHf8AiO7** (FIM estimation for pruning, scores: 3, 3, 5, 6, avg ~4.25, Reject): Rejected for insufficient novelty and weak empirical validation. The paper under review is significantly more rigorous than this anchor.

- **RwCxxaHvyp** (Manifold Learning via Foliations, all 5s, Reject): Also uses FIM variants but rejected; the paper under review has cleaner contribution and more interesting experiments.

**Assessment:** The paper makes a genuine, mathematically sound contribution — extending pairwise generalized eigen-distortions to N models in a principled way — and produces the interesting adversarial training finding. Relative to accepted papers in the representational geometry space (kvByNnMERu at ~7.5, k9t8dQ30kU at ~6.75), the paper under review falls slightly below due to the absence of quantitative evaluation and human data, but clearly above the rejected anchors. The primary motivation (psychophysics efficiency) is undemonstrated, and the evaluation is exclusively qualitative, which is a real but not fatal limitation for a methodology paper.

**Evaluation on key axes:**
- **Originality**: Moderate — natural extension of Zhou et al. (2023), but the N>2 generalization is non-trivial and the adversarial training finding is novel.
- **Importance**: Solid — model comparison is an active area and the framework is applicable to a broad class of representations.
- **Claims vs. support**: Partially mismatched — the psychophysics efficiency claims are unsupported; the DNN architectural/training claims are well-supported.
- **Soundness**: Good — the mathematics is correct; the experiments are replicated across images and initializations.
- **Clarity**: Good — Figure 1 is effective; PCA analogy works.
- **Value to the community**: Moderate — the DNN adversarial training finding has implications for the representational similarity community; the framework is computationally accessible.

Final score: **6.0** — borderline accept, leaning toward accept. The mathematical contribution is clean and the adversarial training finding is novel and well-supported. Major weaknesses (no quantitative baseline comparison, no human data) are addressable and don't invalidate the core contribution, but the paper would be significantly stronger with them.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>