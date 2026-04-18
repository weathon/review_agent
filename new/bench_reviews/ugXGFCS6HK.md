The paper content is provided in the prompt. Let me verify key claims by the harsh reviewer against the actual paper text provided.

Key verifications:

1. **"Metric" claim**: The paper says "For any pair of distortions u, v, the function m_{u,v}(·,·) defines a (pseudo-)metric" and notes in footnote that it's technically a pseudometric because m=0 doesn't imply I_A=I_B. The harsh critic claims the paper "blurs the distinction" between a direction-dependent scalar function and a "canonical metric on the full FIM manifold." Let me check this more carefully.

2. **"PCA analogy" claim**: The paper says "In analogy with principal component analysis, our method can be viewed as a dimensionality reduction technique that preserves as much of the variability in the local representational geometry as possible." The harsh critic says this is misleading because PCA is linear, orthogonal, etc.

3. **Efficiency argument**: The paper claims the method reduces the number of distortions from 2N or N(N+1) to just 2, with iterative pruning giving 2log₂(N) scaling. The harsh critic says this is speculative because the optimization doesn't target human alignment.

4. **Qualitative evaluation**: The paper relies heavily on visual inspection and log-ratio plots. The harsh critic says there's no quantitative validation.

5. **Local approximation**: The paper acknowledges this limitation but the harsh critic says it's not addressed experimentally.

6. **No baselines**: The harsh critic notes no comparison against eigen-distortions, generalized eigen-distortions, or simple alternatives.

Now let me synthesize these into a final review, keeping only well-grounded weaknesses.

Let me think about which criticisms are valid:

**Harsh Critic Point 1 (metric/PCA claim is under-justified)**: This is partially valid. The paper does call m_{u,v} a "metric on the local geometry" (Section 3.1 title) which is somewhat overclaimed since it's a pseudometric that depends on the choice of u,v. However, the harsh critic's claim that the paper is "circular" is wrong—the paper doesn't claim there's a single canonical metric space independent of u,v; rather, it defines m_{u,v} as a metric on FIMs for a fixed pair of directions, then optimizes over u,v. The PCA analogy is indeed loose, but the paper qualifies it as an analogy ("In analogy with principal component analysis"). The variance formulation in Eq. 4 is genuinely analogous to PCA in the sense of maximizing variance. The connection to Fisher-Rao is mentioned and deferred to the appendix, which the paper is transparent about. This is a valid minor criticism about overclaiming, not a fatal flaw.

**Harsh Critic Point 2 (efficiency gain is speculative)**: This is partially valid. The paper does acknowledge that this is a theoretical argument about stimulus efficiency, and indeed says "If one could reduce the number of models by, say, a factor of two on each iteration of this process, the total number of stimuli to be assessed scales as 2 log₂(N)." However, the paper's claim that the distortions are a "parsimonious choice of stimuli" is reasonable on its own terms—the optimization does select the 2 most discriminative directions. The criticism that this doesn't guarantee alignment with human judgments is valid but somewhat scope-creepy: the method is about model-model comparison, and the human comparison is presented as a future application. The recursive pruning argument is indeed speculative, but it's presented as a possibility ("If one could...") rather than a proven fact. This is a fair minor-to-moderate concern.

**Harsh Critic Point 3 (evaluation is qualitative)**: This is valid. The DNN experiments are indeed primarily qualitative—log-ratio plots and visual inspection of distortions. No quantitative spectral analysis, no comparison against baselines, no psychophysical data. However, the paper does show results across 100 images (Fig. 3E) with error bars, and provides supplementary figures. The early vision section explicitly states the qualitative nature and mentions future work. This is a valid major weakness but not fatal—the methodological contribution + interpretable findings still hold value.

**Harsh Critic Point 4 (local approximation not validated)**: Valid concern but the paper does explicitly acknowledge this limitation in Section 5. The paper is not hiding this. The distortions are visualized at finite scales, and checking validity at those scales would strengthen the work. This is a moderate weakness.

**Harsh Critic Point 5 (no baselines)**: This is valid. The paper doesn't compare against simpler alternatives. Even showing that random distortions don't produce the same separation would help (though Fig. 2A partially does this). More direct comparison with generalized eigen-distortions would strengthen the paper. This is a moderate weakness.

Now let me assess the paper's quality relative to calibration papers:

- The "Perceptual Scales" paper (z7K2faBrDG) - Fisher information + perception, scores 8,5,5,3 (avg ~5.25) - accepted as poster
- The "Local Geometry of Generative Models" paper (etif9j1CnG) - local geometry + qualitative results, scores 8,6,6,6 (avg 6.5) - accepted as poster
- The "Emergent Geometry" paper (KJFyOwAnLR) - neural geometry + limited experiments, scores 3,6,3 (avg 4) - withdrawn/rejected
- The "Population Response Geometry" paper (kSISSDUYFh) - digital twins + geometry, scores 8,5,6 (avg 6.3) - accepted as poster
- The "Dropout/efficiency" paper (ADDCErFzev) - simple approach + brain alignment, scores 6,6,6,6 (avg 6) - accepted

This paper has a stronger theoretical foundation than most of these, but has weaker experimental validation than several. It's similar in profile to the "Perceptual Scales" paper—novel theoretical framework, interesting qualitative demonstrations, but limited human validation. That paper averaged around 5.25 and was accepted as poster. The current paper has a cleaner theoretical contribution but even less empirical validation. The "Local Geometry of Generative Models" paper also had qualitative evaluation concerns but got 6.5 average. The "Emergent Geometry" paper had limited experiments + overclaimed conclusions and got rejected (~4).

I think this paper is solid but has real gaps in evaluation. It's a methods paper that demonstrates proof-of-concept, with genuinely interesting findings (architecture vs training effects). The theoretical contribution (natural generalization from pairwise to multi-model comparison) is clean and meaningful. The lack of baselines and quantitative evaluation are the main weaknesses. I'd place it around 5.5-6 — a borderline-to-moderate accept as a poster.</think>Now I have enough calibration context. Let me write the final review.

---

## Summary

This paper introduces "principal distortions," a method for comparing the local geometries of multiple image representations simultaneously. It defines a pseudometric on log-sensitivity ratios derived from Fisher Information Matrices (FIMs) and finds the pair of distortion directions that maximize variance across N models, generalizing prior work on eigen-distortions (single model) and generalized eigen-distortions (pairwise comparison). The method is demonstrated on early visual system models and deep neural networks (AlexNet, ResNet50, and variants), revealing that architecture drives local geometry differences for standard-trained networks while training type (adversarial) drives differences for adversarially-trained networks.

## Strengths

- **Principled generalization of prior work.** The extension from pairwise (Zhou et al., 2023) to multi-model comparison via a variance-maximization objective is natural and well-motivated. The pseudometric in Eq. 3 has appealing properties—scale invariance, symmetry, and a connection to Fisher-Rao distance—that are clearly presented and properly situated relative to prior work (Sections 2–3).

- **Interpretable and novel findings.** The observation that principal distortions separate standard-trained networks by architecture (AlexNet more sensitive to high-frequency "stuff," ResNet50 more sensitive to smooth regions) but separate standard vs. adversarially-trained networks by training type, is a genuinely new finding about local geometry differences that, to my knowledge, has not been documented before (Section 4.2). The consistency across 100 base images (Fig. 3E) and across random initializations (Supp. Fig. SI.5) adds credibility.

- **Efficiency advantage clearly articulated.** The requirement of only 2 distortions regardless of N models (compared to 2N for eigen-distortions or N(N+1) for pairwise generalized eigen-distortions) is a concrete practical benefit, especially for psychophysical experiments where stimulus presentation time is limited (Section 4.1).

- **Clear presentation and good writing.** The paper is well-structured, with Figure 1 effectively illustrating how the method generalizes prior approaches. The mathematical development is accessible, and Section 2 provides a valuable tutorial on the FIM-based framework.

## Weaknesses

### Fatal

None.

### Major

- **Predominantly qualitative evaluation without quantitative baselines or human validation.** Across both the early vision and DNN experiments, conclusions rest on visual inspection of distortion images and log-ratio ordering of models. For the early vision models, the claim that "LGN model was the best... in terms of consistency with human distortion sensitivity" (Sec. 4.1) is supported only by the authors' subjective impression that certain distortions "appear visible," with no psychophysical measurements. For DNNs, the characterizations of ε₁ as "stuff/high-frequency" and ε₂ as "background/low-frequency" are not quantified (no spectral analysis, foreground/background masking metrics, or quantitative spatial statistics). Claims like "this qualitative difference... has not been documented" and "demonstrate that our method can reveal interpretable differences" are thus anecdotal rather than rigorously established. Importantly, the paper does not compare against natural baselines—e.g., whether simple analyses like singular vectors of concatenated Jacobians, random distortion pairs, or direct application of generalized eigen-distortions to all model pairs would reveal the same architectural/training separations. Fig. 2A shows principal vs. random distortions for the early vision models, which is useful, but no analogous comparison appears for the DNN experiments. Without such baselines, it is impossible to determine whether the observed separations reflect specific virtues of the principal distortion objective or generic properties of the FIMs that any reasonable probe would capture.

- **The scope of the "metric on local geometry" claim exceeds what is formally established.** The paper frames m_{u,v} as a "metric on the local geometry of image representations" (Section 3.1 title) and the optimization as "analogous to principal component analysis" (Section 1). However: (1) m_{u,v} depends on a chosen pair of directions (u,v), so it does not define a canonical metric on the space of FIMs; it defines a family of pseudometrics parameterized by direction pairs. (2) The PCA analogy is loose—the variance objective in Eq. 4 is a nonlinear function of the FIMs via square-root norms and logs, with no orthogonality constraints or guarantee that additional directions would capture residual variance in a well-defined sense. (3) The connection to the Fisher-Rao distance is noted to hold only for the two-model case with generalized eigen-distortions (Appendix A), not for the general N-model setting. These are not fatal issues—the variance-maximization objective is a sensible heuristic—but the paper's framing as a principled "metric" and rigorous "PCA analogue" overclaims what is established. The method is best characterized as a well-motivated but heuristic extension of generalized eigen-distortions to multiple models, and the paper would be stronger if it acknowledged this more directly.

### Minor

- **The theoretical efficiency argument for psychophysics (2 log₂ N scaling) is speculative.** The claim at the end of Section 4.1 that recursive pruning could reduce stimuli to 2log₂(N) assumes that models misaligned with humans also lie at extremes of the model-only variance in Eq. 4. No theoretical result or empirical evidence supports this assumption. Without human data, the claimed "dramatic improvement in efficiency" for model-human comparison remains conjectural. This is clearly presented as conditional ("If one could reduce the number of models by, say, a factor of two on each iteration"), so the paper is not misleading, but readers may overestimate the practical implications.

- **The local FIM approximation is used at finite perturbation scales without validation.** The method relies on the FIM, which captures local (infinitesimal) differential properties. However, the visualized distortions are scaled by factors of 20–100 (Section 4.1–4.2), and the method is proposed for psychophysics where distortions must be visible. The log-ratio predictions from the FIM-based local approximation may degrade at these scales, especially for highly nonlinear networks. While Section 5 acknowledges this limitation, it would strengthen the paper to quantify how the log-ratio ordering changes as a function of perturbation magnitude—e.g., by showing that the principal distortion directions at least remain stable within psychophysically relevant ranges.

- **Sensitivity to noise model and base image is unexplored.** The method assumes additive Gaussian noise to compute the FIM for deterministic networks (setting Σ = I), and operates on individual base images. No experiments test robustness of the principal distortions to variations in noise model, noise covariance structure, or base image selection. The 100-image averaging shows consistency of the log-ratio ordering, but not whether the distortion *directions* themselves are stable across images.

### Trivial

- The footnote defining pseudometric ("Technically, m_{u,v}(·,·) is a pseudometric because m_{u,v}(I_A, I_B) = 0 does not imply that I_A = I_B") is important but buried; it could be noted more prominently given its implications for what the method can and cannot distinguish.

## Nice-to-Haves

- Quantitative spectral or spatial characterization of the principal distortions (e.g., power spectral density, spatial localization metrics) to move beyond "visual inspection" claims.
- Comparison against baseline methods: random distortion pairs, eigen-distortions applied individually, or pairwise generalized eigen-distortions, to quantify the added value of the joint optimization.
- A small-scale human psychophysics experiment testing whether the early vision model predictions align with actual human thresholds for the principal distortions.
- Analysis of how much between-model variance is captured by the first 2 vs. first k principal distortion directions (the "scree plot" analog mentioned in Appendix A).

## Removed Points

- **"The metric is circular" (from Harsh Critic)**: The critic claims the paper "blurs the distinction" between a direction-dependent scalar and a canonical metric, creating circularity. This is a misreading. The paper does not claim a single canonical metric on the FIM manifold independent of directions; it defines m_{u,v} as a pseudometric for fixed directions and then *optimizes* over directions. The procedure is sequential (define metric → optimize over directions), not circular.

- **"The optimization is not principled for N>2 models" (from Harsh Critic)**: The critic claims the extension to N models is arbitrary because other choices (KL divergence, Wasserstein) could be equally reasonable. The paper explicitly acknowledges this in the text ("There are several other natural extensions... we focus on the case of maximizing the variance (p=2) and leave an exploration of other moments to future work"). The variance objective has a clear intuitive justification (maximize spread across models) and reduces to the two-model generalized eigenvalue problem, making it a natural choice even if not uniquely optimal.

- **"No human psychophysical data" presented as a fatal flaw (from Harsh Critic and Spark)**: The paper explicitly frames human experiments as future work and does not claim to have conducted them. The early vision model comparison is validated against prior qualitative findings from Berardino et al. (2017). The absence of human data is a limitation, but this is a methods paper demonstrating the method on computational models—demanding human experiments is beyond its stated scope. That said, the qualitative claims about which models match human perception (Sec. 4.1) are indeed unsupported without such data, which I note under Major weaknesses above.

- **"Adversarial examples exploit nonlinearity, so the FIM approach is contradictory" (from Harsh Critic)**: This is a mischaracterization. Adversarial examples exploit specific local directions of high sensitivity in a model—this is exactly the kind of local structure that FIM analysis captures. The irony the critic perceives does not exist; adversarial training changes the local geometry (making it more isotropic/smooth), and the principal distortion method correctly captures this change.

- **"Incomplete comparison with alternative multi-model methods" (from Neutral Reviewer)**: While the paper would benefit from such comparisons (noted under Major weaknesses), the reviewer's suggestion that the paper should compare against "concatenating pairwise generalized eigen-distortions" or "PCA on Jacobian spectra" is better framed as a nice-to-have. The paper's contribution is the new method; showing it works differently from a naive concatenation of pairwise results is informative but the method has theoretical motivation independent of such comparisons.

- **"The architecture-vs-training confound" (from Spark)**: The claim that separate experiments with different model sets are needed to dissociate architecture vs. training is reasonable as a suggestion but not a required experiment. The paper's observations (architecture dominates for SIN comparison, training dominates for AT comparison) are internally consistent and interesting; a joint experiment would strengthen but not invalidate the current findings.

- **"Limited model architectures" (from Human Finder)**: The paper tests AlexNet, ResNet50, EfficientNet, and ViT (Supp. Fig. SI.2), with SIN and adversarial training variants. While more diverse architectures would be welcome, the current set is adequate for a proof-of-concept demonstration.

- **Reproducibility concerns (implicit in various reviewers)**: The paper provides a complete mathematical framework and uses standard architectures (AlexNet, ResNet50). No reproducibility issues warrant flagging.

## Novel Insights

The key insight that principal distortions reveal a crossover in organizational structure—architecture-driven local geometry differences for standard/SIN-trained networks versus training-driven differences for adversarially-trained networks—is genuinely novel and suggests that the "source" of local geometry differences in neural networks is not monolithic but depends on what training modifications are made. This has implications for understanding what adversarial training actually does to representations at a local level (it fundamentally reshapes the sensitivity structure, overwhelming architectural differences), and what shape-bias training via SIN does not do (it leaves the local geometry architecture-dominated).

## Suggestions

- Add a quantitative baseline comparison: at minimum, show that random distortion pairs do not produce the same model separations as principal distortions for the DNN experiments (analogous to Fig. 2A for the early vision models), and show the variance captured by the top 2 principal distortion directions vs. top 2 generalized eigen-distortion or eigen-distortion directions.
- Quantify the spectral and spatial properties of the distortions (e.g., power spectral density, foreground vs. background energy distribution) to support the qualitative claims about "stuff" vs. "background" sensitivity.
- Tone down the "metric on local geometry" and "PCA analogue" framing to more accurately reflect what is formally established (a well-motivated heuristic) vs. what is analogy.

## Score and Decision

Calibration against similar papers:

- **Perceptual Scales (Fisher Info, perception)** — avg ~5.25, accepted as poster. Novel theoretical contribution + qualitative validation + some human data. This paper has a cleaner theoretical framework but no human data.
- **Local Geometry of Generative Models (local geometry, qualitative evaluation concern)** — avg 6.5, accepted as poster. New descriptors for local geometry with qualitative but interpretable findings.
- **Population Response Geometry (digital twins, geometry)** — avg 6.3, accepted as poster. Novel methodology + empirical findings about model-brain alignment gaps.
- **Emergent Geometry (neural geometry, limited experiments + overclaimed)** — avg 4, rejected. Broad claims with insufficient experimental support.
- **Shape/Dropout/brain alignment papers** — avg 6, accepted as poster. Simple methods, clear findings, brain data.

This paper falls between the "Emergent Geometry" (rejected, score ~4) and the "Local Geometry"/"Population Response Geometry" papers (accepted, ~6-6.5). It has a cleaner and more novel theoretical contribution than most, and genuinely interesting DNN findings. However, the evaluation is heavily qualitative with no quantitative baselines or validation, and the framing overclaims the theoretical foundations. This is a solid proof-of-concept methods paper that would benefit from more rigorous evaluation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>