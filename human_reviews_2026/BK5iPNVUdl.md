# Synthetic Image Detection via Curvature of Diffusion Probability Flows

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Synthetic image detection (SID) faces two major challenges: high computational cost from reconstruction-based methods and insufficient generalization. To address these issues, we propose a SID paradigm that leverages the ODE formulation of diffusion models. Instead of reconstructing images, our method analyzes probability flow trajectories from data distributions toward a Gaussian prior. We theoretically relate discrete step distances on the Wasserstein manifold to the kinetic energy of the probability flow. We further show empirically that trajectory deviation statistics derived from these distances correlate with reconstruction error and that real and synthetic images differ most in the early half of the diffusion inversion. In this regime, real images tend to exhibit higher curvature variance with occasional extreme deviations, whereas synthetic ones follow smoother and more consistent trajectories. Building on this observation, we introduce curvature features of probability flow trajectories as a discriminative signal for SID. To the best of our knowledge, this is the first work to exploit probability flow curvature for this task. Extensive experiments demonstrate that our method generalizes robustly to unseen models and achieves state of the art results across multiple benchmarks, while requiring less than half the FLOPs of reconstruction based detectors that perform full diffusion inversion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the task of synthetic image detection. Specifically, the authors investigate the effect of curvature information from diffusion model ODE trajectories for this purpose. The proposed method is evaluated with several competitors and the authors claim SOTA results with a more efficient approach compared to diffusion inversion.

### Strengths
1. Overall, the pipeline is clearly described and the methodology is clear.
2. The attempt at a more rigorous and precise treatment of the ODE trajectories and curvature is appreciated.
3. The authors show improved performance on standard benchmarks. TPRs @ 5% FPR are also included for a more thorough analysis.
4. The authors provide ablations on NFEs and training sets.

### Weaknesses
1. I found the paper vague or unclear on various aspects. Please see the Questions section below for specific concerns.

2. The Section on interpretability (line 458) is poorly explained and not particularly convincing. The authors mention "In synthetic
images, areas with inconsistent lighting, incorrect
perspective, or structural anomalies receive the highest saliency". However, I struggle to see any examples of this in Figure 5.

### Questions
1. The abstract reads "We show that the discrete-step distances on the Wasserstein manifold inherently encode reconstruction error" and discussion in Section 4.1 mentions that existing observations regarding the overall reconstruction error, when combined with Theorem 2, imply that synthetic trajectories tend to be straighter than those of real images (line 236). As I understand, Theorem 2 links W2 between intermediate marginals to the velocity, saying nothing about the reconstruction error obtained via inversion. Can the authors clarify?

2. As I understand, curvature features are extracted over the context of the full trajectories (Equation 16). Some discussion on this choice and whether your findings connect with existing work (e.g., [1]) would be appreciated. Have you investigated whether specific regions/time-steps are consistently more informative?

3. The overall method includes curvature features and wavelet features. Does the method not work without them? An ablation study on this, e.g., curvature only or wavelet only or both would be helpful.

4. How should the results in Tables 1 and 2 be interpreted? As I understand, you report conditional metrics on synthetic data from various models. This not sufficient to judge performance without also quantifying performance on real data. While you have provided details about the synthetic data, I couldn't find any information about benchmarking on real data, which is just as important in binary classification. Could you please clarify?


[Minor] Consider merging the Appendix (in supplementary material) with the main paper for better readability.

[Minor] The paper is mathematically dense so consider being a bit more explicit with notation and definitions. For example, it would be helpful to include the definition of geometric curvature and the derivation for the particular ODE resulting in Equation 12 in the Appendix.

[1] Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution, 2024

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel approach for Synthetic Image Detection (SID) that leverages the curvature characteristics of diffusion probability flow ODE trajectories rather than relying on full reconstruction. The idea is that real and synthetic images follow distinct diffusion trajectories — real images exhibit higher curvature variance and higher-energy paths, while synthetic ones produce smoother, lower-energy trajectories during diffusion inversion. Building on the curvature analysis, the paper propose a classification pipeline for SID that leverages features based on a pseudo-Gaussian curvature descriptor and  wavelet features to capture fine-grained spatial cues. Experiments demonstrate strong performance over prior state-of-the-art methods, such as B-Free and FakeInversion.

### Strengths
1. Novel formulation of SID via optimal transport and probability flow curvature, linking reconstruction error to Wasserstein geometry and ODE kinetic energy.

2. Theorems 1–2 coherently connect Wasserstein bounds with velocity field energy, justifying curvature-based descriptors.

3. Convincing performance gains on the final benchmark

### Weaknesses
1. Potential overfitting or data leakage: The claim of generalization from training to unseen models is strong, but the text doesn’t detail whether prompt overlap or data source contamination might occur between training and evaluation sets and also the opensource model used for obtaining synthetic training data.

2. The jump from Eq. 10 (non-optimality term) to the claim that "synthetic images lie on manifold regions more easily represented by the model" lacks rigorous justification.

3. Table 2 shows that performance drops significantly with 5 NFEs and plateaus beyond 10, but there's no explanation of why or how this relates to the theoretical framework.

4. No comparative studies on the contribution to the performance from the curvature features vs that from the wavelet features. 

5. Although the paper claims “less than half the computational cost,” no empirical nor theoretical runtime or FLOPs comparison or estimation is provided.

### Questions
1. Choice of curvature: why choose the pseudo-Gaussian curvature? Fig 2(c) - mean of \tilde{k}_t - seems to also be a discriminative alternative?

2. Can you provide ablation studies separating the contribution of curvature features vs. wavelet features?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reconstruction-free detector for synthetic image detection (SID) that works in the probability-flow ODE view of diffusion models. Instead of inverting a generator, the method computes finite-difference curvature features along a short **noising** trajectory and fuses them with diagonal high-frequency wavelet components. The authors claim strong cross-model generalization to **unseen** generators and **lower compute than full inversion**, reporting average ACC/AUCROC of **0.939 / 0.981** and **TPR@5%FPR = 0.933** on a broad suite of models.

### Strengths
1. **Good performance across many generators with an “unseen model” protocol.** The main table trains all detectors on **SD+LAION** and evaluates on other generators, showing strong averages and near-perfect numbers on several diffusion families. This explicitly targets cross-model generalization.
2. **Comprehensive core experiments and useful ablations.** The paper provides a broad comparison table and a steps/dataset ablation. The ablation indicates that **10–15 ODE steps** are a good operating point, with degradation at **5 steps** and diminishing returns at higher counts.
3. **Clear intuition that is practically helpful.** The authors motivate curvature via an OT/Wasserstein view: they argue that reconstruction-error signals are encoded by probability-flow energy and that *real* vs. *synthetic* images diverge most in the earlier half of inversion; their features focus on this regime. The intuition is coherent and guides the design, even if it is not fully proved to be discriminative (see Weaknesses).

### Weaknesses
1. **Lack of rigorous theory that curvature itself separates classes.** The paper provides theoretical motivation (energy/trajectory analysis) but **does not** prove that curvature must discriminate real vs. synthetic. The discrimination claim is supported **empirically** (distributional observations and accuracy), not by a formal guarantee. 
2. **Missing comparisons to strong contemporary SOTA (FatFormer, NPR).** While **FatFormer** (CVPR’24) and **NPR** (CVPR’24) are acknowledged in related work, they are **not included** in the quantitative table. Both are directly aimed at generalizable SID and do **not** use diffusion inversion; including them would strengthen the SOTA claim.
3. **Compute claims lack concrete measurements.** The paper states it achieves “**less than half the computational cost of full diffusion inversion**,” but it does **not** provide wall-clock timings and FLOPs vs. other methods. This leaves efficiency claims difficult to verify.

### Questions
.

### Soundness
2

### Presentation
3

### Contribution
2
