## Human Reviewer 1

### Summary
This paper focuses on the vulnerability of diffusion models in 3D molecular generation caused by intermediate-time drift and extremely narrow effective domain. The authors first formally propose the data distribution hypothesis of "dense-centralized (DC) structure," and based on this, analyze the overshoot problem in back-inference (Equations (6)(7)). They then propose a pluggable bias-correcting sampling framework, DIST (Diffuse and Steer): at intermediate time points, replication-perturbation forms local "batches," which are scored using pilot inference and inconsistent/low-quality batches are filtered out to obtain the corrected intermediate distribution q_{t}^{c} before continuing back-sampling. Empirical results on QM9 and GEOM-Drugs show improved stability and legitimacy for various backbones (EDM/GeoLDM/RADM), and claim a nearly 50% reduction in the average number of steps.

### Strengths
1.Abstracting the molecular distribution into a DC structure of "multiple narrow peaks + low-density spacing" provides a theoretical explanation and intuitive illustration of the trajectories intersecting in the intermediate distribution and the misleading effects of the mean-based score field.

2.Without altering the backbone weights and hyperparameters, the algorithm directly selects the best and discards the worst in the inference process, empirically demonstrating stable performance improvements on multi-backbone architectures and two datasets.

3.The author provides a step count calculation and shows in a table the significant decrease in average steps.

4.Cor. 3.1 gives the conclusion that "closer intermediate distributions lead to closer final distributions" as the TV distance does not increase; Prop. 3.1 gives the upper bound of the error after selective correction as dependent on α(τ), β(τ) and the conditional TV deviation.

### Weaknesses
1.Cor. 3.1 actually utilizes the general fact that "any Markov kernel is 1-Lipschitz (non-expansion) on TV" (k∈[0,1] often only takes the value 1), but does not guarantee strict contraction (k<1). Calling it "TV–contraction" is misleading to the reader into thinking it means "inevitable contraction." It is suggested to change it to "non-expansive step / TV non-expansion," and clarify the typicality of k=1.

2.The upper bound of Prop. 3.1 depends on the quality of the true distribution, but in practice the true distribution is unavailable. The authors approximate it with "pilot inference results (stability/legitimacy)"; however, the consistency of this approximation and the confidence bound (the estimation error of α, β) are not analyzed, so the upper bound is difficult to instantiate.

3.The paper calculates the average number of steps (e.g., (T−t)/|B|+t) based on "replicating batches up to t + full replay of a small number of pilot samples + continuing only for batches that pass the threshold." However, the description of the complete reverse cost of the pilot samples is somewhat vague. The "pilot cost of discarded batches" should also be included in the total budget before comparing the end-to-end total cost with the baseline's fixed 1000 steps. It is recommended to provide the precise accounting formula and pseudocode in Appendix D, and report the wall-clock time with the same hardware and parallelism.

4.Only rule-based indicators such as stability, legitimacy, and uniqueness are reported. It is recommended to add: skeleton diversity , ring structure and heteroatom distribution, QED/SA/synthetic feasibility, chirality and geometric configuration preservation, etc., to reflect the impact of the "screening + replication" strategy on diversity and pattern coverage.

5.The settings and scales of "radius r", the trade-off between quality and diversity in "replication count/perturbation intensity", the specific definition and threshold selection of pilot score s_{j} (which seems to be based on stability/legitimacy in the final draft), and the sensitivity to random seed/batch size are currently less detailed in the main text.

### Questions
1.The abstract claims that "computational overhead is reduced to nearly half that of the standard number of steps." Please provide the total end-to-end time (including pilot cost and the overhead of dropped batches) and the equivalent GPU hours, and compare it with the baseline on the same hardware and concurrency level.

2.The image comparison in Fig. 1 is intuitive, but could the differences be exaggerated due to variations in image task semantics, etc.? Could you provide a 1D/2D toy density comparison to more purely represent the mechanism of "multiple narrow peaks + overlap + overshoot"?

3.The reverse update of Equation (5) uses ϵ_{θ} to approximate the score; in molecular tasks, the SE(3) isovariant network or coordinate alignment is often used to reduce ambiguity. The authors' subsequent experiments included isovariant and non-isovariant backbones, but did they examine the sensitivity differences of DIST under canonicalization/coordinate normalization schemes?

4.Please change "TV–contraction" to "TV non-expansion" and clarify that κ=1 in general. Furthermore, providing sufficient conditions under which κ<1 would be more convincing.

5.Please provide accurate accounting (including pilot costs for failed batches) in Appendix D, and supplement with wall-clock and throughput to avoid substituting actual time with just "steps".

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
This work focuses on the "dense-concentrated (DC) structure" inherent to molecular data: chemically valid structures form sharp, densely packed peaks in the representation space, separated by regions of near-zero density. This structure makes diffusion modeling fragile. To mitigate this, this paper propose DIST (DIffuse and STeer), a selective correction method. DIST filters and rescales intermediate distributions during inference, steering trajectories toward the valid molecular peaks. Experimental results demonstrate that DIST consistently enhances performance.

### Strengths
1. The identification of the "dense-concentrated structure" in molecular data as a critical challenge is valuable.

2. The theoretical analysis enhances the plausibility of the proposed approach.

### Weaknesses
1. A thorough survey and comparison to the literature on "exposure bias in diffusion models" (such as [1], [2], [3], and recent works)  are highly necessary. The proposed solution shares conceptual similarities with existing methods—for example, Proposition 2 in [2] reaches conclusions analogous to this work’s Cor. 3.1 and Prop. 3.1, albeit with different metrics. The absence of a clear comparison from existing methods makes it difficult to assess the novelty and uniqueness of its contribution.

2. Lack of experimental validation for the issue of dense-concentrated data: The phenomenon observed in Table 1 mirrors exposure bias and which is also found in image generation with diffusion models. To substantiate the arguments in Section 3.1, additional experiments should be conducted to demonstrate that dense-concentrated data exhibit more pronounced exposure bias than smoother distributions. 

3. Clarity and rigor in writing need improvement: 
    - Line 230: The derivation of \(\|\nabla \log p(z_t)\|\) and Equations 6, 7 needs rigorous mathematical justification. 
    - Section 3.1 claims that the DC-structure causes the "reverse update to step past the peak and cross into high-density opposite regions." I suggest including a 1D Gaussian mixture simulation example, illustrating how specific choices of \(\sigma_*\), \(\Delta\), and \(m_k\) lead to sampling falling into low-density regions.
    - I suggest including a pseudocode implementation of the DIST algorithm for reproducibility.
    - Minor typo: Line 306 "the the reverse process".

### Questions
1. The DIST algorithm involves multiple manually tuned hyperparameters (intermediate timestep t, filtering threshold, batch division, pilot sample ratio). What guidelines exist for tuning them? If time permits, an ablation study for each parameter is recommended.

2. Intuitively, transforming the data (e.g., stretching the "x-axis") could reshape the distribution to be sparser and smoother, eliminating the dense-concentration. Is the DC-structure an inherent property of molecular data, or an artifact of representation? A discussion on this would strengthen the work’s motivation.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper addresses the instability of diffusion models in 3D molecular generation. The authors identify that molecular data exhibits a dense-concentrated (DC) structure—valid molecules correspond to narrow, isolated peaks in distribution space separated by low-density regions. Standard diffusion models often drift into invalid regions due to this concentration, leading to unstable or chemically invalid molecules. To address this, the authors propose DIST, a plug-in corrective sampling module. DIST selectively corrects intermediate distributions during reverse diffusion by evaluating sample batches, discarding off-distribution trajectories, and steering remaining samples back toward high-density, valid regions. They provide theoretical guarantees and show that DIST improves molecular validity and stability across several backbone models on QM9 and GEOM-Drugs datasets, while cutting inference timesteps nearly in half.

### Strengths
This paper is well motivated and has some theoretical insights. The empirical results show consistent improvement across diverse backbones and datasets.

### Weaknesses
1. While the application to 3D molecules is new, the idea of filtering intermediate states resembles rejection or guidance sampling.
2. No comparison to alternative corrective techniques (e.g., score rescaling, gradient correction, or classifier guidance).
3. Scalability to very large molecules or protein-level systems is not demonstrated.
4. Dependence on heuristic thresholds (τ) and pilot sampling design may reduce reproducibility or generality.

### Questions
1. How sensitive is DIST’s performance to the choice of threshold and batch radius?
2. Is the theoretical bound (Proposition 3.1) empirically validated—e.g., by measuring total variation between qt and pt estimates?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3