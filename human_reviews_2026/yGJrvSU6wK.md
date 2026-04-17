# COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Implicit neural representations (INRs) have recently emerged as a powerful paradigm for modeling data, offering a continuous alternative to traditional discrete signal representations. Their ability to compactly encode complex signals has led to strong performance across a wide range of computer vision tasks. In previous studies, it has been repeatedly shown that INR performance has a strong correlation with the activation functions used in its multilayer perceptrons. Although numerous competitive activation functions for INRs have been proposed, the theoretical foundations underlying their effectiveness remain poorly understood. Moreover, key challenges persist, including spectral bias (the reduced sensitivity to high-frequency signal content), limited robustness to noise, and difficulties in jointly capturing both local and global features. In this paper, we explore the underlying mechanism of INR signal representation, leveraging harmonic analysis and Chebyshev Polynomials. Through a rigorous mathematical proof, we show that modulating activation functions using a complex sinusoidal term yields better and complete spectral support throughout the INR network. To support our theoretical framework, we present empirical results over a wide range of experiments using Chebyshev analysis. We further develop a new activation function, leveraging the new theoretical findings to highlight its feasibility in INRs. We also incorporate a regularized deep prior, extracted from the signal via a task-specific model, to adjust the activation function parameters. This integration further improves convergence speed and stability across tasks. Through a series of experiments which include image reconstruction (with an average PSNR improvement of +5.67 dB over the nearest counterpart across a diverse image dataset), denoising (with a +0.46 dB increase in PSNR), super-resolution (with a +0.64 dB improvement over the nearest State-Of-The-Art (SOTA) method for 6X super-resolution), inpainting, and 3D shape reconstruction we demonstrate the novel proposed activation over existing state of the art activation functions. The official implementation and experimental results are available at our [project page](https://cosmo-inr.github.io/).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides a theoretical analysis of activation functions in implicit neural representations (INRs) using harmonic analysis and Chebyshev polynomials, proving that modulating activations with a complex sinusoidal term achieves more complete spectral support. Based on this insight, the authors design a new activation function and integrate a regularized deep prior to improve convergence and stability, achieving state-of-the-art performance across image reconstruction, denoising, super-resolution, and 3D shape tasks.

### Strengths
1. The paper presents a well-motivated and theoretically grounded idea, offering solid mathematical analysis based on harmonic analysis and Chebyshev polynomials to explain and improve activation functions in INRs. The theoretical reasoning is rigorous and clearly supports the proposed approach.

2. The proposed activation function is empirically strong and broadly applicable, achieving state-of-the-art performance across multiple tasks, including image reconstruction, denoising, super-resolution, inpainting, and 3D shape reconstruction, demonstrating both effectiveness and generalizability.

### Weaknesses
1. Lack of efficiency analysis. The method introduces a more complex activation, which typically increases forward/backward costs and gradient computation time. The paper should report training/inference time, FLOPs, and throughput comparisons to quantify the overhead and efficiency–accuracy trade-off.

2. Missing NeRF experiments. Given INRs’ broad use in neural rendering, the absence of NeRF-style evaluations (novel-view synthesis or radiance field fitting) limits the demonstrated scope. At least a small-scale NeRF benchmark would strengthen claims of generality.

3. Underspecified hyperparameters. In Eq. (9), the hyperparameters a and b are not justified. 

4. Notation and formula presentation. In line 185: “as shown in 6.” should be “as shown in Equation (6):”.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a spectral analysis of implicit neural representations (INRs) by examining how activation functions affect frequency support under composition. The authors introduce a complex-modulated root-raised-cosine activation function (COSMO-RC) and show, through harmonic analysis based on Chebyshev polynomial decomposition, that stacking this activation leads to complete spectral support while maintaining controlled bandwidth growth. They further introduce learnable modulation parameters, regulated via sigmoids, enabling the network to adaptively allocate frequency content during training. The proposed activation improves the ability of INRs to capture high-frequency details across tasks such as image reconstruction, super-resolution, denoising, and 3D shape representation.

### Strengths
The paper has numerous strengths. I will list them in order of relevance:

## Strong Theoretical Foundation:

The paper provides a rigorous spectral analysis of nonlinear activations in INRs, supported by harmonic decomposition and Chebyshev-based reasoning. The theoretical results justify the proposed activation design and its ability to achieve complete spectral support under composition.

## Well-Motivated Activation Design:

The COSMO-RC activation is not heuristic; it is grounded in a principled construction combining a root-raised-cosine envelope with complex modulation. The paper shows that this design offers both controlled bandwidth and the capacity to generate rich high-frequency content—addressing spectral bias.

## Adaptive Frequency Allocation Mechanism:

The integration of learnable modulation parameters, regulated via sigmoids, provides a simple yet effective way to adapt the activation’s spectral behavior during training. The paper shows this contributes to improved convergence and flexibility across signals with different frequency demands.

## Broad Empirical Validation:

The experiments span multiple INR tasks (image representation, denoising, super-resolution, and 3D shape representation) and consistently demonstrate meaningful improvements over baselines. Visual quality improvements, particularly in fine detail reconstruction, are compelling.

## Clarity and Presentation Quality:

The paper is well-written and organized. The motivation, theory, and empirical evidence are easy to follow, and the activation design is explained in a manner that will be accessible to the INR community. Figure 3 in particular is very elucidative.

## Impact and Relevance:

The contribution is significant for the INR field, where activation-level spectral control remains an active research challenge. The proposed method is general and may influence future work on frequency-aware neural representations.

## Evaluation:

The paper is a strong contribution and is ready for acceptance. The insights about the frequency attenuation because of the zero coefficients are very elucidative and the proposed solution is solid.

### Weaknesses
Just two weaknesses:

## Presentation of the Prior Knowledge Embedder:

I understand that it is used as a black box from which COSMO-RC builds on, but readers not familiar with the concept may struggle to understand the proposed model architecture. A succinct intuitive description is sufficient to address this problem. Specifically, a description about how the latent code is generated and how it is mapped to the activation function parameters.

## Missing reference:

* **Novello, Tiago, et al. "Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.**

A contextualization in the Related Works section is sufficient. The current evaluation in the paper is good enough for publication.

### Questions
Maybe it is a typo, but Equation (7) uses $f(x)$ to represent the original activation function. That is confusing since $f_\theta(x)$ is previously used to represent the INR. I suggest changing the symbol for better clarity.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes a structural limitation of common symmetric activations in implicit neural representations (INRs): by Chebyshev/harmonic arguments, purely even or odd nonlinearities suppress alternating spectral components, creating “frequency blind spots.” To address this, the authors propose complex sinusoidal modulation, multiplying a base activation by $e^{j\zeta x}$ so that the resulting real/imaginary parts no longer share the same parity-induced zeros. This yields a principled way to restore coverage of previously attenuated harmonics.

The work instantiates the idea with a raised-cosine family (COSMO-RC) and introduces practical stabilization: normalizing complex activations to the unit circle, reading out the real part at the network output, and bounding the modulation parameters $(T,\zeta)$ via a lightweight conditioning module. Conceptually, the contribution is a clear frequency-domain diagnosis and a simple, architecture-agnostic mechanism that can wrap existing INR backbones to mitigate symmetry-induced spectral gaps.

### Strengths
1. Principled spectral analysis. The paper offers a clear, mathematically grounded explanation (via Chebyshev/harmonic analysis) of why purely even/odd activations suppress alternating frequency components, and motivates the proposed complex modulation as a targeted remedy.

2. Well-motivated stabilization of complex activations. The paper introduces pragmatic design choices—unit-circle normalization, real-part readout, and sigmoid-bounded parameterization of   $(T,\zeta)$—that address numerical/optimization issues and make the proposed activation practically trainable.

### Weaknesses
1. No validation on inverse problems.
INR research is expected to test inverse problems, not only forward fitting. There are no experiments on NeRF-style inverse rendering or PINN-style PDE inference. It is unclear if the activation helps optimization and identifiability in these settings.

2. Narrow harmonic analysis.
The “harmonic distortion” study covers only raised-cosine, sine, Gaussian, and ReLU. It omits modern INR activations such as WIRE and FINER. Without these, the analysis may not generalize to current practice.

3. Incomplete comparative characterization and ablations.
Ablations are missing for the prior-knowledge embedder (remove/replace/placement, sensitivity), and for layer placement and $(T,\zeta)$ settings. This makes the source of improvements hard to isolate.

### Questions
1.Inverse-problem validation.
Add at least one inverse task (e.g. NeRF task at NeRF Synthetic Dataset ). If compute is limited, include one case of the dataset for inverse example.

2.Diversity of harmonic analysis.
a. Extend the harmonic/Chebyshev or empirical spectral analysis to WIRE and FINER. 
b. State whether WIRE/FINER exhibit the same alternating-component suppression; include side-by-side plots.

3.Ablations.
Add ablations for the prior-knowledge embedder (remove or add the prior-knowledge embedder to SIREN/WIRE/FINER methods).
Report per-iteration runtime, total training time, peak memory.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces COSMO-INR, a new framework for Implicit Neural Representations (INRs) that is motivated by a novel theoretical analysis of INR activation functions. The authors leverage harmonic analysis and Chebyshev polynomials to argue that common symmetric (odd or even) activation functions suffer from "spectral attenuation," where either the odd or even polynomial coefficients are zero, limiting the signal information propagated through the network . Authors presents a solution network, called COSMO, modulating activation function with a complex sinusoidal term. With this network, full spectral support is preserved. Some good results were obtained for image reconstruction examples.

### Strengths
-- the paper presents a novel and Insightful Theoretical Analysis. The core strength is the mathematical analysis of activation functions using Chebyshev polynomials. This "spectral attenuation" is a novel and clearly articulated problem.
-- good empirical results obtained in experiments. 
-- architecture is designed in a good engineering way, connecting the theory to a practical solution.

### Weaknesses
-- justification is weak when describing complex modulation. The network has a goal for real to real mapping, but the paper states that real part of the output is extracted in the final layer.  not clear, paper fails to prove that the real part coefficients are alone sufficient and non-zero for all n.
-- COSMO-RC is compared to INCODE but COSMO-RC already includes INCODE as embedder.  
-- pinpointing examples are not producing good results. similarly for 3D occupancy. Justification are missing. why? weaknesses? limitation?

### Questions
weaknesses are self-contained and they all should be considered as questions to answer.

### Soundness
3

### Presentation
4

### Contribution
3
