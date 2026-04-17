# Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges

- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
The Schrödinger Bridge provides a principled framework for modeling stochastic processes between distributions; however, existing methods are limited by energy-conservation assumptions, which constrains the bridge's shape preventing it from model varying-energy phenomena. To overcome this, we introduce the non-conservative generalized Schrödinger bridge (NCGSB), a novel, energy-varying reformulation based on contact Hamiltonian mechanics. By allowing energy to change over time, the NCGSB provides a broader class of real-world stochastic processes, capturing richer and more faithful intermediate dynamics. By parameterizing the Wasserstein manifold, we lift the bridge problem to a tractable geodesic computation in a finite-dimensional space. Unlike computationally expensive iterative solutions, our contact Wasserstein geodesic (CWG) is naturally implemented via a ResNet architecture and relies on a non-iterative solver with near-linear complexity. Furthermore, CWG supports guided generation by modulating a task-specific distance metric. We validate our framework on tasks including manifold navigation, molecular dynamics predictions, and image generation, demonstrating its practical benefits and versatility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the nonconservative generalized Schrödinger bridge (NCGSB), which covers a mathematical generalization of SB for nonconservative systems, increasing model flexibility. The noticeable changes include applying decaying energy with a time-varying state, representing the (stochastic) Lagrangian action. Furthermore, this Lagrangian formulation can also be applied to guided generation using a guidance function. The authors used a ResNet to model successive pushforwards for the discretized geodesic, enabling high-dimensional applicability, including image generation.

### Strengths
* The manuscript is clearly written and straightforward.
* The authors introduce a new generalization with purposeful intent and physical motivation, which could draw interest from audiences in many fields.
* The overall approach is sound and appropriate.
* The proposed method can be applied to various domains and is scalable.

### Weaknesses
* Lacking comparison results in the image domain. The overall volume and benchmark comparison for image-to-image translation can be considered insufficient. The authors are encouraged to review scalable SB methods for images from 2024 and 2025.
* I could not precisely follow how the proposed system ensures the uniqueness of the solution. Unlike in conservative systems, the rate of energy decay can be varied in nonconservative systems, so should there be a hyperparameter (or an assumption of 1) to ensure uniqueness? Also, the guidance function is another degree of freedom and seems to require careful design by practitioners based on the specific problem.

### Questions
* Could you please provide the FID scores for MNIST-to-EMNIST SB training?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the Non-Conservative Generalized Schrödinger Bridge (NCGSB) and the Contact Wasserstein Geodesic (CWG) framework to model stochastic processes with energy variation, overcoming the energy-conservation limitation of classical Schrödinger Bridges. By leveraging contact Hamiltonian mechanics, the authors extend the Wasserstein geometry to handle non-conservative dynamics and introduce a ResNet-based non-iterative solver with near-linear complexity. The approach also supports guided generation through metric modulation. Experiments on LiDAR manifold navigation, single-cell sequencing, and image generation demonstrate efficiency and accuracy improvements over strong baselines.

### Strengths
- The extension of SBs to non-conservative systems via contact geometry is quite interesting.
- The proposed method demonstrates faster convergence and better accuracy across diverse tasks.
- The paper is well-organized presentation with clear motivation and theoretical grounding.

### Weaknesses
- Limited large-scale validation:
Although diverse, most experiments remain moderate in scale. Evaluation on large-scale datasets or high-dimensional physical systems (e.g., long protein trajectories or large image benchmarks) would better demonstrate scalability claims.
- Ablation studies and interpretability:
The role of the contact energy term, and its impact on stability or generalization, is not extensively analyzed. A detailed ablation on the energy variation factor and metric modulation would strengthen the empirical evidence.

### Questions
- Could you clarify the numerical stability of the proposed solver, especially under long time horizons or many intermediate marginals?
- How sensitive is the method to the choice of potential function $U(x)$?
- Is the contact Hamiltonian augmentation equivalent to introducing an auxiliary energy channel in the latent space? If so, can this be viewed as a generalization of underdamped diffusion bridges?

- Could the authors provide an intuitive geometric visualization of how the energy-varying geodesic differs from a standard Wasserstein geodesic?

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
This paper addresses the Generalized Schrödinger Bridge (GSB) problem by relaxing certain energy-conserving constraints and instead applying a Hamiltonian mechanism to enable efficient training and inference. Furthermore, by adjusting the time-varying energy term, it allows for guided generation. The proposed method, called CWG, is implemented in the form of a ResNet, which effectively reduces training time. The authors also validate guided generation using interpolation when intermediate marginals are provided partially.

### Strengths
- Powerful Theoretical Basis: This study generalizes the conventional GSB framework from the perspective of Hamiltonian dynamics by introducing an energy-varying term. The theoretical analysis presented in the appendix provides a strong justification for the proposed methodology and effectively supports its validity.

- Compact Model Design: The proposed method is implemented using a ResNet architecture, which represents the most essential structure for expressing the Schrödinger bridge. This choice is appropriate, as additional modules can be incorporated when the task complexity increases.

- Efficient Training Speed: Compared to existing methods such as GSBM and DSBM, the proposed approach demonstrates superior performance and faster training. In particular, its advantage in training efficiency further enhances the novelty of this work.

### Weaknesses
#### Major Weaknesses
- Weak Presentation: Some important details necessary to fully understand the study may be missed unless the appendix is read carefully. Moreover, the core ideas of the proposed method appear too late in the manuscript. In my opinion, it would be better to move the Related Work section toward the later part of the paper and highlight the main contributions more prominently.

- Insufficient Emphasis on Mathematical Notation: The manuscript includes all key equations and attempts to distinguish important terms using various colors. However, the chosen colors are not sufficiently distinguishable, and the equations are interspersed throughout the text, which somewhat reduces readability. To emphasize key terms more effectively, it would be helpful to use a separate table or structured summary highlighting the core notations.

- Inconsistent Explanation of Intermediate Marginals: In Equation (9), the loss function appears to require intermediate marginal distributions. However, in the actual experiments, only a subset of them—or none at all—seems to be used. This inconsistency is not clearly emphasized. It would strengthen the manuscript to consolidate and highlight these explanations right after Equation (9), rather than scattering them across different sections.

- Novelty of Guided Generation: The authors emphasize guided generation as the main contribution of their work. In my view, guidance-based generation should ideally enable generation through indirect guidance rather than direct guidance toward an intended mode. In the current formulation and experiments, however, it seems that only interpolation between intermediate distributions during the optimal transport process is evaluated.

#### Minor Weaknesses

- In Appendix D.2, the notation for the intermediate distribution is inconsistent. It should be corrected from $\rho_n$ to $\rho_m$.
- In Appendix D.3, the time complexity analysis could be strengthened by comparing it with other existing methods. Such a comparison would help to better emphasize the novelty of the proposed approach.

### Questions
See Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
