# Self-Supervised Evolution Operator Learning for High-Dimensional Dynamical Systems

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
We introduce an end-to-end approach to learn the evolution operators of large-scale non-linear dynamical systems, such as those describing complex natural phenomena. Evolution operators are particularly well-suited for analyzing systems that exhibit spatio-temporal patterns and have become a key analytical tool across various scientific communities. As terabyte-scale weather datasets and simulation tools capable of running millions of molecular dynamics steps per day are becoming commodities, our approach provides an effective tool to make sense of them from a data-driven perspective. The core of it lies in a remarkable connection between self-supervised representation learning methods and the recently established learning theory of evolution operators. We deploy our approach across multiple scientific domains: explaining the folding dynamics of small proteins, the binding process of drug-like molecules in host sites, and autonomously finding patterns in climate data. Our code is available open-source at: https://github.com/pietronvll/encoderops.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an end-to-end deep learning framework for learning evolution operators of high-dimensional dynamical systems. It establishes a novel connection between self-supervised contrastive learning and operator learning theory.

Through experiments, the paper demonstrates that the proposed method can extract meaningful dynamical modes that reflect physically relevant processes such as folding-to-unfolding transition, molecular binding, and climate oscillation. It shows that these learned representations can be reused effectively across related systems.

### Strengths
The paper established a novel and rigorous theoretical connection between evolution operator learning and self-supervised contrastive learning. A major contribution is the scalability of the proposed framework to high-dimensional system. Empirically, the model demonstrates stable training and meaningful spectral decomposition in different scientific domians, even in terabyte-scale dataset.

### Weaknesses
(1) The finite-sample analysis is not provided;
(2) The paper lacks a clear comparison against other (Koopman) operator-learning approaches (e.g., kernel EDMD, Neural Operator frameworks).

### Questions
(1) Your Lemmas assume the evolution operator $E$ is Hilbert-Schmidt, yet you acknowledge this is often violated by deterministic dynamical systems. The Lorenz '63 system in your experiments is deterministic (or near-deterministic). So, what happens to the approximation quality when $E$ is not Hilbert-Schmidt?
(2) In the deterministic case, the Koopman operator acts on delta functions. How does your bilinear model $\langle \phi(x_t), P\phi(x_{t+1}) \rangle$ approximate this structure?
(3) How does the approximation error related to the latent dimension?
(4) In the climate experiment, how does your method perform if one applies a kernel-based approach directly to the raw 29,040-dimensional input features, instead of using the learned 128-dimensional embedding? Such a comparison would clarify if standard dimensionality reduction techniques could achieve similar results.
(5) Could you discuss any scenarios or system characteristics under which the learned representations may fail to transfer effectively?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an end-to-end, self-supervised approach to learning evolution operators for high-dimensional nonlinear dynamical systems, with a particular focus on scientific domains such as molecular dynamics and climate modeling. The authors connect contrastive, self-supervised representation learning techniques with the operator-theoretic framework, providing theoretical grounds for their approach and demonstrating its equivalence to established least-squares operator learning methods. Comprehensive experiments are reported on protein folding, molecular binding, and climate data, with claims of scalability, interpretability via spectral decomposition, and transferability of learned representations.

### Strengths
- The authors make a deliberate connection between self-supervised contrastive learning and classical operator-theoretic approaches (notably, the least-squares estimation of evolution operators). This is not only highlighted at an intuitive level but also underpinned with theoretical results.
- The paper supports claims with diverse, high-dimensional benchmarks, spanning molecular simulations (protein folding, ligand binding) and challenging climate datasets.
- The spectral decomposition of the learned operator yields interpretable, physically relevant modes, e.g., hydrogen bonding patterns in protein folding.

### Weaknesses
- While the core mathematical exposition is sound, there are places where notation could be clarified for easier accessibility:
  - In Lemma 2 and Appendix A, the formula for the predictor $P_*$ involves both $C_X^{-1}$ and $C_Y^{-1}$, but the mapping from the loss gradient to the closed-form $P_*$ is sketched rather than fully elucidated.
  - Equation (9) relates the action of E to the covariance of futures, but implementation choices for non-stationary or non-ergodic systems are not fully discussed.
- The limitations section (Conclusion) briefly notes the Hilbert-Schmidt assumption and qualitative evaluation. However, broader issues are not meaningfully discussed. For example, the treatment of deterministic versus stochastic systems, failure cases when the operator is not Hilbert-Schmidt, and scalability bottlenecks in training or memory for extremely large state spaces are not addressed.
- While the online/offline covariance ablation adds value, ablations for the choice of network architectures (GNN vs. CNN vs. MLP), size/sparsity of $P$, or impact of history/context window on model performance would strengthen the empirical narrative.

### Questions
- Can the authors provide (or reference) quantitative metrics for spectral decomposition accuracy on real-world, high-dimensional datasets, or propose benchmarks for such evaluation? For example, can clustering purity or precision/recall be reported for folded/unfolded distinctions in protein folding, or event detection in climate data?
- Can the authors report on representation transfer performance when the target system is more dissimilar to the source (e.g., transfer across molecular families, not just ligands, or different climate regimes)? What are the limits of transferability in practice?
- In extremely large-scale systems, does online or EMA-based covariance computation pose bottlenecks or stability issues? Can sparse or approximate methods be used safely?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an encoder-only method for learning evolution operators of nonlinear dynamical systems by connecting self-supervised learning with operator theory. The approach is demonstrated across multiple domains, including protein folding, molecular binding, and climate pattern discovery.

### Strengths
1. The theoretical analysis provides a thorough and rigorous discussion of the paper's core claims.

2. The experimental design is strengthened by the use of multi-disciplinary datasets, ensuring the scenarios are both diverse and representative.

3. The integration of self-supervised representation learning with evolution operator theory represents a novel and promising research direction.

### Weaknesses
1. When the target function *f* does not lie in the linear span of the encoder, it is unclear how the method should be adjusted. The paper lacks discussion on this point—for instance, whether simply increasing the embedding dimension would suffice to satisfy this assumption.

2. The paper lacks practical guidance on how to choose the embedding dimension in real-world systems.

3. The authors should more clearly distinguish their approach from existing methods that combine deep learning with Koopman theory or DMD, and better highlight what additional problems their method can solve.

### Questions
1. How does your method handle cases where the target function *f* is not in the encoder's linear span?

2. What is the key advantage of your method over existing Koopman or DMD-based approaches, and what new problem does it enable you to solve?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a self-supervised approach to learn the evolution operators, which characterize the temporal dynamics of complex stochastic/deterministic systems. Different from the forecasting or reconstruction type models, the method uses a contrastive self-supervised loss from the spectral contrastive learning to estimate the transfer operators and spectral decomposition directly from data.  The main contribution is to theoretically show the equivalence between the self-supervised loss and minimizing the operator regression objective, connecting the least-square estimator, VAMP-2 score, and HS norms. Experiments on three domains, including protein folding, molecular binding, and climate dynamics, show the effectiveness and generalization of the proposed method.

### Strengths
1. Theoretically explain the operator learning theory and self-supervised contrastive learning, which provides justification using contrastive objectives in scientific dynamic systems. 
2. The method avoids computationally expensive matrix inversions and is implemented with simple matrix multiplications and covariance updates, which are very suitable for large-scale high-dimensional data. 
3. The paper demonstrates the effectiveness through diverse and convincing experimental validation.

### Weaknesses
1. Although the qualitative results, such as the eigenfunction visualizations, look good. The paper does not report any standardized quantitative metrics and comparisons with the baseline on some experiments. 
2. This paper is missing an ablation study that compares different encoder architectures or embedding dimensionalities. 
3. Although admitted by authors that the Hilbert-Schmidt assumption is often not held in deterministic systems. The paper lacks a discussion of how the performance will degrade when the assumption fails. ( Can be one of your ablation studies.)

### Questions
1. It would be a benefit to elaborate more on Equation 8. How can this be viewed as a contrastive learning paradigm? Should state more if positive and negative pairs are not explicitly defined.

### Soundness
2

### Presentation
3

### Contribution
2
