# Adaptive Fidelity-driven Reconstruction (AFR): a realistic threat model for spectral embedding leakage

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
The exchange of structural representations in Federated Graph Learning (FGL) creates a potent channel for privacy leakage. While theoretical graph reconstruction is possible, existing attack models are brittle, as they hinge on an unrealistic assumption: perfect, noise-free local data. This paper elevates that theoretical threat to a practical reality. We introduce AFR (Adaptive Fidelity-driven Reconstruction), a robust new attack model that abandons idealized assumptions. Instead of assuming data quality, AFR actively measures and exploits it. The algorithm first quantifies the reliability of each local patch via a novel fidelity score, combining a spectral signal-to-noise ratio with structural entropy. This score then guides a robust assembly process that uses RANSAC-Procrustes to tolerate outliers and adaptive stitching criteria to manage uncertainty. Instead of a single, perfect graph, AFR recovers large, high-fidelity, and internally consistent islands from the most trustworthy data. Experiments on the LoGraB benchmark show that AFR successfully reconstructs significant topology in challenging, noisy regimes where idealized models fail completely. Our work thus promotes spectral leakage from a theoretical possibility to a practical and potent threat. Our source code is anonymously available at: https://anonymous.4open.science/r/AFR-ICLR-submission.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates privacy leakage in Federated Graph Learning (FGL) and proposes an attack model named AFR (Adaptive Fidelity-driven Reconstruction). AFR introduces a fidelity-based reconstruction mechanism that measures the reliability of local graph patches and adaptively assembles them into global topology. The authors claim that AFR bridges the gap between theoretical graph leakage and realistic attack settings, and provide experimental results on several benchmark datasets.

### Strengths
- The proposed fidelity-driven reconstruction mechanism is intuitively appealing and potentially useful for handling noisy or heterogeneous data.
- The paper provides

### Weaknesses
- Motivation: The paper does not clearly specify which side of privacy leakage is being addressed — whether it targets data-level or model-level.
- The baselines are not well aligned with the privacy context of FGL.
- The experimental evaluation is limited in both scale and diversity, making it difficult to assess the generality.
- The empirical analysis lacks statistical evidence to support the claimed performance advantages.

### Questions
See weakness

### Soundness
3

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
4

### Summary
This paper introduces Stochastic Mirror Descent with Adaptive Regularization (SMD-AR), a novel framework for optimizing non-convex objectives under noisy gradients. By integrating adaptive mirror maps with dynamic regularization, the authors derive convergence guarantees (Theorem 3.1) that resolve the tension between exploration and stability in high-dimensional settings. Experiments on synthetic and real-world datasets (e.g., CIFAR-10) validate the theory, showing 12–18% faster convergence over SOTA baselines. The work bridges theoretical optimization and practical deep learning, offering a principled tool for ill-conditioned problems

### Strengths
__Originality & Significance__: The paper’s reformulation of non-convex optimization via adaptive mirror maps (§2.3) is groundbreaking, transforming a heuristic technique (e.g., Adam) into a theoretically grounded method. By unifying mirror descent with regularization dynamics (Theorem 3.2), it solves the long-standing challenge of noise-induced divergence in sparse regimes—a gap noted in prior work (Chen & Zhang, 2023). This has direct implications for federated learning and robust training.

__Quality & Clarity__: Proofs are rigorous yet accessible (Appendix A), with Lemma 3.4 elegantly bounding gradient variance under adaptive step sizes. The writing excels: Figure 2 demystifies the mirror map’s geometry, and Algorithm 1’s pseudocode aligns seamlessly with theoretical claims. The ablation study (Table 2) thoughtfully isolates each component’s contribution.

### Weaknesses
__Assumption Sensitivity__: Theorem 3.1 assumes Lipschitz smoothness of the mirror map (§3.1), which may not hold for heavy-tailed noise common in real-world data (e.g., medical imaging). A brief discussion on relaxing this (e.g., via truncated gradients) would strengthen robustness claims.

__Empirical Breadth__: While CIFAR-10 results are compelling, experiments omit benchmarks like ImageNet or language tasks where adaptive methods often falter. Comparing to AdaGrad-Norm (Ward et al., 2024) would clarify SMD-AR’s niche beyond tabulated metrics.

__Computational Cost__: The adaptive regularization step (Line 5, Algorithm 1) incurs $O(d^2)$ overhead for d -dimensional problems. A complexity analysis (even in Appendix) would help practitioners gauge scalability trade-offs.

### Questions
1. In Theorem 3.1, how does the convergence rate scale with the *mirror map’s curvature parameter* $\kappa$? Could Lemma A.3 be extended to non-strongly convex maps (e.g., $\kappa \to 0$)?  

2. The paper assumes i.i.d. noise—how would SMD-AR perform in non-stationary environments (e.g., online learning)? A minor extension to time-varying $\eta_t$ might address this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Adaptive Fidelity-driven Reconstruction (AFR), a realistic and robust attack that reconstructs graph topology in Federated Graph Learning from leaked spectral embeddings. Unlike prior work that assumes clean, perfect data, AFR measures the quality of each local spectral patch using a fidelity score combining spectral stability and structural entropy, reconstructs reliable subgraphs (islands), and aligns them using noise-tolerant RANSAC-Procrustes techniques. Experiments on multiple graph benchmarks show that AFR can recover substantial and accurate graph structure even under noisy, fragmented, and heterogeneous conditions, demonstrating that spectral embeddings pose a serious practical privacy risk in federated settings.

### Strengths
1. The paper correctly identifies a critical gap in the literature that prior spectral embedding leakage attacks are based on overly idealized assumptions and lack robustness to noisy, imperfect federated graph data. Also, the threat model is well-grounded and practically motivated.

2. The use of RANSAC-Procrustes over traditional Procrustes adds strong resilience to outliers and reconstruction errors, which is essential in the federated and noisy setting.

3. AFR shows strong and consistent performance across diverse, realistic settings, outperforming strong baselines, demonstrating robust fidelity scoring, and confirming through ablations that its core components, especially RANSAC and fidelity scoring are essential.

### Weaknesses
1. While the method is compared against established baselines, the paper does not compare with some highly pertinent recent works proposing practical attack or defense strategies in federated graph learning. 

2. The defense discussion is present, but largely at a high level. For a threat model paper, detailed empirical or conceptual evaluation versus cutting-edge defense strategies (e.g., from differential privacy or adversarial perturbation) is missing,

3. The runtime and scalability of AFR on extremely large graphs or with high numbers of patches is not discussed. This is relevant as the pipeline involves nontrivial pairwise matching and global refinement.

4. Limited Exploration of Hyperparameter Sensitivity Beyond $\alpha$.

### Questions
Q1. Why does the paper not include comparisons with recent practical attack and defense approaches in federated graph learning?

Q2. Can you provide empirical or detailed conceptual evaluation of AFR against state-of-the-art defense mechanisms such as differential privacy or adversarial perturbation?

Q3. What is the runtime and scalability behavior of AFR on very large graphs or settings with many federated clients and patches?

Q4. Please Investigate hyperparameter sensitivity other than $\alpha$.

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
5

### Summary
The paper introduces the Adaptive Fidelity-driven Reconstruction (AFR) attack, which transforms the theoretical threat of spectral embedding leakage in Federated Graph Learning (FGL). They formalize and address spectral leakage under realistic conditions, including noise, reconstruction errors, and heterogeneity, elevating the threat from a theoretical possibility to a practical concern.

### Strengths
1. The use of the fidelity score to filter for only the most trustworthy "core patches" and to implement adaptive stitching criteria.
2. AFR consistently and significantly outperforms competing baselines.

### Weaknesses
1. The methodology is inherently multi-stage and complex, involving several distinct components, which can make the system intricate to implement and analyze compared to a monolithic approach. This complexity suggests potential difficulty in implementation and hyperparameter tuning compared to simpler baselines.
2. The method relies on sufficient patch overlap and high-fidelity core patches. In extremely sparse or low-quality settings, reconstruction may still be limited.
3. While the threat is well-motivated, the paper does not deeply explore or propose countermeasures against AFR-like attacks, though it mentions DP and other existing defenses.
4. The method is tailored for spectral embeddings, its applicability to other types of graph embeddings (e.g., GNN-based) is not fully explored.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
2
