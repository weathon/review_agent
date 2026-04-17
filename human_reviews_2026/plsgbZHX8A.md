# Bayesian Evidence-Driven Prototype Evolution for Federated Domain Adaptation

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Federated learning (FL), as a privacy-preserving distributed machine learning paradigm, enables clients to collaboratively train a global model without sharing local data. However, in real-world scenarios, domain shift caused by different source clients leads to structural discrepancies in the feature space, resulting in performance degradation of the global model. Although existing prototype-based FL methods offer improvements in cross-domain feature alignment, they still struggle to adapt to dynamic semantic structures and fail to continuously respond to the changing semantic separability and variance structure during training. To address this, we propose FedPTE, an FL framework with prototype topology evolution. Specifically, FedPTE treats prototype clusters as variable topological units, employing Bayesian Gaussian Mixture Models and marginal likelihood ratios on the server to perform probabilistic inference, which enables adaptive structural adjustments. Meanwhile, FedPTE introduces a stability constraint mechanism to balance the adaptability of topological evolution and training stability. By conducting prototype topology-aware contrastive learning on clients, it enhances the discriminability and cross-domain consistency of features. Experimental results demonstrate that FedPTE achieves superior performance across multiple cross-domain datasets, showcasing its strong expressiveness and generalization capability in heterogeneous domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes FedPTE, a framework designed to address domain shift among clients. FedPTE dynamically adjusts the structure of global prototype clusters using Bayesian Gaussian Mixture Models (BGMM) and marginal likelihood ratios, determining when to split or merge prototypes based on statistical evidence. A stability constraint mechanism is introduced on the server to prevent unstable topology changes, while topology-aware contrastive learning on clients enhances feature discriminability and cross-domain consistency. Experiments demonstrate that FedPTE achieves generalization and stability.

### Strengths
1. The work targets a critical pain point in FL——domain shift under heterogeneous data, making the motivation both practical and research-relevant.
2. FedPTE maintains good performance under moderate privacy noise, suggesting potential for deployment in privacy-sensitive fields.

### Weaknesses
1. Relies on Gaussian mixture assumptions: performance may drop if data significantly deviates from this distribution.
2. Server-side Bayesian inference and covariance inversion have $O(d^3)$ complexity, which can be expensive in high-dimensional spaces.
3. Although probabilistic evidence is used, the paper lacks a deeper theoretical analysis of convergence guarantees or probabilistic bounds.
4. Lack of large-scale or real-world deployments where communication and noise issues are more severe.

### Questions
Please see the weaknesses.

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
3

### Summary
This paper addresses domain shift in federated learning by proposing FedPTE, a framework that dynamically evolves prototype topology through Bayesian inference. The key innovation is treating prototype clusters as variable topological units that can split or merge based on statistical evidence from Bayesian Gaussian Mixture Models with Normal-Inverse-Wishart priors. The framework includes penalty mechanisms to ensure stability and uses prototype topology-aware contrastive learning on clients to enhance feature alignment across domains.

### Strengths
1. The use of Bayesian inference with NIW priors for prototype topology evolution is mathematically principled and moves beyond simple averaging or static clustering approaches.
2. FedPTE consistently outperforms baselines, achieving 93.98% on Digit and 73.01% on Office, with particularly notable improvements on challenging domains.
3. The penalty terms for split/merge operations effectively balance adaptability with training stability, addressing a key challenge in dynamic prototype methods.
4.  The paper addresses important aspects like communication overhead, computational complexity, and privacy protection with differential privacy experiments.

### Weaknesses
1. The method introduces several hyperparameters (βsplit, βmerge, κ0, ν0, S0)
2.  The O(d³) complexity for marginal likelihood calculations could become prohibitive for high-dimensional features, though this isn't thoroughly discussed.
3. While the Bayesian framework is principled, the paper lacks convergence guarantees or theoretical analysis of when/why the topology evolution helps with domain shift.
4. Some of deign choices needs to be further clarified, e.g., Why use FINCH clustering initially? How sensitive is performance to the NIW prior initialization?

### Questions
Please find in the above weaknesses.

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
4

### Summary
FedPTE frames federated domain adaptation as evidence-driven prototype topology evolution. The server uses BGMM with NIW prior and marginal likelihood ratio to split/merge global prototypes, adding a stability penalty; the client uses topology-aware contrastive loss for training. This has achieved performance improvements on other feature shift datasets such as Digit/Office.

### Strengths
1. The article mainly focuses on the fact that static clustering methods are difficult to solve the semantic separability and variance structure changes between prototype clusters in feature offset scenarios, and from the experimental results, it has excellent performance, and the experiment is complete.
2. The ablation experiment verified the effectiveness of the effects of each component, which solved my confusion to a certain extent.

### Weaknesses
1. The causes and impacts of semantic separability and variance structure changes under feature transfer have not been fully explained. In addition, how should I understand the concept of “numerical coincidence” mentioned in Section 3.2? Why does this lead to situations where spatial proximity but different semantics occur?
2. The ablation experiments were not performed on the same experimental dataset, which makes me speculate about the method's effectiveness. For example, based on Table 3 and Figure 4, I cannot judge the effect of the stability component without adding $P_{split}$ and $P_{merge}$ and adjusting their corresponding parameters.
3. In Formula 2, the calculation of local prototypes is done by generating different feature centers through FINCH clustering, while the left side is the local prototypes of samples of the same type, which are not equal in number.
4. Though the authors have extended experiments on mixed data from more clients in the appendix, the main results are conducted in a small-scale environment (5 clients), limiting the reliability of the results and raising potential scalability concerns of the proposed method. 
5. Comparison with some latest SOTA methods on federated domain shift is not included, for example: 
[1]  "FedPall: Prototype-based Adversarial and Collaborative Learning for Federated Learning with Feature Drift." ICCV, 2025.
6. The distinction between related work and other work should be further clarified.

### Questions
1. More ablation experiments are needed to verify whether the reference of the stability constraint mechanism will affect the specific performance.
2. What specific weaknesses are addressed relative to FedPLVM/MPFT? And what are the advantages of NIW proofs compared to previous heuristics?
3. "Numerical coincidence": Give an example where spatial proximity hides a semantic mismatch or provide insight into why this happens and how to prevent it through penalties.
4. Reports communication and runtime comparisons to prototype baselines.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FedPTE, a Bayesian prototype-topology–based federated learning (FL) framework designed to tackle domain shift and data heterogeneity in cross-domain FL. Traditional prototype-based FL methods rely on static clustering or averaging and fail to adapt to evolving class structures.
FedPTE models prototypes as dynamic topological units whose number and structure evolve via Bayesian Gaussian Mixture Models (BGMM) and marginal likelihood ratio tests under a Normal–Inverse–Wishart (NIW) prior. The server decides whether to split or merge prototype clusters based on evidence strength, guided by penalty constraints to maintain stability. Clients perform prototype topology–aware contrastive learning, aligning local representations with global prototypes.

Experiments on Digit (MNIST/SVHN/USPS/Synth/MNIST-M), Office (Amazon/Caltech/DSLR/Webcam), and multiple medical imaging datasets (Camelyon, ISIC, Polyp, Prostate) show that FedPTE outperforms baselines such as FedProto, FedPLVM, FedHEAL, and FPL. The paper also reports ablation studies, non-IID analysis, privacy perturbation effects, and multi-domain scalability. Overall, FedPTE claims better adaptability, robustness, and generalization across heterogeneous domains.

### Strengths
1 Original idea: Introduces Bayesian evidence to dynamically evolve prototype topology—bridging probabilistic inference with federated representation learning.

2 Comprehensive experiments: Covers IID/Non-IID, unseen domain, medical imaging, and privacy scenarios.

3 Strong empirical performance: Consistent gains over FedProto, FedPLVM, FPL across all datasets.

4 Well-designed ablations: Component and hyperparameter studies are clear (Figure 3–4, Table 3).

### Weaknesses
1 Limited novelty beyond existing prototype frameworks.
While Bayesian modeling adds rigor, the overall structure (prototype aggregation + contrastive alignment) resembles prior works like FedPLVM and FPL. The innovation lies mainly in the split/merge inference.

2 Assumption of Gaussian prototypes.
Real FL data distributions (e.g., in medical or digit datasets) may be non-Gaussian; the NIW assumption can oversimplify multi-modal or skewed clusters.

3 Hyperparameter sensitivity.
Penalty coefficients (β_split, β_merge) and evidence thresholds are empirically set; no adaptive rule or theoretical guidance is provided.

4 Computational overhead.
Marginal likelihood evaluation (O(d³)) and frequent topology updates could become expensive for large models or high-dimensional features, though claimed “acceptable” (Table 10) without quantitative scaling analysis.

5 No theoretical convergence guarantee.
The stability of prototype evolution across rounds is discussed qualitatively but lacks formal proof or bound.

### Questions
1 How sensitive is FedPTE to NIW hyperparameters (κ₀, ν₀, S₀)? Are priors fixed or updated adaptively across rounds?

2 Could the split/merge evidence thresholds be learned dynamically rather than hard-coded?

3 Does the BGMM assumption hold when feature distributions are highly non-Gaussian (e.g., with ReLU activation skew)?

4 How does FedPTE behave when one domain is highly dominant (client imbalance)?

5 Is the Bayesian evidence computation parallelized, and what is its runtime complexity per round in large-scale FL?

### Soundness
3

### Presentation
3

### Contribution
3
