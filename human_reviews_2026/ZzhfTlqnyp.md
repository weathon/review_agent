# FedCova: Robust Federated Covariance Learning Against Noisy Labels

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
This paper addresses the critical challenge of federated learning (FL) under noisy labels by exploiting intrinsic robustness grounded in covariance structures. Noisy labels in distributed datasets induce severe local overfitting and consequently compromise the global model in FL. Most existing solutions rely on selecting clean devices or aligning with public clean datasets, rather than endowing the model itself with robustness. In this paper, we propose *FedCova*, a noise-resistant federated covariance learning framework, to enhance the model's intrinsic robustness via a new perspective on feature covariances. Specifically, FedCova encodes data into a discriminative but resilient feature space to tolerate label noise. Built upon mutual information maximization, we design a novel objective for federated lossy feature encoding, which is driven solely by the feature covariances of different classes with an error tolerance term. Leveraging feature subspaces characterized by covariances, we construct a subspace-augmented federated classifier. FedCova unifies three key processes through the covariance: (1) training the network for feature encoding, (2) constructing a classifier directly from the learned features, and (3) correcting labels based on feature subspaces. The server aligns the federated classifier via covariance aggregation, which devices use to build local external correctors for relabeling, avoiding self-correction. We implement FedCova under heterogeneous data distribution across various noisy settings. Experimental results on CIFAR-10/100 and real-world noisy dataset Clothing1M demonstrate the superior robustness of FedCova compared with the state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces FedCova, a federated learning framework designed to address the challenge of noisy labels in distributed datasets. The framework integrates three components through a unified covariance-based approach: (1) lossy feature encoding via mutual information maximization with an error tolerance term, (2) intrinsic classifier construction using covariance-based MAP estimation with subspace augmentation, and (3) label correction using external correctors to avoid self-bias.

### Strengths
1. The main idea is interesting. The paper presents a principled information-theoretic approach by maximizing mutual information I(Z;Y) while focusing on covariance structures rather than mean statistics.
2.  The integration of feature encoding, classification, and label correction through covariance structures provides an elegant and cohesive solution, avoiding the need for auxiliary clean datasets or duplicate models required by existing methods.
3. The covariance transmission overhead is only ~1.4% of model parameters, making the approach practical for federated settings.

### Weaknesses
1. While the mutual information objective is well-motivated, the paper lacks formal convergence guarantees or theoretical bounds on the robustness to label noise.
2. The method introduces several hyperparameters (ε², α, d, ηc) that require tuning. 
3. The approach requires estimating and storing J×d×d covariance matrices. For problems with many classes (J>>100) or high feature dimensions, this could become prohibitive despite the claimed efficiency.
4. Only symmetric noise is considered; asymmetric or instance-dependent noise patterns are not evaluated. And the largest dataset (Clothing1M) still has relatively few classes (14)
5. There is no comparison with recent self-supervised or semi-supervised federated learning methods.

### Questions
1. How does the method perform under class imbalance, where some classes have significantly fewer samples?
2. Can you provide theoretical analysis on the convergence rate or sample complexity bounds?
3. How sensitive is the method to the correction schedule {Tc}?
4. Could the framework be extended to handle feature noise in addition to label noise?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenging problem of federated learning (FL) under noisy labels by proposing FedCova, a novel covariance-aware federated learning framework. The key insight is to leverage feature covariance structures rather than feature means to build robustness against label noise. The authors make three main contributions: (1) a lossy learning objective based on mutual information maximization that depends solely on class-conditional feature covariances with an error tolerance term, (2) a federated classifier alignment strategy via covariance aggregation with subspace augmentation, and (3) an external correction mechanism that uses global covariances to correct noisy labels while avoiding self-bias. Experiments on CIFAR-10/100 and Clothing1M demonstrate that FedCova outperforms state-of-the-art methods across various noise levels and heterogeneous data distributions.

### Strengths
1. The paper tackles the important challenge of noisy labels in federated settings, where labels from distributed edge devices are vulnerable to annotation errors, sensor faults, and adversarial attacks. The covariance-based approach naturally avoids bias from mislabeled data by focusing on feature structures rather than centroids. Unlike existing methods requiring warm-up rounds, clean public datasets, or extremely noisy devices, FedCova achieves robustness without these additional dependencies, making it more practical for real-world deployment.
2. The experiments systematically evaluate performance across multiple noise configurations, three datasets, and various data heterogeneity settings. FedCova consistently outperforms state-of-the-art baselines with substantial margins, particularly under severe noise. The extensive ablations validate each component's contribution, and supplementary analyses on orthogonality evolution, correction performance, and hyperparameter sensitivity provide valuable insights.

### Weaknesses
1. Insufficient privacy analysis and vulnerability to potential attacks: While the paper claims that covariance transmission poses lower privacy risk than raw features due to dimensionality reduction, this assertion lacks rigorous analysis. Recent work has shown that covariance matrices can still leak sensitive information about training data through gradient-based attacks or reconstruction methods. The paper provides no formal privacy guarantees, no discussion of differential privacy mechanisms for covariance sharing, and no evaluation against known federated learning attacks. Given that covariance matrices encode second-order statistics of local data distributions, they may be vulnerable to privacy attacks that could reconstruct class-specific feature patterns or infer properties of local datasets.
2. Unclear relationship between covariance learning and noise patterns: The paper does not clearly characterize what types of label noise the covariance-based approach can effectively handle. While avoiding class centroids helps with symmetric noise, it is unclear how feature covariances capture or mitigate different noise patterns. More critically, the method may fail under instance-dependent or semantic noise where mislabeling depends on sample-specific characteristics rather than class-level confusion. For example, if certain instances within a class are systematically mislabeled due to semantic ambiguity or fine-grained confusion, the class covariance would still be corrupted by these noisy samples. The paper only evaluates symmetric noise patterns and does not discuss or test robustness to asymmetric or instance-level semantic contamination.
3. High hyperparameter sensitivity with limited guidance: The method introduces multiple hyperparameters requiring careful tuning. While sensitivity analysis is provided for some parameters individually, their interactions are unstudied and optimal values appear dataset-dependent with inconsistencies between default settings and actual optimal values. The paper provides no principled guidelines for setting these parameters in new scenarios, and the correction schedule appears manually designed. For practitioners, the tuning burden may offset the benefit of avoiding additional resources.
4. Incomplete computational complexity analysis: The communication overhead analysis only accounts for parameter transmission but ignores crucial computational costs: covariance computation at each client, storage and inversion of covariance matrices, and correction round overhead. For datasets with many classes, managing multiple covariance matrices may be prohibitive for resource-constrained mobile devices. The paper provides no runtime comparisons with baselines, no analysis of how computation scales with the number of classes, and no discussion of parallelization. These missing details raise serious concerns about practical deployability.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the noise learning problem in federated learning by leveraging feature subspace covariance. The authors propose a subspace-augmented classifier that unifies data encoding, classifier construction, and label correction within a cohesive framework. The proposed method, FedCova, demonstrates superior performance across three benchmark datasets, highlighting its effectiveness in handling noisy data in federated settings.

### Strengths
1. The proposed solution is conceptually interesting and demonstrates creativity in its construction.
2. The notation throughout the paper is clear and mathematically rigorous.
3. The toy example provided in the Appendix effectively illustrates the main idea and makes the proposed solution easy to understand.
4. The ablation studies presented in the Appendix (e.g., Figures 2 and 3) show promising results and provide useful insights into the method’s behavior.
5. The analytical discussions in the Appendix are also promising and strengthen the overall contribution of the paper.

### Weaknesses
1. **Baselines:** More baseline methods should be included for a fair comparison. For instance, a straightforward approach to address the noise learning problem in federated learning is to apply existing noise learning techniques locally within the FedAVG framework. The authors should consider this variant to improve the completeness and credibility of the experimental evaluation.
2. **Ablation study:** The ablation study should analyze the impact of the number of clients on model performance, as this is a critical factor in federated learning settings.
3. **Notation:** The definition of X appears redundant and could be streamlined for clarity.
4. **Figures and algorithm presentation:** Figure 1 is difficult to interpret, and Algorithm 1 provides limited information since it mainly describes relabeling and aggregation steps in federated learning. It would be more effective to merge Figure 1 and Algorithm 1, presenting them as a single integrated schematic to better illustrate the proposed method.

### Questions
1. Assumption of zero means (Line 180): It is unclear why the authors assume that all component means are zero. In practice, it is reasonable to expect that some class distributions may have non-zero means. The paper should provide justification for this assumption or discuss its potential impact on the model’s validity.
2. Notation clarity: The meaning of \boldsymbol{Z}_{m}^{*} is not clearly defined. The authors should explicitly explain what this notation represents to improve readability and mathematical clarity.

### Soundness
3

### Presentation
3

### Contribution
3
