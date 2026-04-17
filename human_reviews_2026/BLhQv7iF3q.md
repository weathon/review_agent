# Measuring Model Robustness via Fisher Information: Spectral Bounds, Theoretical Guarantees, and Practical Algorithms

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
The robustness of deep neural networks is critical for their deployment in safety-sensitive domains. This paper establishes a novel theoretical framework for quantifying model robustness through the lens of Fisher information. We first start with the known conclusion that maximizing the KL divergence of the posterior probability is equivalent to minimizing half the Mahalanobis distance defined by the Fisher Information Matrix (FIM), and further reveal that the FIM is equal to the variance of the input Jacobian matrix. Based on this insight, we propose the FIM's principal eigenvalue (or its reciprocal) as a principled robustness metric. We derive closed-form spectral bounds for common architectural components (e.g., ReLU, convolution) and theoretically compare the robustness of VGG, ResNet, DenseNet, and Transformer. To enable scalable computation, we resort to efficient algorithms, including power iteration and randomized Hutchinson, to estimate the robustness metric. Furthermore, we propose to use Hutchinson and finite differences to achieve robust estimation in a black-box setting. Extensive experiments validate our theoretical claims and demonstrate the metric's utility in predicting adversarial vulnerability. Code: https://anonymous.4open.science/r/8F4D7E6R/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an alternative approach to quantifying the robustness of deep learning models by leveraging Fisher information.

### Strengths
The paper establishes an interesting connection between robustness and Fisher information, offering a novel theoretical perspective. Additionally, the authors provide a practical method for estimating the proposed robustness measure, which could be useful for empirical evaluation.

### Weaknesses
* The authors criticize attack-dependent metrics (lines 40–45); however, the proposed method remains data-dependent. Would the reported robustness values change if additional data were included in the CIFAR-10/100 datasets?

* Can the proposed method be applied to commonly used models listed on RobustBench [1], such as WRN-34-10, XCiT-L12, Swin-L, or ConvNeXt-L? Demonstrating this would strengthen the paper’s practical relevance.

* The metric defined in Eq. (19) is not a conventional robustness measure. Typically, robustness is reported in terms of accuracy against adversarial examples. Could the authors provide a conversion table or mapping between their proposed metric and standard robustness metrics?

* RobustBench ranks models based on their accuracy under AutoAttack across various defense algorithms. Could the authors provide a comparison table similar to RobustBench to verify whether their proposed metric yields a comparable ranking of models?

* The use of PGD and CW attacks with only 20 steps is not sufficiently strong for a robust evaluation. The authors should include results using stronger attacks (e.g., AutoAttack or multi-step PGD variants). Furthermore, the experiments were conducted on only 500 examples, which limits the generality and significance of the reported findings.

[1] Croce, Francesco, et al. "Robustbench: a standardized adversarial robustness benchmark." arXiv preprint arXiv:2010.09670 (2020).

### Questions
Please refer to the points raised in the Weaknesses section above.

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
3

### Summary
The paper proposes a new metric to quantify the adversarial robustness of neural networks and algorithms for computing it. The metric is based on the Fisher information matrix and on its spectral norm. The effectiveness of the norm and of the algorithms is evaluated on various benchmarks.

### Strengths
- The proposed metric based on the Fisher Information matrix has a clear link with the robustness of the model

- The proposed algorithms seem sound, and their effectiveness is evaluated on more benchmarks and against different existing metrics

### Weaknesses
- In the related works, the authors claim that no existing works have established the link between the Fisher Information Matrix (FIM) and the robustness of the model. However, this is not correct, as this link was established, for instance, in [Shi-Garrier, Loïc, Nidhal Carla Bouaynaya, and Daniel Delahaye. "Adversarial robustness with partial isometry." Entropy 26.2 (2024): 103] or [Shen, Chaomin, et al. "Defending against adversarial attacks by suppressing the largest eigenvalue of fisher information matrix." arXiv preprint arXiv:1909.06137 (2019).]

- Theorem 2 is unclear to me. In fact, if f(x) is the input of the softmax, this is generally deterministic in standard neural networks, so it is unclear where the stochasticity of f comes from.

- Computation times of the proposed algorithm are similar to those of the other existing metrics. Therefore, the advantage of the proposed algorithm/metric compared to existing metrics based on the Lipschitz constant is unclear to me. It would be nice to illustrate the advantages empirically or at least in a discussion.

### Questions
- Can you please clarify the differences with the related works mentioned above?

- Can you clarify the role of f(x) in Theorem 2 and where its randomness comes from? A formal definition would help

- Can you comment on the advantages of the proposed metric compared to those based on the average Lipschitz constant?

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
The authors propose an information-theoretic framework to evaluate the robustness of neural networks based on the geometry of deep learning input-output manifolds. The authors provide a theoretical proof that the largest FIM eigenvalue encodes the worst-case sensitivity. The authors provide abundant theoretical analysis and confirm the effectiveness of the proposed robustness evaluation by numerical comparisons in various scenarios.

### Strengths
1. The paper generally reads smoothly.
2. Abundant work has been done analyzing the upper bounds of spectral norms of basic components of DNNs.
3. The paper discusses the relations between the proposed evaluation FIM eigenvalue and other classic evaluation metrics.

### Weaknesses
1. This paper bases the robustness of neural networks on the geometry of input-output manifolds, but do different inputs affect the geometry in the first place?
2. This may be minor. The authors repeated the same comparison with related work in the introduction on attack-dependent metrics and heuristic theoretical bounds.
3. Authors did not define $\|F\|_2$ in the second page.
4. Power iteration is used to find eigenvalues of matrices, which is supposed to take O(n^2) for dense matrices. Is this algorithm efficient enough when facing large-scale problems?

### Questions
1. Is Theorem 1 a well-known fact? Can you please provide the source?
2. Are the numerical results of robustness evaluation performed on the optimally trained model?
3. In page 3 the authors conclude that the robustness ranking is DenseNet121 < VGG16 < ResNet18 ≤ ViT-B-16, which is not exactly the same order as shown in table 3?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a unified framework for quantifying the robustness of deep neural networks using the Fisher Information Matrix (FIM). The key idea is that the largest eigenvalue of the FIM (or its reciprocal) captures the model’s worst-case sensitivity to input perturbations and thus serves as a principled robustness metric.

The authors:
- Derive spectral bounds and theoretical connections between FIM, Jacobian variance, Lipschitz constant, and other robustness metrics such as CLEVER.
- Provide efficient algorithms (power iteration, Hutchinson approximation) for scalable computation in both white-box and black-box settings.
- Conduct experiments on CIFAR-10, CIFAR-100, ImageNet, and medical datasets to validate the metric’s correlation with adversarial vulnerability and architecture-level robustness comparisons (VGG, ResNet, DenseNet, ViT).

### Strengths
- The connection between KL divergence, Fisher information, and robustness is rigorously derived and bridges geometric and probabilistic interpretations.
- Unlike attack-dependent measures (PGD, CW), this approach is attack-agnostic and interpretable.
- The approach works for both white-box and black-box scenarios, with efficient approximations for large models.
- The paper includes layer-wise spectral bounds and architecture-level robustness comparisons.
- Extensive experiments show consistent correlation with established robustness metrics.

### Weaknesses
- Although theoretically elegant, it is not yet evident how well the proposed metric can predict real-world robustness beyond controlled settings.
- While algorithms are efficient in theory, runtime comparisons show significant overhead versus CLEVER or Lipschitz methods; large-scale applicability (e.g., full ImageNet models) may still be limited.
- Since FIM depends on the data manifold, robustness comparisons across datasets are not always consistent.
- Experiments focus mostly on classification tasks; applications to other modalities (e.g., NLP, multimodal models) are not explored.
- The reported “robustness ranking” (e.g., DenseNet < VGG < ResNet ≤ ViT) might depend heavily on spectral norm estimation assumptions and lacks uncertainty quantification.

### Questions
- How stable are the FIM eigenvalue estimates across different random seeds or mini-batches?
- Can the proposed robustness metric predict generalization to out-of-distribution shifts, or is it limited to adversarial perturbations?
- How does the metric behave under adversarial training; does it improve monotonically with increased robustness?
- Could this framework be extended to parameter-space Fisher information (e.g., K-FAC approximations) for joint training and robustness evaluation?
- What is the computational bottleneck in practice - is it gradient computation or the spectral estimation step?

### Soundness
3

### Presentation
2

### Contribution
2
