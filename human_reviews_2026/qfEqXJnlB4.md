# Memory-Statistics Tradeoff in Continual Learning with Structural Regularization

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
We study the statistical performance of a continual learning problem with two linear regression tasks in a well-specified random design setting. We consider a structural regularization algorithm that incorporates a generalized $\ell_2$-regularization tailored to the Hessian of the previous task for mitigating catastrophic forgetting. We establish upper and lower bounds on the joint excess risk for this algorithm. Our analysis reveals a fundamental trade-off between memory complexity and statistical efficiency, where memory complexity is measured by the number of vectors needed to define the structural regularization. Specifically, increasing the number of vectors in structural regularization leads to a worse memory complexity but an improved excess risk, and vice versa. Furthermore, our theory suggests that naive continual learning without regularization suffers from catastrophic forgetting, while structural regularization mitigates this issue. Notably, structural regularization achieves comparable performance to joint training with access to both tasks simultaneously. These results highlight the critical role of curvature-aware regularization for continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies continual learning (CL) for two linear regression tasks under random design and covariate shift. It analyzes a generalized $\ell_2$-regularized CL algorithm (GRCL) that, during the second task, penalizes deviation from the first-task solution using a user-chosen positive semidefinite matrix $\Sigma$. The work derives matching upper and lower bounds on the joint excess risk and shows a quantitative trade-off between memory and statistics: richer $\Sigma$ (higher memory, measured by the number of stored vectors/eigen-components) yields lower excess risk. It proves that ordinary CL (OCL) and isotropic $\ell_2$ regularization can suffer catastrophic forgetting, while an appropriately chosen $\Sigma$ can match joint training performance. Extensions include lower/upper bounds for Gaussian design, a memory-risk curve with top-$k$ regularization, a discussion of NTK-regime neural networks, and small-scale experiments illustrating the theory.

### Strengths
The paper makes a clear and useful move: it treats continual learning through a memory-statistics lens and asks how much structure one must store to avoid forgetting. This framing is fresh and concrete. The theory then backs it up with matched upper and lower bounds. The main theorem decomposes risk into bias and variance terms that depend on the regularization matrix. This gives readers a direct way to see why naive OCL or an isotropic ridge can fail, and why a structured $\ell_2$ penalty helps. The "top-$k$" view of the matrix is especially appealing. It turns memory into a dial that one can set and shows that risk falls toward joint training as the rank grows. The examples are well chosen: they show catastrophic forgetting when the method ignores key directions, and they show how the proposed structure fixes that. The writing of the core results is also clear. Assumptions are stated, quantities are interpretable, and the logic builds in a steady way.

### Weaknesses
1. Strength of assumptions and scope: The shared optimal parameter and commuting covariances (diagonalizable in the same basis) simplify analysis but may be restrictive in practice.

2. From one-hot to Gaussian (and beyond): Main sharp results are for one-hot design; Gaussian design coverage is partial (OCL bounds, phenomena via Example 9), while GRCL bounds are deferred.

3.  Model selection for $k$ (or $\Sigma$): The theory shows performance as a function of $k$, but there is no principled selection rule.


4. Positioning vs. prior theory: The paper cites related work, but the novelty over Li et al. (2023) and Zhao et al. (2024) could be crisper, especially regarding random design + memory-explicit lower bounds.


5. Memory accounting and implementability: Memory is "number of vectors to store" or "rank of $\Sigma$, but concrete byte/parameter costs are not spelled out, especially for neural nets where curvature is layer-structured.

### Questions
1. Strength of assumptions and scope.

- Regarding the optimal parameter, can you add a proposition or appendix note quantifying the error when optimal parameters differ (the text hints at triangle-inequality arguments) and provide an explicit bound to show robustness to small misspecification?
- Can you relax "commuting $G, H$" via a perturbation result? For example, if $||G H-H G||$ is small, how do constants in the bounds inflate? Even a qualitative theorem would help.

2. From one-hot to Gaussian (and beyond).

- Can you provide at least an upper bound for GRCL in Gaussian design under mild spectral decay (e.g., polynomial/log-corrected tails)? Even if looser, it would connect the top-k message to the more common model. I think you can use Theorem 4.2 of Zhao et al. (2024, ICML) to do this.
- Can you include a short "what breaks" paragraph? For example, I was curious which steps in the one-hot proof fail for Gaussian, and what tools are perhaps needed?

3.  Model selection for $k$ (or $\Sigma$).

- Can the authors propose a simple criterion?
- The authors mention CountSketch can be used to specify the memory utilized in GRCL. Can the authors state a data-driven estimator for $\Sigma$ with guarantees? (Formalize an estimation step and propagate its error into the risk bound).

4. Positioning vs. prior theory.

- Can the authors add a comparative lemma or table? For example, which settings (fixed vs random; isotropic vs structured $\Sigma$; with/without matching lower bounds) each prior paper covers, and what this paper adds.

5. Memory accounting and implementability.

- Can the authors first define “memory” precisely, and then provide worked examples that show how to compute and compare storage costs for different choices of examples? For example, diagonal $\Sigma$, block-diagonal by layer, K-FAC with factor sizes, and a sketch-based $\Sigma$ showing CountSketch dimension versus error. I think this effort would close this gap and make the proposed trade-offs operational.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the trade-off between statistical performance and memory cost in Continual Learning (CL) with structural regularization methods. Focusing on a two-task linear regression model under a random design setting with covariate shift, the authors propose a Generalized ℓ₂-regularized Continual Learning (GRCL) algorithm and establish upper and lower bounds for its joint excess risk. The theoretical analysis reveals that Ordinary Continual Learning (OCL) without regularization and simple ℓ₂-regularized CL (ℓ₂-RCL) suffer from catastrophic forgetting in certain tasks, but a well-designed structural regularization (GRCL) with sufficient memory can mitigate forgetting and achieve statistical performance comparable to joint learning. The theoretical results are validated through numerical experiments and extended to neural networks in the Neural Tangent Kernel (NTK) regime.

### Strengths
First to provide rigorous theoretical formulation of the memory-performance trade-off in continual learning, with a unifying GRCL framework.

Quality:

Solid theoretical analysis with matching bounds and well-constructed examples. Clean extension to neural networks. Experiments properly validate theory.

Clarity: 

Well-structured and logically coherent. Core ideas are clearly communicated despite some heavy notation.

Significance: 

Establishes valuable theoretical foundation for continual learning. Provides practical guidance for algorithm design and opens directions for future work.

### Weaknesses
(1)	Limited Empirical Validation

The experimental validation, while supporting the theoretical findings, remains somewhat limited in scope and practical impact:
Neural network experiments are conducted only on small-scale MNIST variants (Permuted/Rotated MNIST). To better demonstrate real-world relevance, experiments on more challenging benchmarks (e.g., CIFAR-10, Split CIFAR-100) would strengthen the claims.
The comparison focuses primarily on methods within the structural regularization family. Including strong baselines from other CL paradigm would provide a more comprehensive assessment of where GRCL stands in the current landscape.

(2)	Strong Theoretical Assumptions

The analysis relies on several strong assumptions that may limit practical applicability. For example, the commutativity assumption of covariance matrices (Assumption 2), while mathematically convenient, rarely holds in real-world datasets. The authors could discuss how violations of this assumption might affect their conclusions, or provide empirical studies showing the theory's robustness to approximate commutativity.

### Questions
(1)	Could you discuss the practical implications of the commutativity assumption (Assumption 2)? How might your conclusions change when this assumption is violated in real-world datasets?

(2)	Would you consider adding experiments on more challenging benchmarks (e.g., CIFAR-100, ImageNet subsets) to better demonstrate the practical relevance of your theory?

(3)	Have you compared GRCL with strong baselines from other CL paradigms (e.g., experience replay, expansion methods)? Such comparisons would help position your method in the broader CL landscape.

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
The paper studies continual learning with two linear-regression tasks under random design. It analyzes three algorithms: OCL,l2-RCL and GRCL and establishes sharp upper/lower bounds on the joint excess risk of three lags. It shows that GRCL algorithm that uses a curvature-aware quadratic penalty (a PSD matrix) derived from the first task to mitigate catastrophic forgetting.

### Strengths
1. The paper considers an interesting problem in theoretical continual learning community.
2. The paper is well-written and well-structured.
3. The results driven by the data covariance are novel and interesting.

### Weaknesses
1. Several related work about theoretical CL are not discussed, for example:

[C1] Banayeeanzade et al., Theoretical Insights into Overparameterized Models in Multi-Task and Replay-Based Continual Learning, TMLR 2025.

[C2] Zheng et al., Towards Understanding Memory Buffer Based Continual Learning, 2025.

[C3] Ding, et al. Understanding forgetting in continual learning with linear regression. 2024.

[C4] Wen, et al. Information-theoretic generalization bounds of replay-based continual learning. 2025.

[C5] Wan et al. Understanding the Forgetting of (Replay-based) Continual Learning via Feature Learning: Angle Matters. 2025.

2. Two tasks in linear setting is simply and not enough to discuss the paper's findings. If it is only two tasks, what is difference between transfer learning vs continual learning? Based on equation 2, 3, it seems to be hard to extend the paper's results to $M$ tasks setting.

3. How to understand the “tail” of the distributions in equation 5? 

4. Can the author decompose their results to show the separate loss evaluations on each tasks? Since the main results are based on joint learning.

5. The results in section 4.2, why only give the example instead of general results. Since the paper investigates the impact of constrained memory while it still assumes that the data covariance ($\mu_i$) is related to the constrained memory setting ($k$). Can the authors explain more?

### Questions
Above

### Soundness
3

### Presentation
3

### Contribution
2
