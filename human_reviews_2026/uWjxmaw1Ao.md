# Statistical Optimality of Newton-type Federated Learning with Heterogeneous Data

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Most federated learning algorithms, such as FedAvg and FedProx, only communicate first-order information, which can be inefficient under heterogeneous data and leaves their statistical behavior poorly understood.  We propose FedNewton, a second-order federated learning method that shares both gradient and curvature information while retaining a lightweight communication pattern.  In a kernel ridge regression setting, we derive non-asymptotic excess-risk bounds for FedNewton and establish minimax-optimal learning rates, explicitly quantifying the roles of local sample size, data heterogeneity, and model heterogeneity.  Our theory further shows that, under benign conditions, the federated error of FedNewton decays exponentially in the number of communication rounds.  Beyond this RKHS regime, we instantiate FedNewton in a practical _backbone+head_ federated fine-tuning setting and conduct large-scale experiments on standard vision benchmarks, demonstrating that FedNewton achieves strong accuracy and efficiency compared with state-of-the-art first-order and second-order baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose FedNewton, which communicates the global gradient and local inverse‑Hessian preconditioned increments. The theory gives minimax‑optimal generalization rates while quantifying effects of local sample size, covariate shift, and response shift. A key message appears to be that if local data are not too small and heterogeneity is modest, the federated error decays exponentially in $t$.

### Strengths
1. Prior second‑order FL work focused on optimization, but here we get excess‑risk bounds with minimax‑optimal rates.

2. The claim that the proposed algorithm has an exponential decay in the federated error in terms of iterations is an astonishing and shocking result, if true.

### Weaknesses
1. The paper is quite involved, and would benefit from more intuitive scaffolding. 

2. Can you clarify the statement that this second-order method drawing on local hessians is only "2 times" as compared to first-order FL algorithms?

### Questions
Please see weaknesses. Also, why is $C-C_{\mathcal{D}}$ PSD in equation (11)?

Note: I am not particularly knowledgeable about the area in which this paper is situated in. This is reflected in my confidence score.

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
This paper proposes FedNewton, a second-order federated learning algorithm that shares both first-order (gradients) and second-order (Hessian) information across local devices. The authors analyze this method in the kernel ridge regression (KRR) setting and derive generalization bounds that quantify the impact of local sample size, data heterogeneity (covariate shift), and model heterogeneity (concept shift). The main theoretical contribution is establishing minimax-optimal learning rates for federated Newton methods and showing that under sufficient local samples and moderate heterogeneity, the federated error decreases exponentially.

### Strengths
- This appears to be the first work providing rigorous generalization guarantees (not just optimization convergence) for Newton-type federated learning under both data and model heterogeneity. The gap between optimization and generalization analysis in federated learning is significant, and this work makes progress on bridging it.
- The paper provides a unified treatment of both covariate shift and concept shift, with explicit quantification of their impacts on learning rates. The error decomposition in Theorems 1-2 cleanly separates these effects.
- The authors extend beyond the standard $r \in [1/2, 1]$ regularity condition to $r > 0, 2r + \gamma ≥ 1$, which is more general than prior DKRR work (Zhang et al., 2015; Guo et al., 2017).
- The paper provides explicit communication costs ($\mathcal{O}(M)$) and shows that FedNewton achieves linear convergence when conditions are met, requiring fewer rounds than first-order methods' complexity.

### Weaknesses
- The entire theoretical analysis is limited to squared loss and kernel ridge regression. While Remark 4 claims the algorithm applies to "twice differentiable" loss functions, no theoretical guarantees are provided for other losses. This severely limits practical applicability, especially for classification tasks, which dominate federated learning applications.
- Computing $H^{-1}_{D_j, \lambda}$ requires $\mathcal{O}(|D_j|M^2 + M^3)$ operations. While Remark 1 mentions existing techniques (BFGS, L-BFGS), the paper dismisses this as "beyond scope". For practical federated learning with large $M$, this is a critical limitation that undermines the "efficiency" claims. The paper should either provide concrete solutions or temper its efficiency claims.
- Assumption 1 (capacity condition) requires $\max(\mathcal{N}(\lambda), \mathcal{N}_1(\lambda),...,\mathcal{N}_m(\lambda)) ≤ Q^2 \lambda^{-\gamma}$, constraining all local effective dimensions. Why is this reasonable when local distributions are heterogeneous?
- Experiments use random Fourier features (finite M=200 or 2000), but theory assumes infinite-dimensional RKHS. Remark 6 briefly mentions finite-dimensional cases but doesn't provide the main results.
- The initialization $w^0_{D_j, \lambda} = H^{-1}_{D_j,λ} \Phi^T_{D_j} y_{D_j}$ is non-standard and already requires expensive local computation before any communication.
- Table 1 shows many recent Newton-type FL methods (FedNL, SHED, FedNS, Fed-sofia) but the paper only compares experimentally with FedAvg and FedProx. Direct empirical comparison with these second-order baselines is essential to validate the claimed advantages.
- How sensitive is the method to hyperparameter choices ($\sigma^2, \lambda$)? The experiments mention grid search but don't discuss robustness.
- Proposition 1 claims "partitionability" but this only holds for squared loss. This limitation should be emphasized.
- No wall-clock time comparisons, only iteration counts. 
- Missing experiments on larger-scale datasets common in federated learning (e.g., CIFAR-10, FEMNIST).

### Questions
Refer to the weaknesses section.

### Soundness
3

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
3

### Summary
This paper introduces FedNewton, a second-order federated learning algorithm that communicates local curvature information to achieve faster and statistically optimal convergence under heterogeneous data. The method bridges optimization and generalization by decomposing total error into centralized and federated components and establishes optimal convergence rates. Experiments on synthetic and LIBSVM datasets support the theoretical claims, showing improved accuracy and convergence over FedAvg and FedProx. Overall, the work makes a strong theoretical contribution, though experiments remain limited in scale.

### Strengths
1. Provides a rigorous theoretical framework connecting optimization and generalization in second-order federated learning.
2. The FedNewton algorithm (Algorithm 1, Figure 1) is well-designed, balancing curvature-based updates with communication efficiency.
3. The error decomposition (Section 4) clearly explains the effects of heterogeneity on performance.
4. The theoretical analysis is detailed, logically structured, and establishes minimax-optimal convergence rates.
5. Experimental results (Figure 3, Table 1) align with the theory, showing consistent improvement over baseline methods.
6. The comparative analysis (Table 1) clearly positions the work relative to other Newton-type and first-order approaches.

### Weaknesses
1. Experiments are limited to small synthetic and LIBSVM datasets; larger or more diverse benchmarks like ( FEMNIST, CIFAR-FL) would strengthen the results.
2. Computing local Hessian inverses remains expensive; the paper mentions approximations but provides no empirical evaluation.
3. The heavy notation makes the theory difficult for non-specialists; brief intuitive explanations or visual aids would improve readability.
4. Experimental plots (Figure 3) lack variance bars or confidence intervals, making it hard to assess robustness.
5. The heterogeneity settings used are synthetic; connecting them to real-world non-IID data distributions would enhance practical relevance.

### Questions
1. Can FedNewton use approximate or low-rank Hessian inverses without losing its reported performance?
2. Does the observation that the method converges within two rounds hold across different datasets?
3. Have you tested the approach with non-squared loss functions such as logistic or cross-entropy loss?
4. How well do the theoretical results transfer to neural-network-based or kernelized models in practice?
5. Include runtime and memory comparisons with FedAvg and FedProx for  example ( number of communication rounds vs. total time).
6. Add simple convergence plots showing performance across different data heterogeneity levels (as in Figure 3).
7. Discuss how FedNewton could be extended to non-convex or deep learning settings (Section 6).

### Soundness
2

### Presentation
1

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
This paper investigates the statistical optimality of Newton-type federated learning (FL) algorithms under heterogeneous data distributions. The authors propose FedNewton, a second-order federated optimization method that leverages both global gradients and local Hessians to improve convergence and generalization. The authors further quantify how local sample size, data heterogeneity, and model heterogeneity jointly affect the excess risk and convergence behavior. Experimental results on synthetic and real-world datasets validate the theoretical findings, showing that FedNewton achieves exponential convergence with minimal communication rounds.

### Strengths
1. The paper presents the generalization analysis for Newton-type federated learning methods under data heterogeneity. 
2. The authors derive non-asymptotic excess risk bounds and demonstrate minimax-optimal learning rates under mild assumptions. The decomposition of federated error and centralized excess risk provides clear interpretability.

### Weaknesses
--Limited diversity and scalability of experimental settings.
The experimental evaluation in Appendix A mainly uses a synthetic dataset and small-scale benchmarks from LIBSVM. These datasets are low-dimensional and domain-specific, which restricts the demonstration of FedNewton’s capability in large-scale or high-dimensional federated learning scenarios, such as image or language applications. Furthermore, all experiments focus on convex regression problems, without extending to non-convex architectures like neural networks. 

--Restrictive theoretical assumptions.
The theoretical analysis relies on kernel ridge regression with squared loss and assumes that all local functions lie in a reproducing kernel Hilbert space (RKHS). While this setting facilitates mathematical tractability, it is less reflective of practical federated learning where non-convex objectives, neural network architectures, or unbounded losses are common. 

--Incomplete communication–computation tradeoff analysis.
Section 3 provides a complexity discussion but does not present any runtime or communication cost comparisons in the experiments. While the authors claim that FedNewton achieves similar per-round cost as first-order methods with exponentially faster convergence, this claim is not quantitatively supported. For example, the cost of computing and inverting local Hessians can be substantial for large feature dimensions. Without practical wall-clock evaluations or scalability analyses, it is unclear whether the method is truly more efficient in real federated environments.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
