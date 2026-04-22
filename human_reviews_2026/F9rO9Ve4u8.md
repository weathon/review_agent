# Time-Series Causal Discovery via Differentiable Permutations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Causal discovery with instantaneous effects in multivariate time series is challenging, as the instantaneous structure must be acyclic. Prior methods enforce this by either separating instantaneous and lagged estimation into multi-stage pipelines or imposing algebraic acyclicity constraints via complex augmented Lagrangian optimization, both of which incur high computational cost. In this work, we propose a different approach: we learn a differentiable permutation of variables using the Gumbel–Sinkhorn operator and triangularize the instantaneous coefficient matrix of a Structural Vector Autoregressive (SVAR) model in the learned order. This converts acyclicity from a hard constraint into a parameterization and keeps it valid throughout optimization. In doing so, our method enables unified, continuous optimization with gradient-based learning, leading to improved efficiency in time series causal discovery. Across three real-world benchmarks, our method achieves the best overall performance compared with 12 baselines in both discovery accuracy and efficiency. On the large-scale benchmark, it further demonstrates strong scalability, achieving more than a 6× speedup over competing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel, fully differentiable strategy for enforcing acyclicity of the instantaneous effect matrix in Structural Vector Autoregressive (SVAR) models. This approach offers a notable speedup and an elegant, single-stage optimization solution compared to existing methods that rely on multi-stage procedures.

While the theoretical contribution is interesting, the empirical evaluation requires improvement to substantiate the method's practical utility, especially as no theoretical guarantees are provided.

### Strengths
- The paper is clearly structured and easy to follow, making the core concepts accessible.
- The effort to evaluate the proposed method on multiple real-world datasets is commendable as it is often neglected in the community.

### Weaknesses
**Major**

**Lack of Distinction in Causal Effect Estimation**

The use of a single F1 score to evaluate the estimation of both lagged and instantaneous causal effects is a limitation. This makes it impossible to discern how well the proposed method uncovers the instantaneous causal structure ($B_0$) and whether potential inaccuracies in this estimation affect the estimation of lagged effects ($B_1$). To address this, a more detailed analysis, potentially including simple synthetic examples and comparisons of predictions on real-world data, would be highly beneficial.

**Questionable Performance on CausalRivers Dataset**

The bad performance of the standard VAR model on the CausalRivers dataset is puzzling, especially given that the causal relationships in this dataset are almost exclusively lagged due to the geographical distances between nodes. Theoretically, the proposed method should not be able to  uncover additional instantaneous links that a standard VAR model would automatically miss in this context. This might be an issue that is also persisting for the other datasets. Some comments on the causal structure beyond the summary graph are necessary, if the main contribution is an alternative way to estimate especially instantaneous effects. 

**Experimental Design for CausalRivers**

The decision to perform a single estimation on the entire CausalRivers graph (as far as I understand) presents a computationally intensive and likely intractable problem for most methods. A more feasible and informative approach would be to conduct experiments on subsets of the graphs, as suggested in the original CausalRivers paper. This would also mitigate the issue of a majority of the experimental runs timing out.


**Minor**

- Code Accessibility: The provided link to the code appears to be inaccessible.

- Citation Formatting: Some cited sources incorrectly list the first name as the family name.

- The intro motivation seems a little off. If two variables are actually cyclically causing each other, why would one enforce acyclicity? Here reality and theory seem to be mixed up. Some reformulations could improve the motivation.

- It is unusual for a method that relies on linear models, something that is extremely well explored in the literature, to refrain from stating any theoretical guarantees and general assumptions of the proposed approach.  While the lag of theoretical guarantees could be alleviated with an extensive empirical evaluation, including statements concerning assumptions like Causal sufficiency or Faithfulness  would benefit the paper

### Questions
*   **Choice of Maximum Lag:** Could you please clarify the rationale behind reporting the maximum lag of three for your experiments in the main part of the paper?
*   **Hyperparameter Details:** What were the specific values used for the temperature parameter in your experiments?
*   **Data Preprocessing:** Was any data preprocessing performed on the IT monitoring and CausalRivers datasets? If so, could you describe the steps taken? If not, could you please provide a justification for this decision?
*   **F1-Score Threshold:** What threshold was used to binarize the predictions for calculating the F1-scores?
* While not directly investigated for Time-series data, do you know how data standardization might affect your method? **
*  **Computation Cutoff:** The three hour cutoff seems arbitrary. Why did you choose this limit specifically?

**https://arxiv.org/pdf/2102.13647
**https://arxiv.org/abs/2104.05441

### Soundness
2

### Presentation
3

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
The paper proposes a time-series causal discovery method that enforces instantaneous acyclicity by learning a differentiable permutation with a Gumbel–Sinkhorn relaxation inside an SVAR model. This turns acyclicity from a hard constraint into a parameterization and enables a single‐stage, gradient-based optimization over both instantaneous and lagged effects. On three real-world benchmarks (IT Monitoring, SWaT, CausalRiver) the method reports strong F1 and runtime improvements relative to 12 baselines, with notable scalability at higher dimensionalities.

### Strengths
- Clear reformulation: acyclicity as a learnable permutation yields a clean, unified objective and removes augmented-Lagrangian loops. 
- Practical efficiency: consistent speedups on medium and large datasets while keeping accuracy competitive or superior. 
- Sensible evaluation choices: reporting F1 rather than ROC under class imbalance, and focusing on real-world datasets. 
- Well-scoped limitations section that surfaces pruning-threshold sensitivity and hardware considerations.

### Weaknesses
1.  Missing discussion of models with no contemporaneous edges at a sufficiently fine measurement scale. Many physical systems propagate over nonzero ε-time; at a fine enough Δt, contemporaneous edges vanish and only lagged links remain. There is a substantial body of work on undersampling and measurement-timescale mismatch showing that apparent instantaneous edges at coarse sampling can be artifacts, and that recovery should target the system-timescale graph: 
- Hyttinen et al. 2017 (A constraint optimization approach to causal discovery from subsampled time series data),
- Gong et al. 2015 (Discovering Temporal Causal Relations from Subsampled Data), 
- A Tank,  E B Fox,  A Shojaie (2019 Identifiability and estimation of structural vector autoregressive models for subsampled and mixed-frequency time series Free ), 
- M. Liu, X. Sun, L. Hu, and Y. Wang (2023 Causal discovery from subsampled time series with proxy variables), 
- Abavisani et al. (2023 Grace-c: Generalized rate agnostic causal estimation via constraints)

2. Claim about sparsity feels over-general. The manuscript leans on L1 and statements that “causal structures are typically sparse.” That is domain-dependent. Many real systems are moderately or highly connected. Sparsity is a useful regularizer, but not a universal property. Please qualify this assumption and discuss robustness when graphs are denser.

3. good breadth overall, but I’m missing comparisons to additional constraint-based and non-Gaussian structure-learning methods that are often strong in practice:
• FASK (Fast Adjacency Skewness) and related two-step procedures for dense feedback networks. These have shown good precision/recall in neuro and simulation studies. 
• Two-Step (Adaptive Lasso, “2-step Alasso”). Frequently competitive and relevant for SVAR-like linear settings.

4. Overwhelmingly positive results: the gains are large on SWaT and river datasets. Given the pruning-threshold sensitivity noted by the authors, I’d like to see a brief stability analysis over λ and pruning thresholds, or a bootstrap-based edge-stability plot, to rule out tuning luck.

### Questions
1 - The paper mentions standard SVAR identifiability via non-Gaussianity or equal error variances. Please make explicit which condition is assumed in each experiment and whether performance degrades under violations.

2 - Which experiments rely on non-Gaussian noise vs equal error variances? Any diagnostics to detect when assumptions fail?

3 - Provide edge-stability across random seeds, λ-grids, pruning thresholds, and temperature/iterations for Sinkhorn; include variance bands on F1.

### Soundness
3

### Presentation
4

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
This paper proposes a new temporal causal discovery method based on a structural vector autoregressive model. Its core contribution is to learn a differentiable permutation of variables using the Gumbel-Sinkhorn operator. This permutation is used to triangularize the instantaneous coefficient matrix, avoiding the need for hard acyclicity in previous continuous optimization-based methods. This enables a unified, gradient-based optimization, achieving better accuracy and speedups on real-world benchmarks.

### Strengths
1. The use of the Gumbel-Sinkhorn operator to learn an adaptive, soft permutation of the variables is interesting.
2. The method's effectiveness is validated on three real-world benchmarks. 
3. The writing is well-structured.

### Weaknesses
1. The paper's theoretical setup relies on established identifiability results for SVAR models. It explicitly states that a sufficient condition for identifiability is the presence of non-Gaussian noise. Later, the paper defines its objective function as the Mean Squared Error. It explicitly justifies this loss function by claiming it "is equivalent to maximizing the data likelihood". However, this claim of likelihood equivalence is only true if the noise is assumed to be Gaussian. This creates a fundamental inconsistency: the MSE objective is the maximum likelihood estimator for a model (Gaussian noise) that violates the identifiability condition (non-Gaussian noise). 
2. In the introduction, the authors argue that VAR-LiNGAM involves a discrete combinatorial search for the causal order and has an exponentially high computational cost. However, this claim is incorrect. VAR-LiNGAM is a well-established two-stage method. Stage 1: Use least-squares to obtain residuals (polynomial-time). Stage 2: Apply the LiNGAM algorithm on these residuals. Crucially, the core LiNGAM estimation, whether using ICA (as the paper itself correctly acknowledges in Section 2) or the DirectLiNGAM algorithm, is not a combinatorial search. It is a continuous optimization or a series of polynomial-time regressions.
3. Based on point 2, compared with VAR-LiNGAM, the core advantage of the proposed method is limited. It replaces a simple, two-stage, polynomial-time procedure (e.g., least-squares + ICA/ least-squares + regression + independence tests) with a single-stage, unified, but far more complex non-convex optimization problem. This raises the crucial question of whether this trade-off is justified, or if it just makes the problem more complex. Overall, I think the necessity and significance of this method are questionable, and the paper does not provide convincing evidence.
4. The paper includes no synthetic data experiments. While the justification—that such benchmarks can be "easy to game" —is understandable, this choice still creates a gap in the paper's validation. Controlled sensitivity analysis (e.g., sparsity, maximum lag) cannot be constructed on real-world benchmarks. Without these, it is impossible to reliably determine when and why the proposed method works well.

### Questions
1. I think the use of the Gumbel–Sinkhorn operator to obtain a differentiable permutation of variables does not necessarily need to be limited to time-series scenarios. Could this technique be directly extended to static (non-temporal) settings as well?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a novel method in time series causal discovery based on differentiable permutations. In particular, instantaneous effects are rephrased and masked via differentiable permutations, resulting in a parameter space that does not require constrained optimization to ensure acyclicity. The proposed method is then benchmarked using several real-world datasets and modern time series causal discovery algorithms.

### Strengths
- The paper is very well-written and easy to follow. While I have some reservations about related work with respect to the broader causal discovery literature (see below), the paper does a good job contextualizing causal discovery for time series, motivates the benefits of an unconstrained parameterization well.
- Experimental evidence suggests empirical benefits of the proposed method, with strong performance against many baselines on difficult benchmark tasks.

### Weaknesses
The authors acknowledge one existing work using differentiable permutation learning to help enforce a DAG constraint, but such applications are now widespread. In fact, the resulting methods often get exact acyclicity. For example, [1] explicitly uses Sinkhorn iterations and differentiable permutations to sidestep acyclicity. [2] introduces an algorithm leveraging interventional data. [1] is based on an equivalence to the successful NoCurl parameterization [3], suggesting a larger body of related work. Such approaches should be discussed in related works, and the novelty with respect to their formulation should be clarified.

The use of CPUs only for evaluation can also disadvantage some methods; just because some baselines are CPU-only does not imply that all baselines should be evaluated on CPU, especially when some architectures (e.g., cMLP/cRNN) explicitly benefit from GPU use and are designed accordingly.

As a minor note, some references (particularly in the experimental section) are not properly formatted.

[1] Annadani, Y., Pawlowski, N., Jennings, J., Bauer, S., Zhang, C., & Gong, W. (2023). BayesDAG: Gradient-based posterior inference for causal discovery. Advances in Neural Information Processing Systems, 36, 1738-1763.

[2] Chevalley, M., Mehrjou, A., & Schwab, P. (2024). Efficient Differentiable Discovery of Causal Order. arXiv preprint arXiv:2410.08787.

[3] Yu, Y., Gao, T., Yin, N., & Ji, Q. (2021, July). DAGs with no curl: An efficient DAG structure learning approach. In International Conference on Machine Learning (pp. 12156-12166). PMLR.

### Questions
1. How does the proposed approach compare theoretically to other methods using differentiable permutations to identify causal orders in an unconstrained optimization problem?

2. How does time performance change when applicable methods are run on GPU?

### Soundness
3

### Presentation
4

### Contribution
1
