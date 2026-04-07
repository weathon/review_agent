## Summary
This paper introduces Multi-Grade Deep Learning (MGDL), a training framework that decomposes deep network optimization into sequential stages, each training a shallow network on residuals from previous grades. It provides theoretical convergence guarantees for gradient descent, shows that for single-layer ReLU grades the problem reduces to convex subproblems, and offers extensive empirical demonstrations of improved stability and performance across image reconstruction, CIFAR classification, and transformer-based time series regression.

## Strengths
- **Theoretical contributions**: Theorems 1 and 2 establish convergence of gradient descent for both SGDL and MGDL under smooth activations, and Theorem 3 proves that for single-layer ReLU grades, MGDL decomposes into a sequence of convex programs, extending convexification results to deep architectures.
- **Comprehensive empirical validation**: MGDL consistently outperforms SGDL in image regression, denoising, deblurring (PSNR gains of 0.42–4.23 dB), and CIFAR-100 classification (lower training loss), with evidence from fully connected networks, CNNs, and transformers (Tables 1-3, Figures 10-19).
- **Insightful mechanistic analysis**: Eigenvalue analysis of the GD iteration matrix shows that MGDL keeps eigenvalues within (-1,1), leading to stable convergence, while SGDL eigenvalues often exit this range, causing oscillations (Section 7, Figures 4-6).
- **Demonstrated robustness**: MGDL is empirically more robust to learning rate choices in synthetic and image regression tasks, maintaining performance over a wider range than SGDL (Section 6, Figure 20).
- **Novel extension to transformers**: Application to Multi-Grade Transformers (MGT) shows improved generalization on synthetic and financial time series regression, with test error reductions of 84% and 80%, respectively (Section 8, Tables 4-5).

## Weaknesses
- **Theory-experiment mismatch** — Convergence and eigenvalue analyses (Theorems 1, 2, 4) assume twice or thrice continuously differentiable activations, but all experiments use ReLU, which is non-smooth. This undermines the theoretical relevance to the empirical results and leaves the guarantees inapplicable to the presented settings.
- **Limited scope of convexity result** — Theorem 3 only applies to single-layer, bias-free ReLU grades, yet experiments use multi-layer grades with biases (e.g., in image tasks and transformers). The practical relevance of this theoretical insight to deep MGDL is unclear and unsubstantiated.
- **Unclear architectural parity** — The paper does not ensure that SGDL and MGDL models have comparable total parameters or depth (e.g., in image regression, SGDL: (2,1,128,8) vs. MGDL: (2,1,128,2,4)), raising concerns that improvements may stem from capacity differences rather than the training strategy.
- **Non-standard loss for classification** — CIFAR-100 experiments use mean squared error (MSE) instead of the standard cross-entropy loss without justification, which may disadvantage SGDL unfairly and limits the interpretability of classification performance.

## Nice-to-Haves
- Ablation studies on the number of grades and depth per grade to understand sensitivity and design choices.
- Extended learning rate robustness analysis to classification tasks (beyond regression).
- Computational efficiency comparison including wall-clock time and FLOPs to substantiate scalability claims.
- Comparison to related progressive training methods (e.g., greedy layer-wise training) to clarify novelty, though the paper focuses on MGDL vs. SGDL.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Statistical significance tests or multiple runs**: While valid, single-run evaluations are common in the field for the tasks presented, and the consistency across experiments mitigates this concern.
- **Demand for large-scale benchmarks (e.g., ImageNet)**: The paper claims broad empirical improvements on established tasks; large-scale benchmarks are a future direction rather than a core flaw.
- **Insufficient hyperparameter tuning details**: The use of Adam with described architectures is typical, and the learning rate study partially addresses robustness.
- **Requests for visualizations like loss landscapes or residual targets**: These would enhance the paper but are not required for the core claims.

## Novel Insights
The paper provides a novel convexification of deep ReLU networks through multi-grade decomposition for single-layer grades, extending prior work on shallow networks to a sequential setting. The eigenvalue analysis offers a mechanistic explanation for MGDL's stability by linking spectral properties of the GD iteration matrix to optimization dynamics, showing that MGDL confines eigenvalues within (-1,1) while SGDL does not.

## Suggestions
- Address the theory-experiment mismatch by either adapting the theory for non-smooth activations (e.g., via subgradients) or using smooth approximations in experiments to align with assumptions.
- Ensure fair comparisons by matching model capacities (e.g., reporting parameter counts and depths) or justifying architectural choices to isolate the effect of the training strategy.
- Report test accuracy for CIFAR classification tasks and consider using cross-entropy loss for standard benchmarking, or justify the use of MSE.
- Clarify in the discussion how the convexity result for single-layer grades relates to practical multi-grade networks, acknowledging limitations and potential extensions.