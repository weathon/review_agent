# Correlating Cross-Iteration Noise for DP-SGD using Model Curvature

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Differentially private stochastic gradient descent (DP-SGD) offers the promise of training deep learning models while mitigating many privacy risks. However, there is currently a large accuracy gap between DP-SGD and 
 normal SGD training. This has resulted in different lines of research investigating orthogonal ways of improving privacy-preserving training.
One such line of work, known as DP-MF,  correlates the privacy noise across different iterations of stochastic gradient descent -- allowing later iterations to cancel out some of the noise added to earlier iterations. In this paper, we study how to improve this noise correlation. We propose a technique called NoiseCurve that uses model curvature, estimated from public unlabeled data, to improve the quality of this cross-iteration noise correlation. Our experiments on various datasets, models, and privacy parameters show that the noise correlations computed by NoiseCurve offer consistent and significant improvements in accuracy over the correlation scheme used by DP-MF.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes using public data to estimate the Hessian's eigenvalues for improving DP matrix factorization.

### Strengths
The experimental results demonstrate that the proposed method can improve utility over DP-SGD and DP-BANDMF.

### Weaknesses
1. The core solutions (S1-S4) are presented as empirical observations without theoretical justification. It is unclear why these specific choices work or how they relate to the underlying theory of DP optimization.

2. The proposed algorithm is not formalized with clear, step-by-step pseudocode. This makes it difficult to understand the exact procedure.

### Questions
1. The concept of a "model Hessian" is central but not well-explained. If it's data independent, why and how to compute it? If it's data dependent, how does it generalize to the private data distribution using public data?

2. Why does using random labels for the Hessian calculation work? Intuitively, this should produce meaningless curvature information.

3. For the quadratic loss example (line 202), the variable d is not clear. Is this the label? How to understand it when adding H in the middle?

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
4

### Summary
This paper proposes NoiseCurve, a method to improve the DP-SGD by incorporating model curvature information into cross-iteration noise correlation. The authors propose a new objective on top of the DP-MF framework that interacts between model curvature and DP-SGD with noise. The experiments show promising improvements over DP-SGD and DP-BANDMF across multiple vision datasets.

### Strengths
1. The observation on how noise propagates through gradient computation points in DP-MF methods (Eq.2 in Section 2) is an important aspect of correlated noise design. The distinction between direct noise effects and Hessian-mediated effects is well-motivated.
2. Theorem-1 shows that under quadratic loss, only Hessian eigenvalues matter for the optimal objective. This reduction from $O(p^2)$ to $O(p)$ parameters theoretically shows that the geometric structure of the loss landscape relevant to correlated noise is fully captured by the eigenspace. It offers to estimate the curvature by computing eigenvalues alone rather than full Hessian matrices.
3. The paper addresses four challenges (S1-S4) in applying curvature-based noise correlation: using public data for eigenvalue estimation, handling changing/negative eigenvalues, and scaling to large models. Each solution is empirically supported.
4. The experiments span multiple datasets (CIFAR-10, ChestX-ray14), architectures (CNN, ResNet, VGG, ViT), training strategies (full training, fine-tuning, LoRA), and privacy budgets ($\varepsilon$), showing consistent trends in performance.

### Weaknesses
1. The paper provides no convergence analysis for the non-convex case or approximation guarantees when using estimated eigenvalues. Additionally, there is no analysis to place bounds on the error introduced by setting negative eigenvalues to zero.
2. Section 6 acknowledges "insufficient understanding'' of when public data choice matters. The claim that eigenspectra "look similar'' across datasets (Figure 2) is based only on small CNNs and limited to vision domains, with no experiments on NLP, tabular or structured data to validate generalization.
3. The eigenvalue approximation strategy is undervalidated despite being critical for scalability to large models. The power-law curve-fitting approach (Section 4.4), while orthogonal to the main contribution, lacks principled guidance for selecting hyperparameters $p_+$ and $\mu_{p_+}$, and is only qualitatively validated on small-scale ($\sim$30,000 param) models (Figure 4).
4. The reported improvements seem moderate (1-2\% in many cases) and often fall within/near the standard deviation ranges, such as in Table 1a, where NoiseCurve achieves $76.5 \pm 0.2$ compared to DP-BANDMF's $75.94 \pm 0.04$. Without statistical significance testing, it is unclear whether these gains are meaningful, and some experimental settings show minimal improvements over the DP-BANDMF baseline.
5. The paper does not report any results on computational overhead, including training time, memory usage, and the costs associated with eigenvalue computation. Therefore, it is difficult to assess the practical feasibility and scalability of the proposed method.

### Questions
1. Could the authors provide any theoretical analysis regarding the approximation quality when extending from the quadratic case to non-convex settings? Additionally, it would be helpful to understand how performance degrades as public data becomes less private.
2. What is the computational overhead of NoiseCurve compared to baseline methods like DP-BANDMF? This would help better assess the practical feasibility and scalability of the NoiseCurve approach.
3. Were any statistical significance tests conducted to confirm that the performance NoiseCurve is meaningful? Additionally, it would be helpful to understand under which conditions NoiseCurve provides the most substantial benefits over DP-BANDMF.
4. Is there any way to trace the Hessian powers more efficiently using techniques like Hutchinson's estimator or stochastic trace estimation? This might avoid the computational bottleneck of eigenvalue decomposition while still capturing the essential curvature information.
5. Could the authors comment on whether the similar performance holds for other modalities such as NLP or tabular data? Are there any preliminary experiments or theoretical reasoning to support the generalization of this approach beyond computer vision domains?

### Soundness
2

### Presentation
3

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
To resolve the problem of noise addition in Differentially private stochastic gradient descent (DP-SGD) scenarios, NoiseCurve is proposed in this work, that uses model curvature, estimated frompublic unlabeled data, to improve the quality of this cross-iteration noise correlation. The experiments on various datasets, models, and privacy parameters show that the noise correlations computed by NoiseCurve offer consistent and significant improvements in accuracy over the correlation scheme used by DP-MF.

### Strengths
- formulates the ojjective for the interaction between model curvature and differentially private SGD with correlated noise.
- obtaining curvature information from unlabeled public data, and coping with the fact that neural network Hessians change during training and have negative eigenvalues.
- the scalability is verified and a method integrating eigenvalue estimation with implicit estimation of the size of the subspace the gradients is presented.

### Weaknesses
- Heavily heuristic core objective. The main objective is derived for a quadratic loss with a fixed, known Hessian and then applied to deep, nonconvex networks via proxies; there’s no bound quantifying the gap between the quadratic surrogate and real training dynamics under clipping + DP noise.
- Eigenvalue approximation by curve-fitting lacks guarantees.
- Small-sample reporting. Many figures/tables average over only 3 runs; several plots lack error bars or confidence intervals, making it hard to assess robustness and variance under DP randomness.

### Questions
- Multiple runs on the experiments are necessary, such as Table 1.
- More ablation study is necessary, such as, band size, learning rate, clipping norm, noise multiplier, etc.

### Soundness
2

### Presentation
2

### Contribution
2
