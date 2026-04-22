# PLUMAGE: probablistic low-rank unbiased min variance gradient estimation framework for efficient large model training

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Accelerator memory and network constraints are dominant bottlenecks when training large language models (LLMs) with billions of parameters. Low-rank gradient estimators have been successfully applied by methods such as Galore and Flora for LLM training on consumer hardware, by compressing gradients and optimizer tensors. However, the underlying gradient estimation methods are biased or subject to a high variance. Moreover, low-rank optimizer states, such as the first and second moments under the previous subspace, become misaligned whenever the projection is updated. This misalignment can lead to instabilities during training with low-rank gradients.
We propose Plumage: Probabilistic Low‑rank Unbiased Minimum‑vAriance Gradient Estimator. 
Plumage can be applied as a drop-in replacement for low-rank LLM training without introducing new hyperparameters beyond the chosen rank $r$ and the update interval $\tau$. In addition, we resolve the misalignment of low-rank statistics.
We apply Plumage as a drop-in replacement for the fixed top-k components estimate used in Galore to observe how the gradient bias-variance tradeoff impacts the optimization of LLMs. 
The resulting Plumage+Adam shrinks the full-rank optimization's gap in the pre-training evaluation loss by 33\% on average across models, with a 34\% improvement on the commonsense benchmark for the 1B models. In finetuning tasks, the average training loss gap across the GLUE benchmark is shrunk by 28\% --- without retuning the full-rank learning rate and within a similar computational and memory footprint as Galore. 
Alternatively, in our 1B pretraining benchmark, Plumage+Adam surpassed the terminal loss of Galore within 30\% fewer steps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PLUMAGE, a probabilistic low-rank gradient estimator designed to enable efficient large language model (LLM) training by unifying unbiasedness, minimum variance, and adaptive subspace projection. 
Unlike prior low-rank optimizers such as GALORE (deterministic top-\(k\) truncation) or FLORA (random Gaussian projection),
 PLUMAGE samples singular components probabilistically with analytically optimal inclusion probabilities, ensuring that the expected reconstructed gradient equals the true gradient while minimizing estimator variance. 
The method further incorporates a one-sided projection to reduce memory footprint, a moment realignment scheme to stabilize stateful optimizers when switching subspaces, and an optional adaptive update interval \( \tau \) to balance computation and adaptivity. 
Empirical results across language modeling and fine-tuning benchmarks show that PLUMAGE achieves faster convergence and lower perplexity than prior low-rank and stochastic baselines.

### Strengths
1. The paper introduces a probabilistic formulation for low-rank gradient estimation that unifies unbiasedness and variance minimization under the MVUE framework. This formulation advances beyond prior deterministic (GALORE) and stochastic (FLORA) approaches by analytically deriving optimal sampling probabilities for singular components and incorporating stabilizing mechanisms such as moment realignment and adaptive projection intervals, effectively balancing bias, variance, and efficiency in subspace-based gradient compression.  

2. The paper provides extensive empirical validation across both pretraining and fine-tuning tasks. Its contributions are practically significant: PLUMAGE achieves comparable or superior performance while reducing memory footprint and computational overhead.

### Weaknesses
1. The methodological exposition lacks clarity and completeness. The probabilistic sampling process, moment realignment mechanism, and adaptive projection schedule are described conceptually but not operationally, leaving ambiguity in how these components interact in practice. A more detailed algorithmic breakdown or pseudocode would improve reproducibility and interpretability.  

2. The bias–variance–adaptivity analysis, though theoretically motivated, is not convincingly supported by empirical data. The paper would benefit from concrete diagnostics or visual evidence—such as gradient–subspace alignment plots, variance trajectories, or bias decomposition—to demonstrate that PLUMAGE achieves the claimed statistical balance during optimization.  

3. The empirical results are not sufficiently competitive with current state-of-the-art baselines. The reported improvements over GALORE are modest, and there is no direct comparison with FLORA, which is a closely related stochastic low-rank optimization method. Moreover, the evaluation focuses mainly on perplexity, limiting insight into generalization and robustness. Including comparisons with FLORA under equivalent conditions, as well as additional downstream benchmarks, would make the empirical validation more comprehensive and convincing.


4. The scope of efficiency is narrowly defined. PLUMAGE focuses primarily on computation and memory efficiency but does not consider communication efficiency or distributed scalability, which limits its practical applicability in large-scale multi-GPU or distributed training scenarios.

### Questions
Q1: PLUMAGE operates within a fundamental bias–variance–adaptivity trade-off:  Switching subspaces enables exploration of diverse gradient directions and reduces bias but increases variance, while fixing a single subspace lowers variance at the cost of stale, biased gradients that cannot capture evolving curvature. Although the paper claims to balance these effects through probabilistic linear projection and periodic realignment, it remains unclear how this mechanism explicitly mitigates bias without constraining directional diversity. Could the authors clarify whether any quantitative metric (e.g., gradient–subspace alignment, principal-angle drift, or bias–variance decomposition) or visualization (such as alignment trajectories or spectral evolution) is available to demonstrate that PLUMAGE indeed achieves an optimal trade-off between bias, variance, and adaptivity?

Q2: GALORE deterministically selects the top-\(k\) singular components via SVD, preserving gradient fidelity but introducing bias, whereas PLUMAGE stochastically samples \(k\) components using the “Wheel-of-Fortune” mechanism. However, this raises concerns about sampling instability—what if, by chance, low-energy (“bad luck”) components are selected, causing large deviation from the true gradient? The paper should clarify how such cases are prevented or bounded and provide justification for why this probabilistic selection can outperform deterministic top-\(k\) truncation. Is the improvement primarily due to reduced bias, and is this the main reason PLUMAGE requires fewer training steps compared with GALORE? A quantitative or theoretical comparison between probabilistic and monotonic top-\(k\) selection would substantially strengthen this claim.

Q3: PLUMAGE samples gradient components probabilistically, enforcing unbiasedness by matching the expected subspace to the full gradient and minimizing estimator variance. However, the FLORA paper also performs stochastic decomposition, treating LoRA as a gradient compressor through random Gaussian projections. What, then, is the fundamental distinction between PLUMAGE’s probabilistic sampling and FLORA’s stochastic approach, is it primarily the variance minimization objective? Furthermore, given that PLUMAGE only samples a fixed budget of components, is the full SVD decomposition strictly necessary? In principle, FLORA’s random-projection method might offer comparable computational efficiency with lower overhead. The paper would benefit from a direct quantitative comparison with FLORA under the same stochastic-variance setting to justify the need for SVD-based probabilistic sampling and its claimed advantages in variance reduction and convergence stability.


Q4: PLUMAGE derives an unbiased, minimum-variance gradient estimator by taking an expectation over the stochastic sampling of singular components. However, in practice, this expectation is never explicitly computed—the unbiasedness is instead guaranteed through the projection mechanism, where each sampled component is reweighted by its inclusion probability \(p_i\). Could the authors clarify how this projection-based design ensures the expectation property is maintained throughout training, especially when projections are reused over multiple steps? Moreover, does this expectation formulation translate into tangible variance reduction during optimization, or is the benefit primarily theoretical? Empirical evidence—such as variance trajectories or gradient–projection alignment metrics—would help validate that the projection mechanism indeed achieves the claimed minimum-variance behavior in practice.



Q5: PLUMAGE emphasizes using only a left-sided projection matrix for gradient estimation, unlike prior low-rank compression methods such as  PowerSGD[1] that employ both left and right projections. Could the authors clarify whether a symmetric right-sided counterpart exists—i.e., would applying a right-sided projection yield an equivalent estimator in expectation? Furthermore, what is the theoretical or empirical rationale for retaining only the left projection? Does this one-sided formulation fully preserve the statistical properties (e.g., unbiasedness and minimum variance) of the two-sided estimator, or might it introduce asymmetry or a loss in representational fidelity? Providing a short theoretical justification or ablation demonstrating the equivalence between left- and right-sided projections would strengthen the validity of this design choice and its practical implications.



Q6: The paper claims that PLUMAGE can employ an adaptive update interval \( \tau \) to dynamically control how often the SVD and projection subspace are refreshed, yet no quantitative or ablation comparison with a fixed \( \tau \) is presented. Could the authors clarify how this adaptive mechanism is implemented in practice—specifically, what metrics or thresholds determine when the projection is updated—and whether it leads to measurable efficiency or convergence benefits compared to a fixed interval? In the absence of empirical evidence, it remains unclear whether the adaptive \( \tau \) contributes to meaningful computational savings or improved training stability. A direct comparison between adaptive and fixed \( \tau \) settings would help validate this claim.


Q7: Although PLUMAGE is presented as an efficient training approach for large language models, the paper primarily reports validation perplexity as the key metric for evaluating optimization quality and performance. However, perplexity alone may not fully capture broader aspects such as generalization, calibration, or downstream task performance, even if this evaluation setting is consistent with GALORE. Could the authors justify why perplexity is considered a sufficient proxy for optimization progress in this work, and whether improvements in perplexity reliably translate to better model capability? Including complementary evaluations—such as downstream task accuracy, calibration error, or convergence efficiency—would help substantiate the claim that PLUMAGE achieves meaningful training efficiency beyond loss-level improvements.

Q8: While PLUMAGE is presented as an efficient optimization framework, the paper appears to focus primarily on reducing computational and memory costs (e.g., via low-rank projection, one-sided estimation, and adaptive SVD updates) rather than communication overhead in distributed training. Could the authors clarify whether PLUMAGE is intended solely as a computation-efficient method, or if it can also be extended to communication-efficient scenarios such as multi-GPU or distributed optimization? Additionally, have the authors considered evaluating the method’s impact on communication volume or synchronization cost to substantiate its scalability claims in distributed environments?

[1]Vogels, Thijs, Sai Praneeth Karimireddy, and Martin Jaggi. "PowerSGD: Practical low-rank gradient compression for distributed optimization." Advances in Neural Information Processing Systems 32 (2019).

This paper deserves a certain degree of discussion. Although the current version is not yet sufficient for acceptance, it presents several promising ideas that could develop into a strong contribution with further clarification and empirical support. I have raised multiple questions (Q1–Q8) regarding the methodology, empirical validation, and evaluation scope, and there remain additional issues beyond those listed. Nevertheless, I am willing to give the authors the benefit of the doubt at this stage, as the proposed framework shows potential and relevance for advancing efficient large-scale model optimization.

### Soundness
3

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
This paper introduces PLUMAGE, a novel framework for memory-efficient training of large models that addresses key limitations in existing low-rank gradient methods. The authors identify two primary issues: 1) the bias introduced by deterministic top-k gradient projections, and 2) the optimizer state misalignment. To solve this, PLUMAGE proposes a probabilistic, unbiased, and minimum-variance gradient estimator based on an efficient fixed-rank sampling strategy. Furthermore, it introduces a statistics realignment method to correctly transform the first and second moments of stateful optimizers into the new subspace. Experiments show that PLUMAGE closes the performance gap to full-rank ADAM compared with other low-rank-based methods.

### Strengths
1. The paper theoretically proposes an unbiased minimum-variance low-rank gradient estimator (PLUMAGE) and applies it to the efficient training of large-scale language models. This approach is theoretically inspiring and provides a new perspective on solving the bias problem in low-rank gradient estimation.
2. The paper introduces several practical and insightful contributions. The "wheel-of-fortune" sampling algorithm is an interesting and efficient trick for the k-sparse sampling problem. More importantly, the paper correctly identifies that optimizer state realignment is a critical and often overlooked issue in low-rank optimization methods that use periodic projection updates. The proposed realignment strategy (Eq. 18 and 19) provides a well-reasoned potential solution to this problem, which is highly inspiring for future work in this area.

### Weaknesses
1. Minor Typos in Appendix: The analysis in Appendix A, while ultimately arriving at a standard and correct result (Eq. 22), contains several confusing typos. Specifically, Equation (21) misrepresents the matrix inner product within the Trace. Maybe it should be $v_i u_i^\top u_j v_j^\top$ in the left one and $v_i v_i^\top$ in the right one. 
2. The paper's primary contribution is its formulation as an unbiased minimum-variance estimator, distinguishing it from biased top-k methods like GALORE. However, the practical significance of this distinction is worth discussing given the known properties of LLM gradients. The premise of methods like GALORE is that LLM gradient spectra are "top-heavy," with rapid decay, meaning the top-k singular vectors capture the vast majority of the gradient's energy and information. This empirical phenomenon is well-documented [https://arxiv.org/abs/1611.07476], [https://arxiv.org/abs/2403.03507]. Under this top-heavy assumption, the PLUMAGE sampling algorithm will likely behave in a predictable way: the deterministic rank will automatically capture these dominant directions by assigning them $p_i=1$ (because for these directions their dominant singular values satisfy $(k-r)\sigma_r \ge \sum_{i=r+1}^n \sigma_i$. Consequently, the sampling process will be truncated: the top $r^{\*}$ components are selected with high probability (identical to top-k), and the remaining $k-r^\*$ rank budget is used to probabilistically sample from the vast "tail" of non-dominant directions. Because the singular values in this tail are small and relatively uniform, the sampling probabilities $p_i$ for these tail components will be small and near-uniform ($p_i \approx (k-r^\*) / (n-r^\*)$).
3. In practice, PLUMAGE's estimator may be functionally equivalent to a standard top-k projection, augmented with adaptive, near-uniform sampling in the non-dominant subspace. While this achieves statistical unbiasedness, the paper does not sufficiently argue why this sampling of the low-energy tail is so impactful, versus simply being a minor algorithmic difference from top-k projection. Therefore, it may be necessary to provide some metrics during the training process and conduct corresponding ablation studies. For example, what is the specific distribution of $p_i$ for different parameters (like attention or MLP parameters) in the early and late stages of training, and does it indeed align with the characteristics of top-k truncation in practice? Alternatively, what are the spectral properties of the full-space gradient in the early and late stages of training? This could affect the method's difference from methods like GALORE, as this difference might emerge during the training process. For instance, does PLUMAGE's advantage stem from exploring more directions in the later stages of training? I believe these points need to be clarified in the paper.
4. The experimental results presented do not show that the PLUMAGE method has a significant advantage. The training curves in Figure 7 suggest that the advantage over GALORE is not significant, and this effect diminishes as the model size increases. Therefore, I am curious about the true effectiveness of the method in larger-scale practical pre-training tasks. Furthermore, PLUMAGE still relies on SVD in Algorithm 5, which is not eliminated. Therefore, when considering the model sharding and parallel strategies required for large-scale experiments, an additional all-gather operation might be necessary to perform SVD on the full gradient, potentially introducing extra communication overhead.

### Questions
See Weaknesses.

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
The paper proposes PLUMAGE, a probabilistic low-rank gradient estimator that aims to overcome the bias–variance limitations in existing low-rank optimizers such as GaLore and FLORA. PLUMAGE introduces (1) a k-sparse probabilistic estimator with unbiased minimum-variance sampling and (2) a moment realignment strategy to stabilize training under dynamic projection subspaces. It is designed as a drop-in replacement for existing low-rank optimizers without extra hyperparameters. Experiments across pretraining (LLaMA 130M–1B) and fine-tuning (GLUE and common-sense tasks) show consistent improvements over GaLore.

### Strengths
- The experiments design is comprehensive, convering pre-training, fine-tuning (GLUE and common-sense) and multiple ablation studies, providing a complete understanding of the proposed method.

- The framework can be integrated easily with existing optimizers and training pipelines.

- The explanation of the method is clear and easy to understand.

### Weaknesses
- The results in Table 4 is strange , as GaLore’s performance is significantly lower than that of the baseline Adam. Based on my experience and community reports, GaLore’s results are reproducible. It would be helpful to analyze this discrepancy and provide detailed hyperparameter settings to clarify the mismatch.

- It would be beneficial to demonstrate end-to-end training efficiency improvements to more convincingly support the efficiency claim.

- Since the proposed method is orthogonal to memory-efficient optimizer designs, it would be interesting to explore their compatibility, i.e., whether there won't be further performance loss. Such results could demonstrate the method’s generality and strengthen its overall impact.

- There are several typos throughout the paper (e.g., lines 102 and 265); careful proofreading is recommended.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
2

### Contribution
3
