# Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Differential Privacy (DP) is becoming central to large-scale training as privacy regulations tighten. We revisit how DP noise interacts with _adaptivity_ in optimization through the lens of _stochastic differential equations_, providing the first SDE-based analysis of private optimizers. Focusing on *DP-SGD* and *DP-SignSGD* under per-example clipping, we show a sharp contrast under fixed hyperparameters: *DP-SGD* converges at a Privacy-Utility Trade-Off of $\mathcal{O}(1/\varepsilon^2)$ with speed independent of $\varepsilon$, while *DP-SignSGD* converges at a speed *linear* in $\varepsilon$ with a $\mathcal{O}(1/\varepsilon)$ trade-off, dominating in high-privacy or large batch noise regimes. By contrast, under optimal learning rates, both methods achieve comparable theoretical asymptotic performance; however, the optimal learning rate of *DP-SGD* scales linearly with $\varepsilon$, while that of *DP-SignSGD* is essentially $\varepsilon$-independent. This makes adaptive methods far more practical, as their hyperparameters transfer across privacy levels with little or no re-tuning. Empirical results confirm our theory across training and test metrics, and empirically extend from *DP-SignSGD* to *DP-Adam*.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The main methodological contribution is to approximate the discrete optimization algorithm with a stochastic differential equation (SDE). It focuses on DP-SGD and DP-SignSGD under per-example clipping and provides the first theoretical framework that connects SDE with convergence dynamics.

### Strengths
The paper presents the first SDE-based theoretical framework for analyzing differentially private optimizers, bridging discrete training dynamics with continuous-time analysis. It provides clear scaling laws for DP-SGD $O(1/\epsilon^2)$ and DP-SignSGD $O(1/\epsilon)$, offering strong insight into how adaptivity interacts with privacy noise. Theoretical predictions are thoroughly validated by experiments across multiple datasets. The work also gives practical guidance for tuning and transferring DP optimizers under different privacy budgets.

### Weaknesses
While the SDE framework is well motivated, the paper does not clearly justify why the continuous approximation is valid or which small terms are neglected. A short discussion explaining that the discretization error between the algorithm and its SDE counterpart can be ignored would make this assumption more convincing. Additionally, although the authors argue that higher-order SDE approximations are unnecessary, a simple analysis or experiment illustrating how the first-order approximation deviates from the discrete dynamics would strengthen the theoretical rigor. Clarifying these points would improve both the transparency and completeness of the analysis.

### Questions
1.The paper could provide more heuristic insight into why DP-SignSGD, which can be seen as a post-processing of DP-SGD and thus should not improve efficiency, nevertheless achieves better empirical and theoretical utility. In particular, what property of the sign operation makes it more robust to DP noise?

2.Additionally, the role of the clipping bound C deserves clarification: while a larger C should intuitively reduce gradient bias, the theory suggests that a larger C moves the iterates further from the optimum.

3.The two-phase assumption (clipped vs. unclipped regime) is central to the analysis but not well justified; a short discussion of why it is valid to analyze the optimization dynamics separately in the clipped and unclipped phases would improve the theoretical clarity of the paper -- it should noted that in a single batch of samples, some gradient would be clipped and some would not but we will only have one update at the end. The analysis in the main theorem is confusing.

4.The paper also lacks comparison with SOTA DP-SGD benchmark, e.g., "unlocking high-accuracy differentially private image classification through scale", "a theory to instruct differentially-private learning via clipping bias reduction", and "differentially private image classification
by learning priors from random processes".

### Soundness
2

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
The paper studies DP stochastic gradient descent and its variants via modelling the DP-SGD dynamics via stochastic differential equations. It develops continuous-time approximations for adaptive algorithms and distinguishes between two training “phases”: an initial exploration phase and a later convergence phase. The analysis aims to explain how noise injection from differential privacy interacts with the optimization dynamics, and how the privacy parameters and optimizers' hyperparameters (clipping bound, batch size and learning rate) influence convergence. The authors attempt to unify previous heuristics (DP-SGD, DP-SignSGD etc.) under a common SDE framework, claiming to provide an interpretable connection between privacy, optimization noise, and generalization. The paper also reports empirical results to illustrate the phase-transition behavior. There is a very interesting phenomenon that is supported both by theory and experiments: DP-SGD has $\varepsilon^{-2}$ behaviour for small $\varepsilon$-values whereas DP-Adam and DP-SignSGD have $\varepsilon^{-1}$ behaviour.

### Strengths
- The paper is well written, seems to be of very high quality.

- The fact that the SDE view is able to capture the experimental behaviour that DP-SGD has $\varepsilon^{-2}$ behaviour for small $\varepsilon$-values whereas DP-Adam and DP-SignSGD have $\varepsilon^{-1}$ behaviour (see Figure 1) is very impressive. 

- The SDE view is well motivated and also commonly considered in the literature (e.g., Blei et al. 2018).

### Weaknesses
- The paper focuses on only on few adaptive optimizers, and I am a bit surprised about their choices: DP-Adam (Adam with DP gradients) and DP-SignSGD (which is not that well-known). The reason might be that the analysis is amenable for them (questions below), and I think the contribution is very valuable neertheless.

- Due to the fact that very few adaptive optimizers seem to actually fit into this SDE framework (or can be seen as discretizations of SDEs, meaning that the weakly converge to them in the vanishing step size limit), I have a feeling that this framework does not actually help in designing new hyperparameter adaptive DP optimizers. The paper seems thus to give an analytical explanation of certain differences in the optimizers' behaviors ( $\varepsilon^{-1}$ vs.  $\varepsilon^{-2}$ error behaviour).

### Questions
DP-Adam refers to the algorithm considered by (Balles and Hennig, 2018), i.e., to the plain Adam with DP-SGD gradients, right? Why not to analyze other adaptive optimizers like the versions of DP-Adam that are tailored for DP (e.g., Tang and Lécuyer, 2023, or Li et al., 2023, the references you also list)? Were DP-Adam and DP-SignSGD chosen because the analysis is amenable for them?

Recently, certain filtering methods have turned out to give good privacy-utility trade-offs, see e.g.

Zhang, X., Bu, Z., Balle, B., Hong, M., Razaviyayn, M., & Mirrokni, V. DiSK: Differentially Private Optimizer with Simplified Kalman Filter for Noise Reduction. In The Thirteenth International Conference on Learning Representations 2025.

Could this SDE view allow analyzing those methods as well?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
-  This paper investigates differentially private learning through the lens of a stochastic differential equation. Based on theoretical understanding, the paper investigates how the DP-SGD, DP-SignSGD, and its variant DP-Adam perform in various privacy budgets. The authors argue that DP-SignSGD is epsilon-dependent, which reduces the burden of parameter tuning in DPDL.

### Strengths
-	The paper investigates the optimization process of differentially private learning in terms of SDE, which has not been actively investigated.
-	The authors provide a theoretical analysis of why DP-SGD and DP-SignSGD differ in training dynamics, especially with hyperparameter setups.
-	Based on their observations, the authors argue two protocols that cover both fixed and tuning parameters.

### Weaknesses
Please refer to the Questions section.

### Questions
-	The paper investigates the difference between DP-SGD and DP-SignSGD in terms of differentially private deep learning. Is there any related work on non-private optimization, or does this comparison solely rely on a DP sense?
-	For protocol A, how did the authors choose the parameters? For private learning, the clipping value is also as important as the learning rate. Can the authors provide more results while varying hyperparameters in both protocols A and B?
-	The paper’s analysis is based on that the optimal performances of DP-SGD and DP-SignSGD are almost similar (without considering parameter search). However, as far as the reviewer knows, the current methods prefer DP-SGD compared to DP-SignSGD. Does DP-SignSGD still provide comparable results with private fine-tuning or bigger architectures? Refer to [1] or recent tuning methods in larger vision or language-based DP papers.

    [1] Unlocking High-Accuracy Differentially Private Image Classification through Scale, 2022.

-	What about the case of DP-SGD-based Adam, instead of DP-SignSGD-based Adam?

### Soundness
4

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
3

### Summary
Thru discussing how DP noise interacts with adaptivity in optimization, DP-SGD and DP-SignSGD are proposed in this work, where DP-SGD is shown to be converged at a speed independent of ε, DP-SignSGD is with convergence speed scales linearly in ε. Under optimal learning rates, both methods reach comparable theoretical asymptotic performance, while this leaves potential issues in practice.

### Strengths
- SDE-based analysis of differentially private optimizers, using this framework to expose how DP noise interacts with adaptivity and batch noise.
- DP-SGD is shown ito be converged at a speed independent of ε.
- DP-SignSGD: its convergence speed scales linearly in ε, while its privacy-utility trade-off scales as O (1/ε)

### Weaknesses
- The assumptions on SNR (signal-to-noise ratio) are built on linear approximations that are only valid in a high-noise, low-signal regime.
- A general Student-t distribution for batch noise is used to capture heavy tails, while it is not used consistently in assumption B.2..
- The experimental validation for Protocol B on the StackOverflow dataset is missing.

### Questions
- The theoretical analysis is derived for DP-SignSGD, while the conclusions are empirically with DP-Adam. Please provide more discussions and valiations.
- The experimental validation for Protocol B on the StackOverflow dataset needs to be provided.

### Soundness
2

### Presentation
2

### Contribution
2
