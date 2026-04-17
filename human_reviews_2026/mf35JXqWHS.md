# Noise Tolerance of Distributionally Robust Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Given the importance of building robust machine learning models, considerable efforts have recently been put into developing training strategies that achieve robustness to outliers and adversarial attacks. Yet, a major aspect that remains an open problem is systematic robustness to global forms of noise such as those that come from measurements and quantization. Hence, we propose in this work an approach to train regression models from data with additive forms of noise, leveraging the Wasserstein distance as a loss function. Importantly, our approach is agnostic to the model structure, unlike the increasingly popular Wasserstein Distributionally Robust Learning paradigm (WDRL) which, we show, does not achieve improved robustness when the regression function is not convex or Lipschitz. We provide a theoretical analysis of the scaling of the regression functions in terms of the variance of the noise, for both formulations and show consistency of the proposed loss function. Lastly, we conclude with numerical experiments on physical PDE Benchmarks and electric grid data, demonstrating competitive performance with an order of magnitude reduction in computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces WBM, a new regression loss to improve robustness to additive and global noise, addressing the limitations of Wasserstein Distributionally Robust Learning when regression functions are not Lipschitz or convex case. Theoretical results demonstrate that WBM loss aligns better with noise variance than WDRL. Numerical evaluation on PDE operator learning and electric grid forecasting further valid the robustness of proposed method.

### Strengths
1.	The motivation of this paper is good. As WDRL lacks of robustness to global additive noise.

2.	The paper provides rigorous proofs Wasserstein Batch Matching and noise scaling case.

3.	Experiments over diverse tasks validate the effectiveness of proposed WBM.

### Weaknesses
1.	Some assumptions of theoretical results are not reasonable. For example, Proposition 4.1 requires bandlimited and continuously differentiable $f$ and co-monotonic of $g$, which may not hold for modern DNNs. Moreover, Corollary 5.2 relies on strong convexity and smoothness with second to even fifth order, that is not reasonable.

2.	No error bars or confidence intervals. The number of runs only has 13 samples, which is small for modern deep learning.

3.	The computational gains do not show practical analysis. Although a complexity gains achieved, the real-time runtime such as GPU time evaluation is missed.

4.	There is limited comparison with other regression baselines. For instance, Noise2Noise.

5.	Appendix B used partial derivatives of Lagrange multipliers $\alpha$, $\beta$ with respect to $\sigma$ without proving differentiability rigorously, only a sketch via implicit function is provided.

### Questions
1.	How does WBM perform in non-convex neural networks beyond CNN operators?

2.	Can WBM expand to multiplicative or correlated noises?

3.	What will happen if the true regression function is discontinuous? 

4.	How does WBM performance depend on batch size? Is there any optimal range balancing between robustness and bias?

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
This paper addresses robustness to additive noise in regression, proposing a more distributional measure as an alternative to sample-wise Wasserstein robustness. The authors argue that standard Wasserstein robustness fails when regression functions are neither convex nor Lipschitz, and demonstrate that their "batch matching" method performs better with high-variance noise while being computationally cheaper.

### Strengths
-- The focus on global noise in WBM is timely and distinct from adversarial/outlier robustness, while the authors provide a clear drawback to standard adversarial robustness measures such as sample-wise Wasserstein

-- The theory seems genuinely novel (at least, it is to me) and interesting

-- The scheme is model-agnostic and plugs into SGD, and the loss is differentiable via envelope theorem

-- The experiments on PDE learning and time series demonstrate practical utility for the scheme

Overall, I find the paper enlightening on an important limitation of standard sample-wise Wasserstein robustness, with an interesting practical fix as well as supporting analysis.

### Weaknesses
My main concern is that the paper is fuzzy on implementation details, which makes me a bit concerned for the reproducibility of the method, e.g.

-- The WBM scheme's performance must heavily depend on batch size (larger batches = more flexible matching?), but this is barely discussed

-- There is a 10 fold run improvement mentioned but no timing comparisons provided

### Questions
-- Is there any connection between your "Wasserstein Batch Matching" approach and the Central Limit Theorem? If so, I think it would be interesting to make this connection explicit.

-- Why not compare to simpler robust losses like Huber loss or quantile regression?

-- Strong convexity is assumed for SGD stationary distribution analysis (Cor. 5.2), then justified via RKHS regularization. This diverges from the deep-net setting used in experiments. Could you provide some discussion here?

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
This paper studies the problem of robustness to global additive noise such as measurement and quantization noise, which is often encountered in practical regression and physical modeling tasks. The authors introduce a new training framework named Wasserstein Batch Matching (WBM). It replaces instance-wise matching between predictions and responses with batch-level Wasserstein alignment between their empirical distributions.  

Theoretically, the paper proves (i) **consistency** of WBM estimators (Proposition 4.1), and (ii) **favorable noise scaling properties** (Proposition 5.1), showing that WBM loss grows sublinearly with noise variance σ² compared to MSE and Wasserstein DRO (WDRL). Empirical results on **PDE operator learning (wave & Navier–Stokes)** and **electric grid forecasting (TSMixer)** confirm that WBM achieves significantly better robustness under Gaussian and especially Cauchy heavy-tailed noise, with lower computational cost than divergence-based DROs (CVaR-DRO, Chi²-DRO).

### Strengths
+ Addresses a **real and underexplored problem** (robustness to global additive and heavy-tailed noise).
+ Provides **rigorous theoretical analysis**, including a new _noise scaling law_ comparison with MSE and WDRL.
+ Demonstrates **consistent empirical gains** on PDE and time-series tasks, matching theoretical expectations.
+ Clear writing and strong logical flow from motivation → theory → experiments.
+ Computation-friendly and easy to implement (batch-level matching without extra hyperparameters).

### Weaknesses
+ The theoretical contribution is relatively limited from the optimal transport perspective. The paper mainly adapts existing Wasserstein formulations to a batch-level matching context rather than developing new theoretical results or convergence guarantees.
+ Given the moderate theoretical depth, stronger empirical support is needed to demonstrate the method’s effectiveness. However, the experiments are narrow in scope, focusing only on two case studies (PDE operator learning and electric load forecasting).
+ The lack of broader evaluations across different domains such as tabular, visual, or structured regression reduces the generality of the claims. More systematic ablation studies on batch size, noise variance, and sample size would strengthen the experimental evidence.
+ Claims about the model-agnostic property and computational efficiency of WBM are qualitative and are not supported by formal complexity analysis or quantitative runtime evaluation.

### Questions
1. The paper shows clear improvements on PDE operator learning and electric load forecasting. How effective is the proposed WBM method more generally? Could it achieve comparable gains on other types of regression or prediction tasks beyond these two domains?
2. How sensitive is the method to design choices such as batch size, noise level, and the Sinkhorn regularization parameter?
3.  The paper claims that WBM is model-agnostic and could, in principle, generalize beyond regression tasks. However, no experimental or theoretical evidence is provided to support this claim. Could the authors clarify whether WBM has been tested or formally analyzed for classification or structured prediction settings ？
4. The appendix mentions that WBM achieves about a 10-fold computational gain mainly because it does not require hyperparameter tuning. Could the authors provide quantitative evidence or runtime measurements to support this statement, or clarify whether the comparison includes hyperparameter search time for other methods?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies robustness to global, additive label noise and argues that standard Wasserstein Distributionally Robust Learning does not improve robustness when the regression function is neither Lipschitz nor convex. It proposes Wasserstein Batch Matching: in each step, match the empirical distributions of predictions and responses within a batch via the 2-Wasserstein distance and train by minimizing that batchwise Wasserstein loss. The authors prove a consistency result for WBM in the noiseless case and analyze noise scaling of the loss and its effect on SGD iterates, showing milder sensitivity for WBM than MSE and linear-in-σ deterioration for WDRL under its assumptions. Experiments on operator learning and electric load forecasting find WBM competitive or better than MSE and divergence-based DRO baselines, especially under heavy-tailed noise, with lower computational burden than WDRL.

### Strengths
- Clear formulation & intuition: Batchwise distribution matching gives a simple, architecture-agnostic robust objective.
- Robustness demonstrated on wave/Navier–Stokes operator learning and ETDataset forecasting with TSMixer; WBM outperforms MSE and beats CVaR/Chi-Sq in cost-adjusted comparisons.
- WBM involves a tractable per-step LP in the response dimension and avoids WDRL’s minimax complexity.

### Weaknesses
- The text says WBM solves an LP costing O(s) with s=dim(Y) and compares to WDRL’s O(s^3) when convex–concave; however, in practice the batch assignment (e.g., Hungarian/OT) can scale with batch size. Please specify the exact solver (e.g., Sinkhorn, network simplex) and report end-to-end wall-clock vs. batch size and s

### Questions
- Could you check the format of the paper? margin and fontsize seem do not mach with the iclr template?
- Missing reference on global regularization using WDR https://arxiv.org/abs/2203.00553

### Soundness
3

### Presentation
2

### Contribution
3
