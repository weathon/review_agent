# QUAD: Q-Gradient Uncertainty-Aware Guidance for Diffusion policies in Offline Reinforcement Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Diffusion-based offline reinforcement learning (RL) leverages Q-gradients of noisy actions to guide the denoising process. Existing approaches fall into two categories: (i) backpropagating the Q-gradient of the final denoised action through all steps, or (ii) directly estimating the Q-gradient of noisy actions. The former suffers from exploding or vanishing gradients as the number of denoising steps increases, while the latter becomes inaccurate when noisy actions deviate substantially from the dataset. In this work, we focus on addressing the limitations of the second category. We introduce QUAD, an uncertainty-aware Q-gradient guidance method. QUAD employs a Q-ensemble to estimate the uncertainty of Q-gradients and uses this uncertainty to constrain unreliable guidance during denoising. By down-weighting unreliable gradients, QUAD reduces the risk of producing suboptimal actions. Experiments on the D4RL benchmark show that QUAD outperforms state-of-the-art methods across most tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers offline RL, and proposes to weigh Q-gradient estimates based on their uncertainty. They show that this leads to clear improvements on the D4RL benchmark.

### Strengths
- Offline RL is an important topic of high practical relevance.
- The paper is well-written and clearly presented.
- The Introduction and Preliminaries does a good job at positioning the paper against previous work while, at the same time, presenting the neecessary background. 
- Figures 1 and 2 are pedagogical and aestethically pleasing.
- Results show improvements on D4RL.

### Weaknesses
**Critical: The core assumption is not true in general, nor backed up with experimental evidence**

On l.202-210 you state that "It is reasonable to assume that $\hat{g}$ provides an unbiased estimate of $g^*$". *I strongly disagree with this statement*. This is a critical flaw, since the rest of paper builds on the alignment loss in eq. (12) that stems from this assumption.

In general, the unbiasedness of an estimator $\hat{y}(x)$ does *not* imply that its gradient $\nabla_x \hat{y}(x)$ is an unbiased or accurate estimator of $\nabla_x y(x)$. Unbiasedness is a *pointwise* property, and interchanging differentiation and expectation, i.e. $\nabla_x \mathbb{E}[\hat{y}(x)] = \mathbb{E}[\nabla_x \hat{y}(x)]$, requires additional regularity conditions such as smoothness and dominated convergence. Even when this interchange is valid, gradient estimators can exhibit high variance or bias in practice. 

At the very least, I would expect an empirical examination of whether the decomposition in eq. (11) is valid.

**Minor: Missing AntMaze baseline**

On most AntMaze tasks, QUAD performs worse than what is reported by Zhang et al., "Entropy-regularized Diffusion Policy with Q-Ensembles for Offline Reinforcement Learning", NeurIPS (2024).

### Questions
- Can you provide convincing evidence (theoretical or experimental) supporting the decomposition in eq. (11)?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors build on a diffusion actor-critic (DAC) style algorithm by introducing an uncertainty quantification (UQ) mechanism to weight an auxiliary loss. In essence, they estimate the Q-function gradient (the gradient of the critic’s Q-value with respect to the action) using an ensemble of Q-networks. This yields an estimate of the mean Q-gradient and its variance for each state-action sample. A per-sample weight $\lambda(s,a)$ is then derived based on the ensemble’s uncertainty: samples with higher variance in the Q-gradient are assigned a lower weight, and those with more confident (lower variance) Q-gradients receive a higher weight. The weight $\lambda$ is applied to the Q-gradient alignment loss in the DAC algorithm, with the intention of down-weighting unreliable guidance. In theory, the optimal weight comes from minimizing a mean-squared error risk function, resulting in an inverse-variance weighting scheme. In practice, the authors approximate this optimal $\lambda$ using the ensemble’s empirical variance and a small regularizer for stability. Finally, this weighted guidance term is incorporated into the diffusion policy training objective (essentially adding $\lambda(s,a)$ times the Q-gradient term to the diffusion model’s loss). The overall approach is a simple fix on top of the DAC framework: it modulates the influence of the Q-gradient guidance by the uncertainty of that guidance.

### Strengths
The experimental results indeed show a little benefit to the original DAC algorithm without using UQ. This justifies the hypothesis that adding UQ to handle bias-variance tradeoff can help with the learning.

### Weaknesses
While this method may improve the original DAC algorithm’s performance by tuning the guidance strength per sample, it amounts to a relatively incremental improvement. Essentially, the authors introduce a well-known statistical technique – weighting by inverse uncertainty – into the existing algorithm. This is a straightforward UQ method rather than a novel RL or diffusion modeling insight. The derivation of the optimal weight $\lambda^(\sigma^2)$ is a direct application of bias–variance trade-off analysis. In fact, the solution simply implements inverse-variance shrinkage, a classic approach where high-variance estimates are systematically down-weighted while low-variance (confident) ones are given full weight. This idea of down-weighting unreliable estimates is not new and has long been used in various domains for risk-sensitive learning.

The contribution here is therefore technically modest: it adds an ensemble-based uncertainty estimation and a weighting formula on top of a prior diffusion RL algorithm. Such a “simple fix” does not substantially expand the theory or capabilities of diffusion models or reinforcement learning algorithms. It offers a practical tweak to improve stability or performance of DAC, but its novelty and conceptual depth are limited.

### Questions
Is there any reason not to compare with offline model-based RL algorithms, e.g., MOPO and MOReL?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents QUAD, a method that enhances diffusion-based offline reinforcement learning by incorporating uncertainty-aware Q-gradient guidance during policy denoising. In diffusion-based offline RL, Q-guidance steers the denoising trajectory toward high-value actions. However, directly estimating Q-gradients for noisy intermediate actions often leads to unreliable guidance. To address this issue, QUAD explicitly models the uncertainty of Q-gradients using a Q-ensemble and adaptively down-weights unreliable gradients throughout the denoising process.

### Strengths
This paper focuses on a critical issue in diffusion-based policy denoising: when estimating Q-gradients for intermediate noisy actions, these actions may lie far from the dataset distribution, leading to unreliable value guidance. To address this, the paper provides a principled theoretical derivation of optimal uncertainty weighting based on mean-squared error (MSE) minimization. Empirically, QUAD achieves SOTA or near-SOTA performance across 18 D4RL tasks, and with its uncertainty-aware formulation, it significantly reduces training variance compared to the backbone algorithm DAC.

### Weaknesses
1.	The diffusion policy’s optimization objective (Eq. 8 in the paper) assumes that $Q_\phi(s, a_t)$ provides meaningful gradients at all timesteps. However, since the critic is trained only on $a_0$ (or near-dataset actions), the gradients for earlier steps are effectively unanchored. Consequently, DAC’s Q-guidance remains reliable only near the final denoising steps (small $t$, low noise) and becomes almost random in the early stages. Although QUAD attempts to down-weight such guidance when the uncertainty of the Q-gradient is high, it does not fundamentally resolve the extrapolation problem. In other words, it cannot yield more accurate estimates when noisy actions are far from the data distribution. While this uncertainty weighting reduces the influence of unreliable gradients, it also weakens guidance precisely when the denoising process requires stronger directional information to reach high-return regions. QUAD’s contribution lies in mitigating this inconsistency through uncertainty weighting, but the core limitation remains—the critic’s validity is unproven for noisy actions.
2.	QUAD trades off computational efficiency for robustness and still relies on heuristic, ensemble-based uncertainty estimation. Training a large Q-ensemble and computing per-sample gradient variance substantially increase computational cost due to multiple forward and backward passes. Moreover, evaluation requires sampling multiple candidate actions, which is inefficient for multi-step diffusion policy sampling.

3.	The uncertainty estimation based on ensemble variance is relatively crude and heavily depends on the diversity of ensemble members. The paper provides no in-depth analysis of the reliability or calibration of the estimated uncertainty.

4.	QUAD assumes independence between the oracle gradient $g^*$ and the stochastic noise term $\xi$ when deriving the variance decomposition (Eq. 28), an assumption that may not strictly hold in practice.

### Questions
1.	How does QUAD compare to bootstrapped ensembles or dropout-based uncertainty estimation in similar settings?
2.	Why we can assume $\hat{g}$ to be an unbiased estimate of $g^*$?
3.	What is the rationale behind Eq. 11 and Eq. 12?

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
5

### Summary
To address the diffusion policy’s challenge of backpropagating the Q-gradient of the final denoised action through all diffusion steps, the paper proposes a novel approach that directly maximizes $Q(s, a_t)$ on the noisy action $a_t$ with a new weighting function $\lambda(s, a^t)$. The authors provide both theoretical guarantees and empirical evidence to support their claims.

### Strengths
The empirical experiments are solid and demonstrate strong performance across benchmarks. The proposed reweighting schedule on $\lambda(s, a^t)$ is an interesting idea that effectively reduces the influence of inaccurate $Q(s, a^t)$ estimates, improving the stability of training.

### Weaknesses
My main concern lies in the theoretical analysis. The proof flow lacks rigor and contains several gaps, and the notations are loosely defined, making it difficult to follow the derivation precisely. Please see my questions

### Questions
- In Eq. (10), why can we assume that $\hat{g}$ is an unbiased estimator of $g^*$? This seems to introduce a large gap — in this case, the expectation of stochastic term is not necessarily zero.
- Please clarify lines 202 and 207: what is the expectation taken over? It seems that it should be $\mathbb{E}_{\phi_k}[ξ] = 0$. What is the distribution of $\phi_k$?
- In Eq. (14) and line 224, please specify which random variable the expectation is taken with respect to.
- Have you considered other kind of risk function, it may provide different property than mse risk function.
- In Eq. (17), note that $v$ is a function of $\theta$, which implies that $v^2$ must be recomputed for each update of $\theta$.
- The paper’s key challenge is estimating $\sigma$ and $v$. However, since these quantities appear to rely on the entire batch of data and $K$ Q-functions for each update of $\theta$ and $\phi$, I am concerned that this may be computationally too expensive in practice.

### Soundness
3

### Presentation
3

### Contribution
2
