# O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Score-based diffusion models, which generate new data by learning to reverse a diffusion process that perturbs data from the target distribution into noise, have achieved remarkable success across various generative tasks. Despite their superior empirical performance, existing theoretical guarantees are often constrained by stringent assumptions or suboptimal convergence rates. In this paper, we establish a fast convergence theory for the denoising diffusion probabilistic model (DDPM), a widely used SDE-based sampler, under minimal assumptions. Our analysis shows that, provided $\ell_{2}$-accurate estimates of the score functions, the total variation distance between the target and generated distributions is upper bounded by $O(d/T)$ (ignoring logarithmic factors), where $d$ is the data dimensionality and $T$ is the number of steps. This result holds for any target distribution with finite first-order moment. To our knowledge, this improves upon existing convergence theory for the DDPM sampler, while imposing minimal assumptions on the target data distribution and score estimates. This is achieved through a novel set of analytical tools that provides a fine-grained characterization of how the error propagates at each step of the reverse process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the (sampling) convergence of the DDPM, providing the status of the arts rate $O(d/T)$ under minimal condition.

### Strengths
The paper is generally well-written, and I mostly enjoyed reading it. The results are insightful, and the "minimal" conditions enhances our understanding of what is really the key to the success of score-based diffusion models.

### Weaknesses
There are several weaknesses and questions/comments.

(1) The authors consider $TV(p_{X_1}, p_{Y_1})$ instead of the target distribution $p_{X_0}$ -- this is because score matching is often very bad close to time $0$ (in the continuous limit). People usually do "early stopping" to avoid this bad region (as did in Chen et al.) The authors may make explanation to better guide the readers.

(2) In Assumption 1, the authors assume that $E|X_0|^2 \le T^{c_M}$, meaning that the data size (depending on $d$) is bounded by polynomial in $T$. Is this a valid assumption? In the original paper of Song et al., $T$ is not set to be too large. In some sense, this condition already assumes a tradeoff between $T$ and $d$ implicitly. 

(3) Theorem 1: often for the convergence analysis there are three pieces: (1) initialization error, (2) score matching error and (3) discretization error. (2) and (3) are there, where is (1)? Probably it is absorbed in one of the terms and the authors should explain.

(4) Theorem 1: the score matching contribution $\varepsilon_{\tiny \mbox{score}} \sqrt{log T}$ is impressive, which is dimension free. I would point out another work https://arxiv.org/abs/2401.13115, which proposed a "contractive" version, which also make the score matching contribution to be dimension free. However, it is at the cost of possibly larger initialization error, which requires to choose the hyperparameters carefully to balance the two. This brings back my question (3) on the contribution from initialization error in this paper's setting.

(5) The authors proved the results for DDPM (or VP in the continuous time). I wonder if the arguments/results are specific to DDPM/VP. It is known that e.g., other popular models as VE can be obtained by a reparametrization trick (https://arxiv.org/abs/2206.00364). I think it may be possible to get the results for general class of models, which may be even more significant.

(6) The authors only stated the convergence results for SDE sampler. What about the corresponding ODE sampler? Is there any expectation on even improving the rate using deterministic sampler?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper establishes a state-of-the-art convergence theory for diffusion probabilistic models, particularly focusing on SDE-based samplers. Under minimal assumptions, the authors derive an $O(d/T)$ convergence rate in total variation distance, substantially improving on prior results that typically require stronger assumptions or yield slower rates. The authors achieve this by introducing novel analytical tools that track error propagation across each step in the reverse process, leading to finer control over both discretization and score estimation errors.

### Strengths
The $O(d/T)$ convergence rate achieved for SDE-based samplers matches the rate of ODE-based models, bridging the gap between these methods under relaxed assumptions. This is a notable advancement for SDE-based models, particularly in high-dimensional settings. The analysis requires only finite first-order moments of the target distribution, which is a much weaker condition than those in previous studies. By developing new tools to capture the propagation of errors through the reverse process, the authors provide an elegant framework that simplifies the analysis without resorting to intermediate metrics like KL divergence. This results in more direct and interpretable bounds.

### Weaknesses
I did not identify any major weaknesses in this paper. However, I have a few minor questions. See the next section.

### Questions
1. In line 269, it claims that the error bound holds provided that $T>> d \log^2 T$ (here I simply choose $\delta=1$). However, in line 247, it is stated that the result holds even when $T\asymp d$. This appears to be a contradiction. Could you clarify the correct relationship between $T$ and $d$?

2. What is the intuition behind introducing the generalized density in Section 4? It seems essential for the proof, but only the density of the auxiliary processes could become $\infty$ at certain points? I am just curious about the reason of introducing those auxillary processes.

3. What is the order of the constants in the error bound? e.g. in line 1127, it mentions that these constants need to be large enough. Could they be on the order of $O(d)$ or $O(T)$? If so, would this affect the order of the result of Theorem 1?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the sample complexity of diffusion models with a stochastic sampling process. By introducing two auxiliary sequences, they divide the discretization complexity and the approximated score error and achieve $O(d/T)$ convergence guarantee under a mild assumption on the score function.

### Strengths
1.	The result is really interesting since this work achieves better results without the assumption on the Jacobian of $s_t$.

### Weaknesses
1.	It would be better to highlight the technical novelty compared to [1]. In fact, I think there are many points that are worth mentioning. More specifically, controlling the Jacobian of the ground-truth score and introducing two auxiliary sequences is the source to remove the Jacobian assumption (If I have any misunderstanding, please correct me.). The author can discuss them in detail to help readers to understand these papers.
2.	The noise schedule is highly specific. Though this schedule has been used in some theoretical works [1][2][3], it would be better to discuss this schedule in a real-world setting.

[1] Li, G., Wei, Y., Chi, Y., & Chen, Y. (2024). A sharp convergence theory for the probability flow odes of diffusion models. arXiv preprint arXiv:2408.02320.

[2] Li, G., Wei, Y., Chen, Y., & Chi, Y. (2024, May). Towards non-asymptotic convergence for diffusion-based generative models. In The Twelfth International Conference on Learning Representations.

[3] Li, G., & Yan, Y. (2024). Adapting to Unknown Low-Dimensional Structures in Score-Based Diffusion Models. arXiv preprint arXiv:2405.14861.

### Questions
1.	I wonder if the noise of the sampling process is pre-defined. More specifically, whether $Z_1,…, Z_T$ in (2.4) and (4.2) is exactly the same. In my opinion, if the noise of the ground-truth process and the approximated process are the same, the problem would become easier. Could the author discuss it in detail?

### Soundness
3

### Presentation
2

### Contribution
3
