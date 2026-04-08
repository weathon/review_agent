## Human Reviewer 1

### Summary
This paper introduces Discrete Adjoint Matching (DAM), which is a novel method for solving the following problem: how to fine-tune discrete generative models (like mask-based diffusion Large Language Models) using the reward-based framework called Adjoint Matching (earlier approach works on continuous diffusion/flow models).

The work is a significant contribution, providing a solution that is not only mathematically elegant and theoretically sound but also demonstrably effective in practice, achieving state-of-the-art results on complex reasoning tasks.

The paper's context is entropy-regularized optimization, which is a formal way to fine-tune a pre-trained generative model ($p^{\text{base}}$) to maximize a reward (or minimize a terminal loss, $g(X_1)$) without straying too far from its original training. The previous method for this, Adjoint Matching (AM), is highly effective for continuous models (like image diffusion models) because it uses gradients ($\nabla g$) as a steering signal. However, this gradient-based approach fails for discrete models because it needs to take a gradient over a discrete set of words. The authors bridge this gap. They propose Discrete Adjoint Matching (DAM), a new formulation that adapts AM to the discrete world. The core innovation is replacing the additive gradient correction of AM with a multiplicative correction factor derived from the exponential of the terminal loss difference (i.e., $e^{-g(y)+g(X_1)}$).

### Strengths
- The authors provide a deep theoretical analysis to prove their method for discrete version of AM. They use fixed-point equations to prove that their practical algorithm is guaranteed to converge to the true, theoretically perfect optimal solution
- The authors address a computationally impossible problem in their theoretically optimal solution. They then methodically build a practical solution: estimation via sampling and approximate the correction factor by sampling a few possible futures (K samples). They also adapts the method to masked diffusion models. 
- The experiments are clear and decisive. DAM achieves state-of-the-art accuracy on three difficult mathematical reasoning benchmarks

### Weaknesses
- The algorithm requires $K$ model-forward passes per training step to build its estimator. While this is clearly effective on an 8B model, the cost for fine-tuning much larger models (e.g., 70B+) is not discussed. A small experiment reporting training time vs. final accuracy for DAM and D1 would make the paper's practical claims much stronger.
- A valuable addition to the empirical analysis would be an ablation study on the number of samples K used in the importance-weighted estimator.

### Questions
Please see the weakness above

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes discrete adjoint matching, which is a theoretical framework to conduct reinforcement learning on discrete diffusion models.

### Strengths
1, The motivation is clear and significant, locating at the need of reward-guided fine-tuning of discrete diffusion-based models.

2, The theoretical seems to be sound.

### Weaknesses
This seems to be a quite good paper. But I am not a theory expert. So I will be alert to any issues raised by other reviewers. Also, I want to raise a question about the performance of Llada-8b on GSM-8K. According to [A], the performance of base Llada model on GSM-8K is 80+. But in your paper, the performance is 60-70. Could you please explain this gap?

Reference:

[A] Revolutionizing Reinforcement Learning Framework for Diffusion Large

### Questions
See Weakness.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
1

---

## Human Reviewer 3

### Summary
The paper introduces **Discrete Adjoint Matching (DAM)**, a framework for **fine-tuning discrete generative models**—such as diffusion-based large language models—through an **entropy-regularized optimization** approach. DAM extends the Adjoint Matching (AM) method, previously designed for continuous diffusion models, to discrete state spaces modeled as Continuous-Time Markov Chains (CTMCs).

The authors derive a **discrete analogue of the adjoint system** using Dynkin’s formula, providing a statistical interpretation that avoids the control-theoretic derivations used in AM. This leads to an estimator for the optimal transition rates between discrete states, enabling fine-tuning without requiring differentiability.

To make DAM computationally feasible, the paper introduces **variance reduction** and **importance weighting** techniques, and adapts the method to masked diffusion models, common in language modeling. Theoretically, it provides guarantees via fixed-point and adjoint equations linking DAM to optimal control formulations.

Experiments on synthetic datasets show DAM closely matches the true optimal distributions, outperforming prior baselines like `D1` and `SVDD`. Applied to mathematical reasoning tasks (`GSM8K`, `MATH500`, `Countdown`), DAM yields consistent improvements in accuracy and reward metrics over `D1`.

### Strengths
1. **Clear conceptual motivation:**
The paper addresses a timely and well-motivated gap — extending adjoint-based optimization methods, previously limited to continuous diffusion models, to the discrete generative setting, which is crucial for language and symbolic models.
2. **Principled extension of Adjoint Matching:**
DAM is a nontrivial discrete analogue of Adjoint Matching (AM), retaining its optimization-by-simulation philosophy while adapting it to the constraints of discrete-time, discrete-state Markov processes.
3. **Practical algorithmic contributions:**
The inclusion of variance reduction and importance weighting makes the method computationally tractable and numerically stable, which is key for large-scale discrete models.
4. **Bridge between theory and practice:**
By grounding the derivation in both stochastic process theory and discrete optimization, DAM provides a conceptual bridge linking probabilistic control, discrete diffusion, and reinforcement-style objectives.
5. **Empirical effectiveness:**
Experiments show consistent and interpretable improvements over strong baselines (e.g., D1, SVDD) on synthetic reasoning and mathematical datasets, validating the theoretical claims.

### Weaknesses
1. **Clarity and depth of the theoretical exposition:**
The theoretical development is solid and well-motivated, but occasionally dense. Some key derivations—particularly the transition from Dynkin’s formulation to the discrete adjoint system—could be presented with more intuition and interpretive discussion, to help the reader understand the underlying mechanics beyond the formal algebra.
2. **Limited discussion of importance sampling techniques:**
The paper briefly introduces importance weighting and variance reduction, which are essential for practical implementation, but this part remains somewhat underexplored.
A more detailed analysis of variance control, proposal distribution selection, or adaptive resampling strategies would strengthen the empirical and theoretical credibility of the approach. It would also help to understand how importance sampling interacts with the entropy-regularized objective in high-dimensional discrete settings.
3. **Positioning within the broader literature:**
The paper’s theoretical contribution could be better contextualized by referencing recent advances connecting stochastic control and diffusion models, such as Pham et al. (2025) [1], where the control–diffusion duality and adjoint computations bear a strong resemblance to those derived here. Acknowledging and comparing these perspectives would clarify the novelty and relevance of DAM within this rapidly developing area.
4. **Empirical diagnostics and ablations:**
Although results are consistent and promising, the paper would benefit from additional ablation studies—for example, isolating the contributions of entropy regularization, importance weighting, and the adjoint update itself. Such diagnostics would offer clearer insight into which components most directly influence performance.

[1] Pham, L.T.N., et al. “_Discrete Markov Probabilistic Models: An Improved Discrete Score-Based Framework with Sharp Convergence Bounds under Minimal Assumptions._” Forty-second International Conference on Machine Learning (ICML, 2025).

### Questions
1. **On the discrete adjoint formulation:** Are there particular conditions (e.g., reversibility, bounded rates) under which the discrete adjoint simplifies to known forms?
2. **On importance sampling and variance reduction:**
- The section on importance weighting is promising but rather concise. Could the authors expand on how proposal distributions are selected or adapted during training?
- Can the authors quantify the variance or bias behavior of the estimator as a function of time horizon or dimensionality?
- How sensitive is DAM to poor proposal distributions, and could hybrid approaches (e.g., reweighting combined with entropy regularization) mitigate this issue?
3. **On the link with stochastic control theory:** There are strong conceptual parallels between this work and recent formulations of diffusion models within stochastic control frameworks, such as in Pham et al. (2025) [1]. Could the authors clarify whether DAM can be interpreted as a discrete analogue of the Hamilton–Jacobi–Bellman (HJB) formulation underlying those control-based methods?
4. **On scalability and applicability:**
- How does the computational cost of DAM scale with the number of discrete states or sequence length?
- Could the authors comment on potential approximations or sparsity strategies for handling large vocabularies in text diffusion models?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper under consideration proposes an algorithm to fine-tune a discrete generative model. The core idea is to develop a discrete analog of the Adjoint Matching method recently proposed by Domingo Enrich et al. for continuous models. To achieve this generalization, the authors obtain at Theorem 2.2 an equation for the adjoint discrete variable that mirrors the stochastic Maximum principle for the auxiliary stochastic control problem used in Domingo Enrich et al. The resulting algorithm (Algorithm 1) is then derived following closely the strategy used in continuous spaces, with some differences. For example, samples from the optimal law $p^{\star}$, which are required to leverage the adjoint equation, are approximated using importance weights (Prop 2.4). The paper is completed by further theoretical results that elucidate similarities with the continuous setting (Sec. 3) and a section on numerical experiments where the method is tested both on synthetic examples and more complex mathematical reasoning tasks.

### Strengths
The paper extends an algorithmical framework that has drawn quite some attention for continuous models to the discrete setting in a principled way. The numerical results appear quite convincing.

### Weaknesses
- There is no theoretical guarantees of convergence for the proposed method (as there is none for its continuous counterpart). This is a sever limitation, in my opinion.

* On the methodological side, AM in continuous state space requires the pre-trained model to be such that the initial and final states are independent. I find this a very strong requirement hard to satisfy in practice, unless the pre-trained model had been designed to have such property. If I am not mistaken the same limitation is present here. I encourage the authors to clarify this point.

### Questions
- It seems to me that in equation (6) $f(y)$ should be replaced by $f(y)-f(X_\tau)$



- The paper puts quite some emphasis on the fact that the derivation of Theorem 2.2 avoids "convoluted control-theoretic derivation adopted in original AM and providing a more general framework for adjoint-based estimators." But at the end of the day, it seems to me that equation (8) is a form of the Pontryagin maximum principle for jump process, and I don't feel that it would be so difficult to obtain following for example  the framework developed in Appendix F of [1] for the control of CTMCs. From this perspective, many aspects of (8), such as the multiplicative nature of the control, appear as rather natural than surprising. I believe that adding the statistical estimator perspective offered in this work is valuable, and I am happy to see it. However, offering also a control theoretical approach would make it even clearer why the method proposed here is the discrete counter part of AM which ultimately relies on equation (3), which is nothing but the Pontryagin maximum principle. Relegating the control-theoretic derivation of Theorem 2.2 to the appendix does not appear to be a good choice to me. 

*[1]  Le-Tuyet-Nhi, P. H. A. M., Shariatian, D., Ocello, A., Conforti, G., & Durmus, A. O. Discrete Markov Probabilistic Models: An Improved Discrete Score-Based Framework with sharp convergence bounds under minimal assumptions. In Forty-second International Conference on Machine Learning.*

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4