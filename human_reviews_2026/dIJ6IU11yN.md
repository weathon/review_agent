# Logarithmic Regret in Preference Learning via Optimistic PAC-Bayesian Particle Ensembles

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The remarkable sample efficiency of preference-based reinforcement learning, which underpins the alignment of large language models with human feedback (RLHF), presents a significant theoretical puzzle. Existing analyses often rely on idealized assumptions, such as infinite-particle ensembles or exact, full-batch gradients, that are disconnected from the practical realities of deployed algorithms. This paperprovides a statistically grounded abstraction of modern RLHF-style training pipelines. We introduce a unified optimistic PAC-Bayesian framework that distills the statistical essence of complex, multi-stage RLHF pipelines into a single, provably efficient online learning algorithm. Our central result is a high-probability regret bound of $\\widetilde{\mathcal{O}}(d_{\mathrm{eluder}}\log T)$ for a rich, non-linear class of reward models, demonstrating when and why logarithmic regret is achievable using finite ensembles and noisy stochastic gradient updates under preference feedback. This unified theory provides an explanation for the sample efficiency of pairwise preference optimization, extends naturally to full Markov Decision Processes, and establishes a theoretical foundation for the empirical success of methods like RLHF.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the dueling bandit problem and introduces a unified optimistic PAC-Bayesian framework that achieves logarithmic regret under general function approximation, even when using finite particle ensembles and noisy stochastic-gradient updates.
The main contribution lies in coupling PAC-Bayesian generalization control with concentration inequalities for stochastic dynamics and Wasserstein stability bounds for particle approximations, providing a cohesive theoretical foundation for efficient preference-based learning.

### Strengths
Overall, the paper is interesting as it proposes a PAC-Bayesian particle analysis framework and successfully achieves logarithmic regret.
However, the main theoretical analyses are too abbreviated, making it difficult to verify the results.
For example, in Lemma D.3, the value of the constant $C$ is not specified, and the proof is overly brief.
Moreover, in Equation (E.3), the exact value of term (II) is unclear, which makes it hard to follow the derivation.

### Weaknesses
**1. Presentation:** Overall, the paper requires significant improvements in presentation and clarity, particularly in the proofs and theoretical explanations.

**2. Regret bounds:** The claimed logarithmic regret appears questionable.
For instance, consider the linear dueling bandit setting, which is a special case under the bounded eluder dimension assumption. In [1], an $\Omega(d \sqrt{T})$ lower bound was established, which seems to contradict the logarithmic regret result proposed in this paper.

**3. Regret bounds in Table 1:** The comparison with Zhao et al. (2025a) may not be appropriate, since they consider a different objective—the KL-regularized regret—rather than the standard cumulative regret.
Therefore, a direct comparison to their setting is not valid.
Moreover, it appears that Russo & Van Roy (2014) established a $\tilde{O}(d\sqrt{T})$ bound, not $\tilde{O}(d\log T)$.

**4. No experiments:** Although the authors claim that this work bridges the gap between theory and practice, no experimental results are provided.
Without any empirical evidence, it is difficult to assess the practical applicability of the proposed algorithm.


[1] Bengs, Viktor, Aadirupa Saha, and Eyke Hüllermeier. "Stochastic contextual dueling bandits under linear stochastic transitivity models." International Conference on Machine Learning. PMLR, 2022.

### Questions
1. What is the key mechanism that enables the proposed method to achieve logarithmic regret, rather than the usual $\sqrt{T}$-type regret?

2. Could the authors provide a clear and thorough comparison with the related studies listed in Table 1, explicitly highlighting the differences in assumptions, regret definitions, and problem settings?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses a significant "theory-practice gap" in Reinforcement Learning from Human Feedback (RLHF), the process used to align large language models. In practice, RLHF is extremely sample-efficient, capable of aligning massive models with relatively few human preferences2. This suggests the theoretical regret (a measure of learning efficiency) should be logarithmic ($O(log~T)$).

### Strengths
The paper analyzes a practical algorithm for preference-based contextual bandits that uses finite ensembles and mini-batch SGD. It proves that this algorithm achieves a high-probability cumulative regret bound of $Regret(T) = O(d_{eluder} \log T)$. This bound includes explicit, lower-order terms that quantify the practical algorithmic costs of using discrete-time updates, finite ensembles, and mini-batching, thereby closing the "four gaps" between theory and practice.

### Weaknesses
1. The authors claim to "close the theory-practice gap" and explain the efficiency of practical methods like DPO and RLHF, but it does not actually analyze DPO or the PPO-based algorithms used in practice. However, the paper asserts that OLE "distills the statistical essence"  of practical pipelines, but this connection is conceptual. The entire work hinges on the assumption that OLE is a faithful representation of practical RLHF, yet the paper is "entirely theoretical"  and provides no empirical evidence to validate that OLE itself is a practical, stable, or effective algorithm. 

The authors are suggested to bridge the gap between theory and empirical.

2. Whether the OLE algorithm is computationally efficient is doubtful.

### Questions
1. Could the authors do some simulations to valide the result of their algorithm?

2. Typos: the symbol for the order O seems not standard, \mathcal{O} should be used instead

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an **Optimistic Langevin Ensemble** (OLE) algorithm for preference-based learning (motivated by RLHF) that maintains a finite SGLD-updated particle ensemble and selects actions using an optimism bonus tied to ensemble variance, arguing this variance is a sound proxy for posterior uncertainty via a variance–information-gain duality. It targets four practical gaps—finite ensembles, stochastic mini-batch gradients, discrete-time updates, and intractable posterior uncertainty—and proves a unified high-probability regret bound: a leading $O(d_{\mathrm{eluder}}\log T)$ *exploration cost* plus explicit lower-order penalties for discretization, finite-ensemble Monte Carlo error, and gradient noise. The result provides a theory-of-practice explanation for the sample efficiency of preference optimization and is extended from contextual bandits to finite- and discounted-horizon MDPs with an additional factor.

### Strengths
It is considered to be novel to proposes an algorithm-native framework, an optimistic SGLD particle ensemble, whose uncertainty bonus is tied to ensemble variance and motivated by a variance–information-gain linkage, aligning exploration with the quantity the analysis seeks to control. Furthermore, the results are extended to finite and infinite horizon MDPs.

### Weaknesses
I apologize if my understanding is incorrect. However, at this point, this paper still possesses the following **three** major weaknesses:
- **Bounded function class:** Although it is common to assume the underlying true parameter has a bounded norm, assuming a global norm bound for the whole parameter space needs more careful treatment. In particular, how do we guarantee that $\theta_t^{(i)}$ stays within the bounded parameter space $\Theta$ throughout the training? What should we do if $\theta_t^{(i)}$ goes outside of $\Theta$?
- **Proof gap:** When proving the main theorem, in Appendix E.2, the paper claims that $C^{-1}I(\theta^*; \mathrm{feedback}\_{1:T})\leq O(d_{\mathrm{eluder}}\log T)$ and then refers to (Russo & Van Roy, 2013). However, this result was not found after examing this reference. Can you explicitly state which theorem and how exactly it was applied here? Meanwhile, the writing of the main theorem proof also needs improvement. For example, how do we decompose $\sum_{t=1}^T(\hat{r}_t - \bar{r}_t)$ exactly? Furthermore, the proof of Lemma D.7 is completely missing as the current proof has nothing to do with this lemma.
- **Inapplicable extension:** If my understanding is correct, the extension to MDPs is problematic since constructing the TD targets requires numeric single-step reward, which is general inaccessible under preference feedback. How should we address this issue?

```
(Russo & Van Roy, 2013) Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems, 26, 2013.
```

### Questions
- What is $z$ specifically and how to compute the per-example loss $\ell_{\theta}(z)$? How will the choice of $\ell_{\theta}$ affects the final regret bound?
- Should the line 7 of the OLE Generic Template be replaced by "Computer mini-batch gradient $\widehat{\nabla}\_t$" of $\nabla_\theta\mathbb{E}\_{z\sim\mathcal{D}\_t}\ell_{\theta}(z) + \beta\nabla_{\theta}(\log\mu(\theta) - \log\Pi(\theta))$?
- In the OLE algorithm, do we have $\theta_t^{i}\sim\Pi_0$ or $\theta_t^{(i)}\sim\Pi_{t-1}$ for $i>N_{t-1}$? Will this choice matter?
- Are $v_t^2$ and $\sigma_t^2$ bounded for any $t$?

### Writing Suggestions
- $P$ -> $\Pi$ in Equation (3.1).
- For general audience, it should be clear that what the ensemble approximates is the Gibbs posterior instead of the vanilla Bayesian posterior.
- Some symbols are missed in Equation (D.1).

### Soundness
1

### Presentation
2

### Contribution
2
