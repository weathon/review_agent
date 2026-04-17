# Decentralized Stochastic Nonconvex Optimization under the Relaxed Smoothness

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
This paper studies decentralized optimization problem  $f(\mathbf{x})=\frac{1}{m}\sum_{i=1}^m f_i(\mathbf{x})$, where each local function has the form of $f_i(\mathbf{x}) = {\mathbb E}\left[F(\mathbf{x};{\boldsymbol \xi}_i)\right]$ which is $(L_0,L_1)$-smooth but possibly nonconvex and the random variable ${\boldsymbol \xi}_i$ follows distribution ${\mathcal D}_i$. We propose a novel algorithm called decentralized normalized stochastic gradient descent (DNSGD), which can achieve an $\epsilon$-stationary point at each local agent. We present a new framework for analyzing decentralized first-order methods in the relaxed smooth setting, based on the Lyapunov function related to the product of the gradient norm and the consensus error. We show the upper bounds on the sample complexity of ${\mathcal O}(m^{-1}(L_f\sigma^2\Delta_f\epsilon^{-4} + \sigma^2\epsilon^{-2} + L_f^{-2}L_1^3\sigma^2\Delta_f\epsilon^{-1} + L_f^{-2}L_1^2\sigma^2))$ per agent and the communication complexity of $\tilde{\mathcal O}((L_f\epsilon^{-2} + L_1\epsilon^{-1})\gamma^{-1/2}\Delta_f)$, where $L_f=L_0 +L_1\zeta$, $\sigma^2$ is the variance of the stochastic gradient, $\Delta_f$ is the initial optimal function value gap, $\gamma$ is the spectral gap of the network, and $\zeta$ is the degree of the gradient dissimilarity. In the special case of $L_1=0$, the above results (nearly) match the lower bounds of decentralized stochastic nonconvex optimization under the standard smoothness. We also conduct numerical experiments to show the empirical superiority of our method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies **decentralized stochastic nonconvex optimization** under *relaxed smoothness* . The authors propose **DNSGD (Decentralized Normalized SGD)**, which integrates normalized gradients, gradient tracking, and Chebyshev-based multi-consensus to address potential unbounded or non-Lipschitz local gradients.
A new **Lyapunov function** involving the product of gradient norm and consensus error is developed to prove convergence to an $\epsilon$-stationary point for each agent.
Theoretical results establish per-agent sample complexity and communication complexity, which reduce to near-optimal known results when $L_1 = 0$.
 Experiments on MNIST and Fashion-MNIST empirically validate the method.  

I wish the author could address my concern, and I will improve my score.

### Strengths
- The paper is clearly organized, and the main results and proofs are relatively easy to follow.
- Extends decentralized optimization to relaxed-smooth functions.
- When $L_1 = 0$, the obtained convergence rates match the known near-optimal results, which validates the analysis framework.

### Weaknesses
- The paper mainly combines *normalized decentralized optimization* with *relaxed smoothness*. Both components have been well studied individually, so the contribution is primarily incremental rather than conceptually new.
- The step size and batch size are quite restrictive — specifically, η = O(ε) and b = O(ε⁻²). These requirements may severely limit practical applicability, and the paper does not provide intuition or empirical justification for them.
- There seem to be mistakes or unclear steps in some proofs, which may affect key results such as the claimed **linear speed-up**. (See Questions 3 and 4 below.)
- Experiments are only performed on MNIST/Fashion-MNIST, which are too simple to demonstrate the benefits of relaxed-smooth analysis.

### Questions
**Proposition 1:** Why not directly assume that the global objective $f$ satisfies the relaxed smoothness property? 

**Use of normalization:** What is the intrinsic connection between the normalization technique and relaxed smoothness? The normalization does not seem to improve the theoretical complexity, yet it introduces constraints on the batch size.
 I recommend the authors to refer *“Momentum Improves Normalized SGD”*, which uses momentum to alleviate large-batch-size requirements and might provide inspiration here.

**Equation (14):** In the second line, why is it written as an equality? It appears that this should be an inequality, and the scaling term should be $1/m$ instead of $1/m^2$. Please double-check or provide the missing conditions. If my understanding is correct, this issue could invalidate the linear speed-up claim.

**Lemma 5:** How is the first equality derived? It seems the numerator should be $v_i \|\bar{v}\| - \bar{v}\|v_i\|$. Please verify this step carefully.

### Soundness
4

### Presentation
4

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
This paper investigates decentralized stochastic nonconvex optimization under the relaxed (L0, L1)-smooth condition. The authors propose Decentralized Normalized Stochastic Gradient Descent (DNSGD), an algorithm that ensures every agent converges to an $\epsilon$-stationary point under (L0, L1) conditions. In the special case of L1 = 0, the above results (nearly) match the lower bounds of decentralized stochastic nonconvex optimization under the standard smoothness.

### Strengths
- Develops a new decentralized algorithm that can converge in (L0, L1)-smooth condition.

- Establishes rigorous convergence guarantees for the algorithm.

- Achieves tight bounds when L1 = 0

### Weaknesses
1. **Novelty.** I have concerns about the novelty of this work. The paper appears to be a straightforward extension of decentralized gradient tracking under standard smoothness to the more general $(L_0, L_1)$-smooth setting. The main theoretical development combines several well-known techniques—decentralized gradient tracking, Chebyshev acceleration, increasing batch size, and gradient normalization—without introducing sufficiently new insights or algorithmic innovations.


2. **Gradient dissimilarity.** The proposed algorithm employs gradient tracking, which is typically designed to mitigate the impact of gradient dissimilarity across agents. However, the paper still assumes gradient dissimilarity in its theoretical setup, which seems inconsistent with the purpose of gradient tracking. The authors should clarify why this assumption remains necessary.


3. **Experiments.** The experimental validation is relatively weak. Although the paper aims to address $(L_0, L_1)$-smooth problems, the experiments are limited to simple image classification tasks on MNIST and FashionMNIST. These datasets are commonly used in decentralized optimization under standard $L_0$-smoothness and may not adequately reflect the benefits of handling $(L_0, L_1)$-smooth objectives. The authors are encouraged to conduct more **challenging and representative experiments**, such as **LLM pre-training or fine-tuning**, where the loss functions are empirically closer to the $(L_0, L_1)$-smooth regime.

4. **Limited utility.**  The algorithm requires a batch size that scales with $1/\epsilon^2$, which significantly limits its practicality in real-world applications.

### Questions
1. **Highlight novelty and challenges.** Clearly emphasize the novelty and specific challenges involved in extending decentralized algorithms to the $(L_0, L_1)$-smooth setting. This will help readers better understand what makes this generalization nontrivial and how it differs from the standard smooth case.

2. **Clarify gradient dissimilarity assumption.** Provide a clear justification for introducing the gradient dissimilarity assumption. For example, the authors could establish a lower bound for decentralized optimization under the $(L_0, L_1)$-smooth condition, explicitly showing that gradient dissimilarity naturally arises and cannot be ignored.

3. **Strengthen experiments.** Include more challenging and representative experiments—such as **LLM pre-training** or **fine-tuning**—to better demonstrate the superiority and practical relevance of the proposed algorithm in the $(L_0, L_1)$-smooth regime.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a distributed optimization problem in which the local objectives satisfy a relaxed smoothness property. The notion of relaxed smoothness was originally introduced by Zhang et al. (2020b), and this work adopts a weaker variant proposed by Zhang et al. (2020a). Under this condition, local gradients are neither Lipschitz continuous nor bounded. The analysis further assumes bounded gradient dissimilarity, which plays a crucial role in establishing the relaxed smoothness of the aggregated objective, as shown in Proposition 3.

While relaxed smoothness has been extensively explored in recent years—particularly in centralized settings for training neural network models—this paper advances the study to decentralized optimization. Notably, the proposed method does not employ gradient clipping. Instead, it integrates normalized gradient descent, gradient tracking, and multi-consensus with Chebyshev acceleration. The authors derive both sample complexity and communication complexity results, showing that their bounds match or surpass the best-known results in certain special cases.

The most relevant prior work is that of Jiang et al. (2025), which also studies first-order decentralized optimization under the relaxed smoothness assumption. In contrast, Jiang et al. employ gradient clipping based on access to the mean vector and exact local gradients. The present paper argues that such an approach is not fully decentralized and stochastic.

### Strengths
- This paper extends decentralized nonconvex optimization to the relaxed-smoothness regime, where gradients may be unbounded, which has rarely been studied in decentralized contexts.
- The algorithm presented achieves near-optimal sample and communication complexities.

### Weaknesses
- The algorithmic components (normalization, gradient tracking, Chebyshev multi-consensus) are adaptations of known modules; the contribution lies mainly in combining them under the relaxed-smoothness analysis rather than introducing a fundamentally new decentralized mechanism.
- There is no comparison with Jiang et al. (2025) in terms of sample and communication complexities and experimental performance. They can still note that Jiang et al. (2025)'s approach requires additional sampling and communication rounds to ensure a common mean vector and exact local gradient.
- The presentation is highly technical. Motivating problems and a qualitative description of the framework could significantly improve the paper. For example, the paper directly imposes Assumption 5 on the mixing matrix without explicitly describing the underlying communication network.

### Questions
How does the presented algorithm perform in comparison to Jiang et al (2025)'s algorithm in terms of sampling and communication complexities and experimental performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a DNSGD method for solving distributed stochastic non-convex problems under the assumption of $(L_0, L_1)$-smoothness. The method is designed by combining normalized gradient descent, gradient tracking, and accelerated gossip procedures. The authors provide theoretical guarantees for the method and compare it to concurrent work. They also provide experiments that validate the theoretical results.

### Strengths
1. The proposed results are novel and have theoretical guarantees.

2. The proposed algorithm has significantly better performance compared to concurrent work (see Section 5).

### Weaknesses
1. P. 19 line 981. I think the term $\frac{4\eta\rho\sigma}{\sqrt{b}}$ was missed.  Will this affect other estimates, for example, on $b, T$, and $K$ on pages 20 and 21?

**Minor comments:**


1. p.1 lines 41-44. I think the work arXiv:2507.09823 should be cited in the context of recent results for relaxed smoothness.

2. P.15 line 801-802. I think this transition should include a reference to the fact that $||\overline{x}^t - x_i^t|| \leq \frac{1}{L_1}$, because this fact is provided only on page 17 in the proof of another lemma.

3. P. 19 line 1003-1014. Can you explain these transitions carefully? I am confused because on page 17, line 887 there are estimates for $\rho$ and it seems necessary to prove that these estimates are less than the terms in lines 1003-1008.

**Typos:**

1. P.16 line 824. $t+1 \to t$

2. P.16 line 862. “First” $\to$ “second”

3. P.17 line 888. I think some constants have been missed.

4. P.17 line 903. $= \ \to \ \leq$

5. P.17 line 917. Constant 3 has been missed.

6. P.18 line 950. $b \to \sqrt{b}$

### Questions
I provided my questions in weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
