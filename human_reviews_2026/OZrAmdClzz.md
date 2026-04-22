# Decentralized Nonsmooth Nonconvex Optimization with Client Sampling

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
This paper considers decentralized nonsmooth nonconvex optimization problem with Lipschitz continuous local functions.
We propose an efficient stochastic first-order method with client sampling, achieving the $(\delta,\epsilon)$-Goldstein stationary point with the overall sample complexity of ${\mathcal O}(\delta^{-1}\epsilon^{-3})$, the computation rounds of ${\mathcal O}(\delta^{-1}\epsilon^{-3})$,
and the communication rounds of ${\tilde{\mathcal O}}(\gamma^{-1/2}\delta^{-1}\epsilon^{-3})$, where $\gamma$ is the spectral gap of the mixing matrix for the network.
Our results achieve the optimal sample complexity and the sharper communication complexity than existing methods.
We also extend our ideas to zeroth-order optimization.
Moreover, the numerical experiments show the empirical advantage of our methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Prior work (ME-DOL (Sahinoglu & Shahrampour, 2024)) proposed a decentralized algorithm by extending the optimal zero-th order nonsmooth nonconvex stochastic optimization algorithm (Kornowski & Shamir (2024)) to the decentralized setting. This paper is further extending ME-DOL to a client sampling setting where each iteration only requires one agent to compute its local gradient oracle, while adopting Chebyshev accelerated gossip communication.

### Strengths
- The numerical experiment shows significant improvement in terms of sample and computation complexity, potentially due to the use of Chebyshev accelerated gossiping.

### Weaknesses
- The algorithm design is not clearly motivated. For instance, it is not explained why the algorithm requires performing gossip on two variables $y_i^{k, t}$ and $\Delta_i^{k, t+1/2}$ in every iteration. Similarly, it is not explained why a double loop structure (iterations $k, t$) is necessary.
- While client sampling requires only one agent on the network to access its gradient oracle per iteration, FastGossip in line 13 and line 18 of Algorithm 1 still requires all agents to participate on the communication of every iteration.
- In Lemma 7, I suggest rephrasing "running Algorithm 3" as "running Algorithm 1, 2 and 3".

### Questions
- Is the application of FastGossip in line 13 and 18 crucial in achieving convergence of Algorithm 1? Can FastGossip (Chebyshev accelerated gossip) be replaced by a one step decentralized gossip $z_i = \sum_{j=1}^n p_{ij} z_j$?
- Can the authors provide additional plots on the consensus error during training to give a better picture on whether the speedup of DOC${ }^2$S actually benefits from better consensus due to adopting FastGossip (Chebyshev accelerated gossip)?
- What does "Computation Rounds" in Figure 1 refers to? Does it refers to iterations?
- What does "Communication Rounds" in Figure 1 refers to? Is FastGossip counted as 1 communication round or $R$ communication rounds? It would be better if the x-axis is presented as clear metrics such as "Total Network Transmission in Bytes".
- What is the benefit of adopting randomized smoothing (i.e., $\mu > 0$) in the first-order oracle (Algorithm 3)? It seems that the result of Theorem 1 depends on the condition $\mu = 0$ due to Lemma 4, therefore randomized smoothing is not adopted when a first-order oracle is used.

### Soundness
2

### Presentation
3

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
This paper studies decentralized stochastic optimization where each client’s function is nonsmooth, nonconvex, and Lipschitz continuous. 
The authors propose **DOC$^2$S** (Decentralized Online-to-Nonconvex Conversion with Client Sampling), which integrates partial client participation and multi-consensus steps. **DOC$^2$S** achieves $(\delta,\epsilon)$–Goldstein stationary points with optimal sample complexity $\mathcal{O}(\delta^{-1}\epsilon^{-3})$, computation complexity $\mathcal{O}(\delta^{-1}\epsilon^{-3})$, and communication complexity $\tilde{\mathcal{O}}(\gamma^{-1/2}\delta^{-1}\epsilon^{-3})$, improving prior decentralized algorithms such as ME-DOL. 
The paper also extends **DOC$^2$S** to the zeroth-order (gradient-free) case and presents both theoretical and empirical utilities.

### Strengths
1. Introduces the first decentralized nonsmooth nonconvex method supporting client sampling, enhancing the scalability of the proposed algorithm.

2. Achieves optimal sample complexity and sharper communication bounds than other state-of-the-art methods.

3.  Extends the proposed algorithm to the settings of zeroth-order optimization with dimension-dependent complexity $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$.

4.  Theoretical results are well-grounded and consistent with known lower bounds.

5.  Comprehensive comparison with prior literature clarifies author's contributions.

### Weaknesses
Assumptions (Lipschitz continuity, fixed mixing matrix) may be restrictive in heterogeneous or time-varying networks.

### Questions
1. How sensitive is DOC2S to the network spectral gap $\gamma$ in practical decentralized systems?

2. Does partial participation introduce statistical bias when data are non-i.i.d.?

3. Could DOC2S be extended to dynamic or temporal communication graphs?

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
3

### Summary
The paper introduces Decentralized Online-to-nonconvex Conversion with Client Sampling (DOC²S), a decentralized algorithm for nonsmooth, nonconvex optimization that uses client sampling. 
It integrates the partial participated computation and the multi-consensus steps into decentralized optimization.

Theoretical analysis shows that DOC²S with local stochastic first-order oracle achieves the $(\delta,\epsilon)$-Goldstein staionary points with sample complexity of $\mathcal{O}(\delta^{-1}\epsilon^{-3})$, the computation bounds of $\mathcal{O}(\delta^{-1}\epsilon^{-3})$ and the communication bounds of $\mathcal{O}(\gamma^{-0.5}\delta^{-1}\epsilon^{-3})$.
Furthermore, DOC²S with local stochastic zeroth-order oracle also achieves the $(\delta,\epsilon)$-Goldstein staionary points with sample complexity of $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$, the computation bounds of $\mathcal{O}(d\delta^{-1}\epsilon^{-3})$ and the communication bounds of $\mathcal{O}(d\gamma^{-0.5}\delta^{-1}\epsilon^{-3})$. These upper bounds are sharper than existing methods. 

The paper also conducts numerical experiments on two different models, which support the shaper upper bounds of the proposed methods compared to other methods.

### Strengths
1. The paper solves the decentralized nonsmooth nonconvex optimization problem to a sharper upper bound than existing methods.
2. It incorporates the steps of client sampling and Chebyshev acceleration into the framework of online-to-nonconvex conversion, which does not require all clients accessing their local oracle in per computation rounds, and thus reduces the computation cost.
3. The experimental results support the theoretical findings.

### Weaknesses
1. The numerical experiments are limited to only two models. More experiments on different models and more recent datasets would strengthen the empirical validation of the proposed method.

### Questions
1. In line 145, it may be "zeroth-order" instead of "first-order" , since the discussion is about LSZO.
2. Can you provide additional experiments on newer datasets and larger-scale tasks to further validate the effectiveness and scalability of the proposed method?
3. Is the performance sensitive to hyperparameters like step size? In the theoretical results, the step size needs to satisfy the rate of $\mathcal{O}(\delta \epsilon^3)$, but in the numerical experiments, it seems that a fixed step size is used. Can you add more discussion or ablation study on this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new algorithm for decentralized stochastic non-smooth non-convex optimization using both first-order and zeroth-order oracles. The method uses an online-to-non-convex approach introduced by Cutkosky et al., 2023, as well as a stochastic smoothing technique for non-smooth functions. The authors provide theoretical guarantees for these results, as well as experiments that validate the theoretical findings.

### Strengths
1. The proposed results have theoretical guarantees

2. The proposed algorithm has significantly better performance compared to existing algorithms (see Table 1).

### Weaknesses
**Minor comments:**

P.18 line 966. $g_i^{k,t}$ is not defined in Alg 1 and Theorem 1, only $g_{i^t}^{k,t}$ is defined
 
P.23 line 1231. Can you explain why you assume $\delta < 1$? I think we can't select $\delta$.

**Typos:**

P.19 line 978 $y_i^{k, t-1/2} \to y_i^{k, t-1}$

P.19 line 984 $2n \to n$

P.19 line 1024 $x^{k,t-1} \to \overline{x}^{k, t-1} $

P.24 line 1262. $\sqrt{T} \to 1$

P.24 line 1295. $\nu + L \to \nu + L\delta$,  also for th 3

P.25 line 1319. $4 \to 8$

P.25 line 1319. $k \to c_0$

P.25 line 1327. $D - \epsilon’ \leq 2/3 \to D/(D - \epsilon’)  \leq 3/2$,  also for th 3

### Questions
I provided questions in minor comments

### Soundness
3

### Presentation
3

### Contribution
3
