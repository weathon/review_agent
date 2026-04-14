# Almost Optimal Batch-Regret Tradeoff for Batch Linear Contextual Bandits

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
We study the optimal batch-regret tradeoff for batch linear contextual bandits. For this problem, we design batch learning algorithms and prove that they achieve the optimal regret bounds (up to logarithmic factors) for any batch number $M$, number of actions $K$, time horizon $T$, and dimension $d$. Therefore, we establish the \emph{full-parameter-range} (almost) optimal batch-regret tradeoff for the batch linear contextual bandit problem. 

Along our analysis, we also prove a new matrix concentration inequality with dependence on their dynamic upper bounds, which, to the best of our knowledge, is the first of its kind in literature and maybe of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the batch linear contextual bandits with stochastic contexts. They design some batch learning algorithms and prove that they achieve the optimal regret bounds up to some logarithmic factors for any batch number, number of actions, time horizon, and dimension. They also develop some new techniques which may be of independent interest.

### Strengths
The paper presents near-optimal results for this type of bandit problem, with matching upper and lower bounds that may represent a definitive solution in this area. Additionally, the authors introduce a novel concentration inequality tailored to this problem, which has the potential to be valuable for addressing other bandit problems as well.

### Weaknesses
It would be better if some numerical experiments were included.

### Questions
- Could the authors consider including additional experiments in this section? Additionally, what would be the key challenges in implementing this algorithm?
- Could the authors provide more detail on the hidden constants in the upper and lower bounds? Specifically, what are their magnitudes, and what parameters do they depend on?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This research explores batched linear bandits, in which the learner decides the number of arms to pull in each batch, with rewards for the selected arms revealed only at the end of the batch.

The proposed algorithm functions within a predefined maximum number of batches denoted as $M$. It aims to minimize regret. Theorem 2 illustrates the balance between regret and $M$, indicating that the trade-off approaches a near-optimal order, as shown by Theorem 4. The findings suggest that $M = O(\log \log T)$ batches suffice to achieve asymptotically minimax optimal regret.

### Strengths
The theoretical strengths of the paper include providing a tight bound on the expected maximum width, $\mathcal{W}(m,n)$. This enhancement builds upon the results of Zanette et al. (2021) and mitigates the limitations identified in Ruan et al. (2021). Additionally, the matrix concentration inequality presented may be of independent interest.

### Weaknesses
1: Theorem 4 is compared with Gao et al. (2019), yet the more pertinent lower bound by Han et al. (2020) might provide a more relevant comparison. It is unclear why this comparison was omitted.

2: The paper overlooks several significant works on batched multi-armed bandits and linear bandits for asymptotic $T$. 

3: The paper often neglects to condition on $F_k$ throughout many parts of the proof. 

Additionally, a minor notational correction is recommended: "$\epsilon + t$" described as an independent sub-Gaussian noise with zero mean should be correctly noted as "$\epsilon_t$".

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper studies the number of batches vs regret tradeoff for contextual linear bandits. The authors propose an algorithm that achieves nearly optimal regret for a given number of batches M.

### Strengths
The paper considers batch learning in stochastic contextual linear bandits which I believe is an important problem. It proves upper and lower bounds on the regret given number of batches M.
Exp-Policy (algorithm 2) is introduced to solve the distributional optimal design problem needed to design the policy in each batch. The algorithm is novel and simple unlike algorithms proposed in Ruan et al.
To the best of my knowledge multiple of the analysis techniques proposed are novel including the matrix concentration inequality in Lemma 20. 
The regret upper and lower bounds are nearly matching.

Overall, the paper introduces important algorithms and analysis techniques for contextual linear bandits, as well as being the first to prove the optimal regret bound dependency on the number of batches.

### Weaknesses
The computational complexity of the algorithms scale linearly with the number of actions which can be very large. However, this is justifiable since this is a challenging problem even without the batch learning constraints.

### Questions
Could the authors include the computational complexity of the proposed algorithms as a function of K, d, T, M?
Could the authors add motivation for stochastic contexts as this is the main setting in the paper? e.g., what applications would satisfy the stochastic i.i.d. context assumption?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of batch linear contextual bandits, a type of online learning model where decisions are made sequentially, balancing exploration and exploitation. The authors design algorithms for both context-blind and context-aware settings of the batch linear contextual bandit problem. In the context-blind setting, the learner cannot use current batch information to make decisions, while in the context-aware setting, the learner can observe the context in the current batch, allowing potentially more adaptive decisions.

### Strengths
The contributions are: 
1. **Regret Bounds and Tradeoffs:** They analyze the regret, or performance loss, for different batch numbers and prove that their algorithms achieve nearly optimal regret bounds (up to logarithmic factors) across varying batch sizes. For instance, they demonstrate that as few as \(O(\log \log T)\) batches suffice to achieve nearly the minimax-optimal regret without constraints.

2. **Simplification of Algorithms:** Compared to prior work, such as Ruan et al. (2021), the authors propose a simpler algorithm that is more practical to implement and works for all time horizons \(T\) as long as \(T \geq d\) (where \(d\) is the context dimension). This represents an improvement over previous methods, which required large polynomial relations between \(T\) and \(d\).

3. **Matrix Concentration Inequality:** A significant technical contribution is a new **matrix concentration inequality** that depends dynamically on upper bounds, which is useful for the algorithm’s exploration policy. This inequality could be valuable for other research areas in machine learning due to its novel structure.

4. **Context-Aware Performance Gains:** In the context-aware setting, they show that observing the context allows the learning algorithm to achieve better regret bounds by taking advantage of additional information in each batch, demonstrating that optimal regret can be further improved compared to the context-blind setting.

### Weaknesses
1. I believe $m$ and $n$ represent the same quantities (albeit separated by batch index). So I don’t see how this is an improvement over Zanette 2021. See lines~276-281.

### Questions
1. The algorithms are based on restarting Lin-UCB at every epoch. How does this relate to designing forgetting-based episodic RL algorithms?

### Soundness
3

### Presentation
3

### Contribution
3
