# Batched Stochastic Matching Bandits

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
In this study, we introduce a novel bandit framework for stochastic matching based on the Multi-nomial Logit (MNL) choice model.  In our setting,  $N$ agents on one side are assigned to $K$ arms on the other side, where each arm stochastically selects an agent from its assigned pool according to an unknown preference and yields a corresponding reward.  The objective is to minimize regret by maximizing the cumulative revenue from successful matches across all agents. This task requires solving a combinatorial optimization problem based on estimated preferences, which is NP-hard and leads a naive approach to incur a computational cost of $O(K^N)$ per round. To address this challenge, we propose batched algorithms that limit the frequency of matching updates, thereby reducing the amortized computational cost—i.e., the average cost per round—to $O(1)$ while still achieving a regret bound of $\tilde{O}(\sqrt{T})$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework called Stochastic Matching Bandits (SMB), which generalizes the standard multi-armed bandit problem to a bipartite matching setting where arms stochastically select agents from assigned pools according to a Multinomial Logit (MNL) choice model with unknown preferences. The goal is to maximize cumulative reward from successful matches while learning these preferences online. A key challenge is the NP-hard computational complexity of the resulting combinatorial assignment problem in each round, which makes a naive approach prohibitively expensive (O(K^N) per round). To address this, the authors propose batched algorithms that limit the frequency of matching updates. Their main contributions are: (1) a batched algorithm (B-SMB) that achieves $\tilde{O}(\sqrt{T})$ regret with $O(1)$ amortized computational cost when a problem-dependent parameter $\kappa$ is known, (2) an improved algorithm that removes the need for knowing $\kappa$, and (3) empirical validation showing significant computational savings with comparable regret to non-batched baselines.

### Strengths
1. The SMB framework effectively models real-world matching markets (e.g., ride-hailing, job platforms) where choices are stochastic and preferences are latent, addressing a significant gap between traditional online matching theory and practical applications.
​​2. The batched approach is a clever and impactful contribution. By reducing the amortized cost to O(1) through limited updates $\Theta (\log\log T)$, the paper makes the problem tractable for large-scale systems without sacrificing theoretical regret guarantees.
​​3. The paper provides theoretical analysis, including regret bounds that match or improve upon existing lower bounds for related bandit problems. The analysis handles the complexity of multiple, interacting MNL models.

### Weaknesses
1. Strong Assumptions:​​ The requirement that the candidate set size |S_k,t| is bounded by a known constant L and that rewards r_{n,k} are known in advance may not hold in all real-world scenarios. These assumptions simplify the problem but limit generality.
2. ​​Empirical Scope:​​ While experiments demonstrate computational gains, the scale of the problems tested (N up to 8, K up to 5) is relatively small. It's unclear how the algorithms perform in massive markets (e.g., thousands of agents and arms), where the computational savings would be most critical.
3. ​​Practical Implementation Hurdles:​​ The algorithms involve solving G-optimal design problems and maintaining complex confidence structures. The practical implementation overhead (beyond just combinatorial optimization cost) is not thoroughly discussed.

### Questions
The proposed algorithms rely on a batched elimination approach that requires solving a combinatorial optimization over the active set M_τ at each update. While the number of updates is small, the cost per update could still be very high if the active set remains large. How does the size of the active set M_τ evolve during the elimination process in practice, and are there safeguards or heuristics to control its size and ensure that the individual batch updates remain computationally feasible, especially for very large N and K?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel "Stochastic Matching Bandits (SMB)" framework, designed to model online matching markets like ride-hailing. In this setting, $N$ agents (e.g., riders) are assigned to $K$ arms (e.g., drivers). The core novelty is that arms select stochastically from their assigned pool of agents according to an unknown Multi-nomial Logit (MNL) preference model. The paper identifies the primary challenge of this framework: the underlying combinatorial optimization problem is NP-hard, leading to a prohibitive $O(K^N)$ computational cost for a naive, per-round approach. The main contribution is a set of "batched" algorithms (B-SMB and B-SMB+) that strategically limit the frequency of these expensive matching updates to only $\Theta(\log\log T)$ times over a horizon of $T$ rounds.

### Strengths
An improved model with solid technical results. The core technical contribution is the elegant solution to the $O(K^N)$ computational bottleneck. The proposal to use a batched algorithm—which runs the NP-hard optimization only $\Theta(\log\log T)$ times—is a highly practical and innovative idea. Proving that this massive reduction in computational effort can still achieve an optimal $\tilde{O}(\sqrt{T})$ regret bound is a significant theoretical achievement.

### Weaknesses
1. Fundamental Mismatch in Problem Scale ($N$ and $K$). This is the most significant concern with the paper. The primary motivation is large-scale ride-sharing platforms (e.g., Uber, Lyft), where the number of agents ($N$, riders) and arms ($K$, drivers) are both massive, potentially in the thousands. However, the paper's entire technical framework is built around solving a core optimization problem that has a computational cost of $O(K^N)$ (as noted in Remarks 5.4). This complexity is exponential in $N$.This leads to a fundamental contradiction:The algorithm is only computationally feasible if $N$ is a very small constant (e.g., $N < 10$). The paper's own experiments, which use $N=7$ confirm this limitation. The proposed B-SMB algorithm could not even run its first step, making the $O(1)$ amortized cost irrelevant.This implies the paper's problem formulation (the $O(K^N)$ optimization) is fundamentally mismatched with its motivating application (the $N=$millions scenario). To make the algorithm's usability and relevance to large markets far stronger, could the authors provide an experiment with a larger $K$ and N  (e.g., $K, N ~ 10,000$)?

2. Theoretical Gap in Regret Bound: The paper itself (Discussion on Tightness) acknowledges a theoretical gap. The final regret bound of $\tilde{O}(K^{3/2}\sqrt{T})$ (for Algorithm 2) is worse than the derived lower bound of $\Omega(K\sqrt{T})$ by a $\sqrt{K}$ factor. While the authors provide a justification (the need to explore all potential matches during elimination), this remains a minor theoretical weakness.

### Questions
see weakness.

### Soundness
3

### Presentation
2

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
This paper investigates the stochastic matching bandit problem with batched updates. The market consists of N agents and K arms, where each arm’s preference is modeled by a multinomial logit (MNL) bandit. Specifically, when multiple agents select the same arm, the acceptance probability of each agent follows a softmax function. The objective is to determine an optimal assortment that maximizes the total expected reward across all arms. The paper proposes a batched learning algorithm and demonstrates that it requires only O(\log \log T) computational complexity while achieving a regret of order O(\sqrt{T}).

### Strengths
1. It is reasonable to incorporate randomness into the decision-making process of the arms.
2. The paper proposes a batched algorithm that effectively reduces the computational cost for an otherwise (per-round) computationally intractable problem.
3. The theoretical analysis is comprehensive, providing both upper bounds depending on $\kappa$ or not.

### Weaknesses
1. It would be helpful to include a motivating example in the problem-setting section. In particular, please provide more intuition about the roles of $x$, $\theta$, and $r$ to clarify how these quantities shape the model and its implications.
2. For batched learning, it is well known that $O(\log \log T)$ switching often suffices to achieve $O(\sqrt{T})$ regret in standard MAB settings. Your result aligns with this pattern for the multi-agent MNL bandit. I encourage the authors to elaborate on the technical challenges unique to this setting—relative both to existing batched bandit frameworks and to the single-player MNL bandit—highlighting what new ideas are required.
3. Though using a batched algorithm can reduce the computational cost, it still needs to solve the combinatorial optimization problem in some rounds, which requires exponential time complexity and is not feasible in large-scale markets. 
4. Minor: Lines 162–163 contain several notational typos.

### Questions
Please see the last part.

### Soundness
3

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
This paper presents a novel approach to solving the stochastic matching bandit problem with batched updates. In this framework, N agents are matched to K arms, where each arm has a preference modelled by a multinomial logit (MNL) model. The acceptance probability for each agent is determined by a softmax function, and the goal is to find an optimal assortment that maximizes the total expected reward. The paper proposes a batched learning algorithm with an amortized computational cost of O(1) and provides a regret bound of Õ(√T), demonstrating both computational and theoretical efficiency. The experimental results show that the proposed algorithm achieves competitive regret while reducing computational overhead compared to traditional methods.

### Strengths
Real-World Applicability: The problem is well-motivated by real-world applications such as ride-hailing or job platforms, where stochastic preferences play a crucial role. The paper's proposed framework extends beyond traditional deterministic matching models to a more realistic, stochastic setting.

Batched Approach: The introduction of a batched learning approach to reduce computational costs is a significant contribution, particularly in large-scale matching markets. This approach ensures scalability while maintaining theoretical guarantees.

Empirical Evaluation: The experimental results demonstrate that the batched approach performs efficiently, with computational cost improvements and comparable regret to existing baselines. The paper provides solid evidence for the practical impact of the proposed algorithms.

### Weaknesses
Lack of Intuition and Motivation: The paper would benefit from a clearer motivating example in the problem-setting section. Specifically, it would be helpful to provide more intuition about the roles of the different parameters and how they influence the model's performance.

Strong Assumptions: The requirement that the candidate set size |S_k,t| is bounded and that the rewards r_{n,k} are known in advance may not hold in all real-world scenarios. These assumptions simplify the problem but also limit the generalizability of the results. The paper should discuss the potential limitations of these assumptions.

Scalability Concerns: While the batched approach reduces computational cost, there is still a combinatorial optimization problem that requires exponential time complexity in some cases. The paper does not fully address how this might limit scalability in very large markets, with thousands of agents and arms.

### Questions
The proposed batched approach eliminates frequent updates, but the cost of solving the combinatorial optimization problem still seems to remain in certain rounds. How does the active set size evolve over time, and are there any safeguards or heuristics in place to ensure that these updates remain computationally feasible for very large values of N and K?

### Soundness
2

### Presentation
3

### Contribution
3
