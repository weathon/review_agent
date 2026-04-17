# On the Optimization Dynamics of RLVR: Gradient Gap and Step Size Thresholds

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR), which uses simple binary feedback to post-train large language models, has shown significant empirical success. However, a principled understanding of why it works has been lacking.
    This paper builds a theoretical foundation for RLVR by analyzing its training process at both the full-response (trajectory) and token levels.
    Central to our analysis is a quantity called the Gradient Gap, which formalizes the direction of improvement from low-reward to high-reward regions of the response space. We prove that convergence critically depends on aligning the update direction with this Gradient Gap.
    Moreover, we derive a sharp step-size threshold based on the magnitude of the Gradient Gap: below it, learning converges, whereas above it, performance collapses. Our theory further predicts how the critical step size must scale with response length and the success rate, thereby explaining why practical heuristics such as length normalization improve stability and showing that, with a fixed learning rate, the success rate can stagnate strictly below 100%.
    We validate these predictions through controlled bandit simulations and LLM experiments, including training Qwen2.5-7B with GRPO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the conditions for global convergence in reinforcement learning from verifiable reward for LLM use cases. With the specific reward structure in RLVR, i.e., a binary reward at the end of the trajectory, the paper introduces the concept of gradient gap, defined as the gradient of the value function divided by the variance of the reward. The paper uses an analysis similar to the classic optimization literature or the Lyapunov stability analysis to show that if the accumulative gradient alignment (analogous to the accumulative negative drift in Lyapunov analysis) blows up, the policy gradient would obtain global convergence. Experiments on contextual bandits and LLMs are provided to partially validate the theoretical claim.

### Strengths
1. The paper is quite well written and easy to follow for a reader with a minimum optimization background.

2. The analysis of the theorems is intriguing to me. The authors chose a potential function (a generalized Lyapunov function) in the form of $\log\frac{J(\pi_k)}{1-J(\pi_k)}$, related to the binary reward structure of the RLVR problem, which is quite rare but interesting in the literature of optimization analysis. This potential function leads to the exponential convergence in the paper.

3. The problem studied is of relevance, and the theoretical and empirical techniques, for the most part, are solid.

### Weaknesses
My main is in the regime where the theory is interesting. Both theorems are much more informative in the convergence case, which requires $M(K)$ to blow up to infinity. This is a very strong assumption to me, because in classic non-convex optimization for general functions $f$, we could also derive some relation similar to (16), and sum them up, we would obtain that the value improved compared to the initial point is upper bounded by the negative of cumulative gradient norm square, i.e., $f^* - f(x_0) \leq f(x_K) - f(x_0) \leq - \sum_{k=1}^K \|\nabla f(x_t)\|_2^2$. But the gradient norm cannot blow up because it has a natural fixed lower bound. 

Similarly, in this paper, $J_q(K)$ is upper bounded by the optimal value $J(\pi*)$ of the best policy, which may not be $1$. If $J(\pi*) < 1$, then $M(K)$ cannot blow up, since if it blows up, you will have $J_q(K)$ converges to $1$, which is strictly larger than the optimal policy, and it leads to a contradiction. In this case, I believe you can only show $\pi_K$ converges to a stationary policy, but not $J_q(K)$ converges to $1$.

Still, studying the case with $J(\pi*) = 1$ is already interesting enough. I believe what the paper lacks is an in-depth study and discussion of this quantity $M(K)$ and on what circumstances it blows up. For example, are there cases with REINFORCE or GRPO where the cumulative inner product with the gradient converges, but $M(K)$ blows up? The best case is if we estimate $w_k$ to be exactly the gradient gap. Then, will the squared norm of the gradient gap diverge in almost all RLVR instances? This needs to be proved. In the presentation of the current version, I don't see clearly the difference between the role of the inner product with gradient gap in (11) and the role of the squared gradient norm, which is a pity. 

I will raise my score if, and only if, I'm convinced by theoretical arguments that $M(K)$ blowing up could happen in most cases, or $M(K)$ blowing up is a strictly weaker notation than the squared gradient norm blowing up, i.e., there is a RLVR instance where $M(K)$ blows up, but the sum of squared gradient norm over time converges.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

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
This paper establishes a theoretical foundation for RLVR by introducing the Gradient Gap concept, proving that convergence depends on aligning updates with this gap and deriving a critical step-size threshold that explains empirical heuristics like length normalization.

### Strengths
1. Current RLVR algorithms are mostly empirically based, and theoretical analysis is indeed a relatively underexplored area. How to train models stably is one of our key concerns.
2. The authors' theoretical analysis throughout the paper is quite rigorous, with step-by-step derivations that are convincing.
3. In Section 4, the authors analyze the strengths and weaknesses of state-of-the-art algorithms (such as GRPO and Dr. GRPO) based on the theory presented, demonstrating its practical relevance.

### Weaknesses
1. The manuscript requires further proofreading. For example, the section numbering of 1.1 Related Work is unusual, and in Section 5.2, the authors mention conducting two sets of GRPO experiments but claim to have conducted three. Such typos somewhat affect readability.
2. Some assumptions are overly idealized, revealing a gap between theory and practice.
3. This paper is almost purely theoretical, and the experimental validation is somewhat insufficient.

### Questions
1. In Equation 14, why is the $max(0, ·)$ operation applied to $\Delta\mu_q(k)$ ? If M(K) is intended to represent "useful progress," wouldn’t it be more reasonable for $\Delta\mu_q(k)$ to be negative when optimizing in the opposite direction? Additionally, could the authors briefly explain how M(K) was designed? Was it solely based on intuition?
2. In Section 4, the authors analyze the advantages and disadvantages of GRPO and Dr. GRPO. Is it possible for the authors to design a more effective algorithm based on the theory presented in this paper?

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
2

### Summary
The paper provides a convergence analysis of RLVR with binary rewards and a single prompt.
It defines the notion of a "gradient gap", a rescaled version of the policy gradient, and derives convergence conditions based on it.
Finally experiments are conducted finetuning Qwen with GRPO.

### Strengths
* The convergence of RLVR methods is an interesting topic
 * I am not aware of a similar analyzes for RLVR
 * The presentation of the paper is overall clear

### Weaknesses
I will preface this by stating that **I do not work on theoretical analyses of policy gradient methods** and my comments are thus not very well informed.

* It seems the notion of "gradient gap" is simply a rescaled policy gradient, which the authors also state. It is introduced as the "improvement direction from low- to high-reward responses", which is exactly the policy gradient direction and well known. It is thus not clear to me why this is a useful new notion.
The paper also argues that "[gradient gap] is not scaled down by variability factor [...], making it a purer indicator of where to move [than the policy gradient]". However, it seems that for example gradient normalization would be a simpler way to achieve the same goal.
 * It is not clear from the draft which parts of the theoretical analysis are novel and which parts are based on prior work.
 * The theoretical analysis focuses on a single state/prompt per batch and thus neglects the group-wise advantage estimation of GRPO. However group-wise estimation is one main difference between GRPO and REINFORCE and would be useful to be discussed.

### Questions
* I would be grateful if the authors could explain further why the notion of gradient gap is necessary as opposed to using the normal policy gradient. 

 * Most applications of RLVR use adaptive optimizers like Adam. It would be useful to discuss how this interacts with the step-size selection?

### Soundness
3

### Presentation
3

### Contribution
2
