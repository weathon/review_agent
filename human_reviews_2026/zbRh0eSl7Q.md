# Optimistic Value Iteration with Representation Learning for Low-Rank POMGs

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Partially Observable Markov Games (POMGs) pose significant challenges for multi-agent reinforcement learning due to the combination of partial observability and strategic interactions.
Recent advances explore the inherent structure of the POMG dynamics and develop efficient representation methods to facilitate planning in the latent space rather than directly operating on the history trajectory. 
In this paper, we focus on the low-rank POMGs and propose a unified  optimistic value iteration (OVI) framework that accommodates different low-rank representation learning methods. 
With a given representation, 
OVI constructs an optimistic bonus and integrates it into the value function to inspire exploration and mitigate the bias caused by the representation approximation error.
When the exact value function oracle is unavailable, OVI instead utilizes the low-rank representation to construct optimistic/pessimistic estimators of the value functions via the Bellman recursion, and selects the final solution based on the optimistic-pessimistic gap.
Our theoretical analysis shows that, once the representation approximation error is bounded, the OVI converges to an approximate equilibrium.
We instantiate the framework with two provable representation learning methods: an MLE-based approach and a spectral decomposition representation method.
Furthermore, we develop a novel representation method, $L$-step Latent Variable Representation (LLVR),  for POMGs with infinite-dimensional latent spaces, i.e., infinite rank, and prove that OVI with LLVR  also achieves approximate equilibria, with an extra $L$-decodability assumption.
Collectively, these results establish the first systematic representation learning perspective for POMGs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies low-rank partially observable Markov games in the context of multi-agent reinforcement learning with partial observability and representation learning. 

The authors propose an optimistic value iteration type algorithm and present two specific instantiations based on maximum likelihood estimation (MLE) and spectral decomposition–based representation learning. 

The paper also provides solid theoretical results supporting the proposed methods.

### Strengths
The proposed setting is both novel and important, addressing a crucial gap in multi-agent reinforcement learning under partial observability. 

The paper presents solid theoretical results.

Assumptions are clearly stated and analyses are well-structured.

Overall, the paper is clearly written and easy to follow, with assumptions and main results stated explicitly and intuitively. 

Theoretical bounds are well-highlighted and effectively summarized in Table 1, which helps the reader grasp the key takeaways at a glance.

### Weaknesses
While the proposed Optimistic Value Iteration (OVI) framework may not be entirely new, and similar representation learning approaches have appeared in prior work, extending it to the low-rank partially observable Markov game setting represents a meaningful and valuable contribution. This extension broadens the applicability of OVI and strengthens its theoretical foundation in a more complex environment.

The paper lacks empirical results and concrete motivating examples, which would help illustrate the practical relevance and intuition behind the theoretical findings. That said, the theoretical development alone provides substantial contribution and novelty, making the paper strong even in the absence of experiments.

### Questions
Two relevant prior works could be incorporated and discussed for a more complete positioning of this paper within the literature:

Model-Free Representation Learning and Exploration in Low-Rank MDPs, and

Reinforcement Learning in Low-Rank MDPs with Density Features.

A discussion of how the proposed approach relates to or differs from these studies would help clarify the paper’s novelty and theoretical advancement.

It would also be interesting to consider whether a model-free representation learning approach could be developed under this framework. While the Partially Observable Markov Game (POMG) setting certainly introduces additional layers of complexity, an analysis or discussion of the challenges and potential feasibility of extending the method to the model-free case would add further depth.

The definition of the D hat dataset appears to be missing or unclear—clarifying what it represents and how it differs from the original dataset D would improve readability and precision.

It would strengthen the paper to explicitly highlight the key technical challenges and sources of novelty compared to prior work, perhaps in the main text or in a dedicated discussion section. Such clarification would make the unique contributions of the paper more visible.

Could the proposed setting reduce to the single-agent case? If so, it would be valuable to provide the corresponding theoretical bounds in this simplified scenario and to compare them against existing single-agent results in the literature. This would help situate the contribution relative to known benchmarks and demonstrate consistency with established results.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on partially observable Markov games (POMGs) with low-rank structures. This paper proposes two types of optimistic value iteration (VI) algorithms to compute NE, CCE, and CE in the games, one is oracle-based and the other is oracle-free. To learn the low-rank representations, this paper introduces both an MLE-based approach and a spectral decomposition representation method. Finally, POMGs with infinite-dimensional latent spaces are discussed.

### Strengths
(+) This paper is technically solid and presents comprehensive results.

(+) The low-rank assumption seems to be novel and promising to address the partial observation.

### Weaknesses
(-) The algorithm operates on the joint action space $\mathcal{A}$ of all the players, which could be computationally costly.

(-) The feasibility of the low-rank assumption (Def 4) needs further justification. See Q1 below.

### Questions
Typo: In Definition 4, should it be $\tau\in\mathcal{T}\_h$ and $\tau'\in \mathcal{T}\_{h+1}$?

Q1. Are there any concrete examples of low-rank representations (Def 4)? Of course, POMGs with a discrete state-action space are finite-rank, but the embedding dimension could be exponential to $H$.

Q2. In the OVI-OF algorithm, the computation in Line 15 could be intractable, i.e., integrals over all possible observations. Are there any solutions?

Q3. What is the relationship between Section 5 and the previous sections? The LLVR algorithm deviates from the low-rank assumption and switches to a neural network-based learning method.

Q4. Can you explain how Algorithm 1 is different from [1]?

[1] Ni et al. Representation learning for low-rank general-sum Markov games. ICLR 2023.

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
2

### Summary
This work proposes a unified optimistic value iteration (OVI) framework for solving low-rank POMGs. Combined with MLE, SDR and L-step latent variable representation (LLVR), the OVI algorithm has shown to converge to the approximate NE, CE and CCE, provided the representation error is bounded.

### Strengths
The proposed OVI framework accommodates different low-rank representation learning methods for learning approximate equilibria in low-rank POMGs with finite sample complexity guarantee. The L-step variable representation extends OVI to infinite-dimensional latent spaces under L-decodability assumption may be of independent interest.

### Weaknesses
The OVI framework is intuitive, and there is limited novel design tailored to POMGs. Further, the analysis is largely based on the existing analysis in low-rank MDPs and POMGs.

### Questions
1 See Weaknesses. The main concern is the novelty of this work. It seems the analysis for MLE and SDR based OVI are rather standard. The novelty of the proposed OVI algorithm and the novel techniques are not clearly stated. 

2 While the LLVR representation is interest, the L-decodability assumption seems quite strong and hard to verity. a) For practical problems, are there simple tests for this property? b) Is it possible to relax the L-decodability assumption? Say, if the L-decodability is approximately true.

3 It is known that learning NE in Markov games is much harder than learning CE and CCE. This works assumes that we have access to an oracle for computing NE,CE,CCE. What is the computational complexity for learning NE,CE,CCE in POMGs?

Minor issues:

1* In background, in many places $r(s,a)$ is adopted and should be replaced with $r(o,a)$. 

2* In Lemma 13, $\alpha_N$  as a function of $\zeta$ is not defined.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies low-rank partially observable markov games (POMG). It proposed a value iteration method with optimism design that incorporates a bonus function using learned representations. It then proposed a representation learning algorithm for infinite-dimension latent space with additional L-decodability assumption. The paper proved a polynomial sample complexity under this low-rank POMG setting.

### Strengths
1. Some techniques proposed or adopted in this paper are new and interesting. For example, the adoption of Spectral Decomposition Representation (SDR) enable the algorithm to remove unrealistic oracles.

2. The paper also proposed way to handle infinite-dimension latent space by using variational methods, which is also novel to some extend.

### Weaknesses
1. One weakness or question is about the low-rank structure for POMG. This is also the primary reason that I recommend reject at the current stage.

In definition 4, the low-rank structure takes the form $P(\tau'|\tau,a)=\phi(\tau,a)^\top\mu(\tau')$, while this is acceptable for MDP, in POMDP, we have if $(\tau, a)$ is not the history of $\tau'$, then $P(\tau'|\tau, a) = 0$. This means that if we want to define low-rank structure in this way, we have to enforce $\mu(\tau')$ to be orthogonal to all $\phi(\tau,a)$ as long as $(\tau,a)$ is not the history of $\tau'$.

This implies that $d\geq (OA)^{h}$ since for each $(\tau,a)$, we need distinct direction of $\phi(\tau,a)$, which seems unrealistic.

### Questions
Please see weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
