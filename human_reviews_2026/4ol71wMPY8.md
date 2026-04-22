# Distributions as Actions: A Unified Framework for Diverse Action Spaces

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
We introduce a novel reinforcement learning (RL) framework that treats parameterized action distributions as actions, redefining the boundary between agent and environment. This reparameterization makes the new action space continuous, regardless of the original action type (discrete, continuous, hybrid, etc.). Under this new parameterization, we develop a generalized deterministic policy gradient estimator, Distributions-as-Actions Policy Gradient (DA-PG), which has lower variance than the gradient in the original action space. Although learning the critic over distribution parameters poses new challenges, we introduce Interpolated Critic Learning (ICL), a simple yet effective strategy to enhance learning, supported by insights from bandit settings. Building on TD3, a strong baseline for continuous control, we propose a practical actor-critic algorithm, Distributions-as-Actions Actor-Critic (DA-AC). Empirically, DA-AC achieves competitive performance in various settings across discrete, continuous, and hybrid control.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework for reinforcement learning called "distributions-as-actions". The authors try to redefine the boundary between the agent and the environment. Instead of the agent outputting an action from a potentially discrete or hybrid space, the agent outputs the continuous parameters of a probability distribution over actions (e.g., the probability vector for a categorical policy, or the mean and variance for a Gaussian policy). The environment is then responsible for sampling the final action from this distribution. This reparameterization transforms any RL problem into one with a continuous action space, allowing for the development of unified algorithms.

The authors derive a corresponding policy gradient estimator. To address the challenge of learning a critic over this new, more complex action space, they propose a simple and effective technique called Interpolated Critic Learning (ICL). Finally, they combine these ideas into a practical actor-critic algorithm, DA-AC, based on td3.

### Strengths
1. The core idea of treating distribution parameters as actions is a simple and interesting concept. It unifies diverse action spaces (discrete, continuous, hybrid) into a single, continuous control problem, potentially simplifying algorithm design
2. The introduction of Interpolated Critic Learning (ICL) is a practical solution to the challenge of learning the critic in this new, higher-dimensional parameter space
3. The authors test their method on over 30 tasks across continuous, discrete, high-dimensional discrete, and hybrid domains, consistently showing competitive performance

### Weaknesses
1. The paper presents the transformation from discrete to continuous action spaces as a clear benefit, but does not sufficiently discuss the potential downsides. Discrete action spaces have a specific structure that is often leveraged by advanced algorithms. For instance:
- Entropy regularization is commonly used in RL for promoting exploration in discrete spaces. It is unclear how an equivalent and effective exploration strategy would be implemented in the continuous parameter space. Simply encouraging entropy over the parameters (a vector of probabilities) is not the same as encouraging entropy in the resulting categorical distribution.
- Methods like mirror descent are well-suited for optimization over the probability simplex. By moving to an unconstrained Euclidean space of parameters, these geometric advantages are lost.
- Planning algorithms like MCTS are fundamentally built on discrete action branching. The proposed framework makes it difficult to see how such methods could be integrated. A deeper discussion of what is "lost" in this transformation and how the framework might address these challenges would significantly strengthen the paper.
2. The paper should better position itself with respect to other works that also re-frame RL problems through representation learning. For example, "Representation-Driven RL" (Nabati et al.)  also learns a latent space for policies and uses bandit algorithms to guide exploration in that space. While the mechanism is different (RepRL learns a representation of the entire policy, whereas this paper redefines the action at each step), the high-level goal of shifting the problem into a more structured latent space is shared. Discussing these connections would provide a richer context for the paper's contribution.
3. The comparison to PPO, a very strong on-policy baseline should be more convincing. PPO's performance is known to be highly dependent on implementation details and computational budget (e.g., total environment steps, number of parallel environments). Without a more detailed analysis or comparison against a highly-tuned PPO baseline known to achieve SOTA on these tasks, it is difficult to be certain that DA-AC's superior performance isn't an artifact of a weakly-scaled baseline. A stronger claim could be made by showing that DA-AC is more sample-efficient or reaches a higher asymptotic performance in a "computationally unconstrained" setting where both algorithms are given unconstrained resources.

### Questions
- Could the authors elaborate on how entropy regularization could be effectively adapted to the distributions-as-actions framework? 
- How does one encourage diverse actions by acting in the continuous parameter space of that distribution?
- What are the authors' thoughts on the compatibility of this framework with planning algorithms like MCTS, which rely on a discrete branching factor? 
- Does this reframing inherently limit the applicability to model-free actor-critic methods?
- Regarding the PPO comparisons, can the authors provide more details on the implementation and tuning? For example, were the reported results for PPO based on a single-environment setup, or was it scaled with parallel environments as is common for achieving top performance?
- The ICL technique uses linear interpolation between the policy's output parameters $U_t$ and the deterministic parameters $U_{A_t}$. Could you provide more intuition on this choice? For categorical policies where the parameter space is a simplex, does this linear interpolation respect the geometry of the space, or does it operate on the unconstrained logits

### Soundness
3

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
The paper proposes a unified framework for handling both discrete and continuous action spaces, which traditionally require distinct design choices in reinforcement learning. Instead of directly learning over the original action space defined by the MDP, the method learns policies and value functions over the distribution parameters from which actions are sampled. Because distribution parameters are typically continuous, both discrete and continuous action spaces can be handled in a unified manner. Central to the paper is the introduction of a gradient estimator defined on these parameters that exhibits lower variance but higher bias. This bias issue is effectively mitigated through an interpolated critic learning mechanism. Taken together, these advantages appear to be the primary factor driving the strong empirical performance of the proposed approach across a range of tasks.

The work evaluates the method on tasks with diverse action space types and demonstrates consistent performance advantages.

### Strengths
The paper is well-structured and comprehensive. While the idea may appear straightforward at first glance, the authors provide thorough motivation, analysis, implementation details, and experimental results, which I find convincing.

### Weaknesses
Based on my reading, the paper does not include a dedicated related work section where the connection between this approach and prior research is explicitly discussed. While the authors do reference commonly used policy gradients in Section 2 and baseline algorithms in the experiments, I would be interested in a deeper discussion of previous efforts that also attempt to learn policy distribution parameters—perhaps separately in continuous and discrete action spaces. Such a contextualization would help better highlight and clarify the contributions of the proposed method.

### Questions
As noted in the weaknesses section, I would encourage the authors to provide a more explicit discussion of how this work relates to prior literature.

### Soundness
3

### Presentation
3

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
Distributions as actions is a new way for representing the policy.
Instead of predicting an action, the policy predicts the action distribution.

This work then presents how to integrate this into the env (the env has a sampler function) and into the critic (the critic estimates V of the distribution) and how to train the critic (since next-states arrive from sampling from the distribution and then simulating).

### Strengths
DA-AC seems like a very promising framework.
The work itself is interesting and the concept is novel.
Results show it outperforms pre-existing methods on simple tasks.

### Weaknesses
While the authors show the interpolated critic is an important feature and very nicely present how this helps the critic learn a better optimization landscape, it isn't clear what part of the agent is benefitting from this method.

When comparing to TD3, how does the policy performance change if sigma is fixed and it only learns the mean? (mimicing how TD3 learns).
What happens if TD3 is trained like in FB-CPR (Tirinzoni et al). There the agent predicts the mean and the action is sampled through a gaussian distribution. The Q policy gradients can still flow through this sampling (reparam trick). In this comparison, TD3 could predict mean and std.

### Questions
When comparing to TD3, how does the policy performance change if sigma is fixed and it only learns the mean? (mimicing how TD3 learns).
What happens if TD3 is trained like in FB-CPR (Tirinzoni et al). There the agent predicts the mean and the action is sampled through a gaussian distribution. The Q policy gradients can still flow through this sampling (reparam trick). In this comparison, TD3 could predict mean and std.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work focuses on the policy distributions and proposes a distributions-as-actions framework, an alternative to the classical RL formulation that treats the parameters of parameterized distributions as actions. By shifting this agent-environment boundary, they develop a continuous-action algorithm for a diverse class of action spaces and achieve lower variance. Across experimental results, this paper achieves competitive performance.

### Strengths
The motivation is novel and interesting, in which parameterized action distributions are an important topic in the RL community.
The reparameterization policy gradient estimator can reduce variance using interpolated critic learning.
And under selected baselines, this work achieves competitive performance.

### Weaknesses
1. The baselines are weak, only SAC, TD3. More baselines should be considered in various settings across discrete, continuous, and hybrid control. The newest baseline was published in 2019.

2. Compared with continuous actions, will discrete action spaces affect performance?

3. The related works on parameterized action distributions are insufficient.

Discretizing continuous action space for on-policy optimization, AAAI, 2020 

Discretizing Continuous Action Space With Unimodal Probability Distributions for On-Policy Reinforcement Learning, IEEE TNNLS, 2024.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3
