# Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game

- Decision: Accept (poster)
- Scores: 8, 5, 6

## Abstract
In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex interim robust Markov perfect Bayesian equilibrium, which we proof to exist and the corresponding policy weakly dominates previous approaches as time goes to infinity. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experiments on matrix game, Level-based Foraging and StarCraft II indicate that, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies under worst-case perturbations, showing resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses Byzantine failures in multi-agent reinforcement learning (MARL), where agents can adopt malicious behavior due to malfunction caused by hardware/software faults. First, a new problem setting, called Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP), is proposed to model adversarial intervention as a transition, where originally cooperative joint actions are modified before being executed in the actual environment. The setting is inspired by Bayesian games, where agents have types that need to be inferred by other agents through their beliefs. The goal is to learn a more fine-grained solution than prior work, where policies are able to collaborate with functional agents, while being robust against adversaries therefore finding a better cooperation-robustness trade-off. Ex post mixed-strategy robust Bayesian Markov perfect equilibrium is proposed as a solution concept for BARDec-POMDPs, which is claimed to be weakly dominant over equilibria that were pursued in previous works.

Based on this setting, a two-timescale actor-critic is proposed based on the robust Harsanyi-Bellman equation, defining the maximin value according to the ideal case (before any adversarial modification) and the adversarial case. The policy gradient uses the value of the adversarial case as critic. The approach is evaluated in a variety of games, including a matrix game, a gridworld game, and a map of SMAC. It is also evaluated against different attacker types.

### Strengths
The paper addresses a very important problem, which is often neglected in the MARL community.

The proposed setting (BARDec-POMDP) can model realistic scenarios, where agents can behave adversarially due to malfunctioning.

While most works on robust MARL focus on defenses against any kind of adversarial attack, which leads to conservative policies, the paper aims to find a trade-off between collaborating with actual functional agents and robustness against actual adversaries, which is a more desirable goal for real-world applications. Bayesian games provide a neat framework to discern between these types of agents, which is a convincing proposal.

The solution concept seems valid.

I like the evaluation, which first starts with a small toy problem before scaling up to larger benchmarks like SMAC. I also appreciate the evaluation with different adversarial strategies.

### Weaknesses
My main concern with the paper is the lack of self-containment: The paper excessively references the appendix, which hurts readability (it is not convenient to peek into the appendix for almost every paragraph of the main paper).

The paper needs a clear prioritization of what is really important and what can be safely left in the appendix. For example, important proofs can be sketched informally with little space, but delegating them to the appendix completely is not acceptable.

Before checking, if I can raise my score, I need a revised version of the paper that is more self-contained, e.g., where the number of appendix references is significantly reduced and proofs are at least sketched such that the reader is not forced to switch documents all the time.

**Minor Comments**
- In Dec-POMDPs (Section 2.1), policies select their actions by the history of past observations and actions. Conditioning only on observations is insufficient since they are not Markovian. The definition in Section 2.2 is correct, though.
- Line 119: “was replaced” - “**is** replaced”
- Line 277: “four type of threats” - “four type**s** of threats”
- The text size in Figure 2 is too small for printed versions.

### Questions
1. Why do adversaries condition on observations despite them not being Markovian (in contrast to the non-adversarial policies)? Is this intentional or a mistake?
2. Do you have any intuition on why M3DDPG performs so poorly in the presence of an adversary compared to the standard MAPPO without any robustness mechanism in Figure 4?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the robustness of multi-agent reinforcement learning against Byzantine failures. A Bayesian Adversarial Robust
Dec-POMDP (BARDec-POMDP) framework is proposed.  The theoretical formulation of the problem is one of the contributions of the
paper and an actor-critic algorithm that produces convergence under some conditions.  Convergence is not ensured but experimentally
the approach shows great resilience against a spectrum of adversaries.

The paper makes some strong assumptions, like the fact that in each episode there is only one attacker. Other assumptions are consistency and sequential rationality, which are used to form an ex interim robust Markov perfect Bayesian equilibrium (RMPBE). With some additional assumptions, the existence of ex ante and ex interim RMPBE are guaranteed. For the ex interim equilibrium, the Q function can be written using the Bellman-type equation for the two Q functions, which the paper calls "the robust Harsanyi-Bellman equation." With a few more assumptions, updating the value function via the Bayes rule guarantees convergence to the optimal value of Q.

The algorithm proposed is applied to different problems, a toy metrics game and some map problems. For the attacks, four types of threats are considered: non-oblivious adversaries, random agents, noisy observations, and transferred adversaries. The experimental results cover robustness over non-oblivious attacks and various types of attacks.

### Strengths
The novelty of the paper is in the theoretical formulation of the problem and in the actor-critic algorithm that produces convergence
under some conditions.  Convergence is not ensured but experimentally the approach shows great resilience against a spectrum of adversaries.

### Weaknesses
- The fact that convergence cannot be guaranteed despite the assumptions made is a weakness of the paper. The method proposed is complex, and the assumptions are strong, yet there is no guarantee of convergence
- The proofs are all in the Appendix, which makes the paper quite long, too long for a conference paper, and that makes it appear incomplete.  Even the analysis of two of the environments used for the experiments is in the Appendix. Expecting the reviewers to read a paper of 36 pages is too much for a conference.  Without the Appendices the paper is incomplete.
- The paper is hard to read, it assumes significant knowledge of the field and is full of acronyms and citations that break the flow of the text.

After the discussion and the changes made to the paper, I reduced my criticism about convergence and increased the rating I gave for contribution. However, the presentation needs more work.  Many of the corrections made to the paper have grammar errors, like mismatches of singular/plural. I understand the rush of making the changes, but the writing is not where it should be.  I also still object to the length of the paper.

### Questions
If you were to write a paper that fits in the page limits, would you include some of the material from the Appendices and drop some of the material currently in the paper or leave the paper unchanged?  I am wondering what you think are the most important parts needed to present the work without exceeding the page limits.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The research explores the stability of cooperative multi-agent reinforcement learning when faced with adversarial strategies and disruptions. It introduces the Bayesian Adversarial Robust Decentralized Partially Observable Markov Decision Process (BARDec-POMDP) as a framework for addressing these challenges. The study demonstrates that conventional robust cooperative MARL approaches only achieve an ex interim equilibrium and advances the concept of an ex post equilibrium for BARDec-POMDP, which it argues is superior under worst-case scenarios. To attain this ex post equilibrium, the study presents a novel actor-critic method named EPR-MAPPO. Testing on simple games, the LBF environment, and the StarCraft Multi-Agent Challenge (SMAC) illustrates that EPR-MAPPO outperforms standard models, showcasing resilience against four distinct threat categories.

### Strengths
The manuscript is easy to follow. Addressing the robustness of Multi-Agent Reinforcement Learning (MARL) is crucial, particularly for its practical deployment in real-life situations. The introduction of the ex post equilibrium within the BARDec-POMDP framework by the authors offers a thoughtful contribution to the field. The paper includes an in-depth examination of the algorithms it introduces. Furthermore, the empirical findings presented are coherent and compelling.

### Weaknesses
The document frequently refers to its appendix, suggesting that it does not stand alone as effectively as it could. To enhance the feeling of a complete and self-contained work, the authors should further take simplification, such as using simpler terms within the equations for greater clarity.

### Questions
How can the proposed method benefit real-world applications? Any concrete examples?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
