# Learning in Circles: Rotational Dynamics in Competitive Reinforcement Learning

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Optimization in competitive reinforcement learning (RL) differs from standard minimization. Actor–critic methods, in single- and multi-agent (MARL) settings, involve coupled objectives, so optimizing them jointly requires finding an equilibrium rather than performing independent descent. Through operator-theoretic viewpoint, we show that actor–critic models inherently exhibit rotational dynamics during learning, cycling around equilibria, thereby explaining in part the instability often observed in practice.  Through the variational inequality (VI) framework for studying equilibrium seeking problems, we adopt the Lookahead method for VIs, which suppresses these rotations in actor–critic RL. Building on this, we introduce *Lookahead-(MA)RL (LA-(MA)RL)* to efficiently mitigate rotational dynamics. Across classical two-player games and multi-agent benchmarks, including *Rock--paper--scissors*, *Matching pennies*, and *Multi-Agent Particle environments*, LA-MARL consistently improves convergence and stability. Our results highlight optimization as a critical yet underexplored lever in RL: by rethinking the equilibrium-seeking dynamics, one can achieve substantial stability and performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates fundamental instabilities in competitive and multi-agent reinforcement learning (MARL), showing that when agents learn jointly (for example via actor–critic methods), the optimization dynamics are more properly viewed not as a minimization problem but as an equilibrium-seeking problem. In particular, the authors argue that the learning dynamics exhibit rotational flows (i.e., cycling around equilibria) rather than simple descent, which partly explains the observed instability or non-convergence in MARL. To address this, they recast the problem in the framework of Variational Inequality (VI) and draw on operator-theoretic methods from VI theory. Concretely, they propose using the Lookahead method (originally developed for VIs) to suppress rotational dynamics and integrate it into MARL, yielding the algorithm they call LA‑(MA)RL (Lookahead Multi-Agent Reinforcement Learning). Empirical results on classical two-player games (e.g., Rock-Paper-Scissors, Matching Pennies) and multi-agent benchmarks (e.g., Multi-Agent Particle Environments) show that their method improves stability and convergence toward equilibrium strategies compared to standard gradient-based MARL methods. Overall, the contribution lies in: (1) highlighting the rotational component of MARL dynamics, (2) connecting MARL with VI theory, and (3) proposing a practical algorithmic remedy through a Lookahead approach.

### Strengths
I find this paper very well-motivated and insightful. It builds a clear bridge between variational inequality (VI) theory and multi-agent reinforcement learning (MARL), showing that many of the stability issues in competitive learning come from the rotational nature of the underlying optimization field. This perspective feels both intuitive and mathematically grounded.

What I especially like is that the proposed solution — introducing Lookahead and Extragradient updates — is simple yet principled. It doesn’t require redesigning the RL algorithm, only rethinking how gradient steps are taken, and the resulting stability improvements are convincing.

Finally, the paper is very well presented: the visuals clearly illustrate the rotational dynamics, and the theory connects nicely to the experiments. Overall, it’s a rare example of a paper that deepens our conceptual understanding of MARL while also offering a practical fix.

### Weaknesses
While the paper is strong conceptually, its experimental scope is somewhat limited. Most results focus on small-scale or two-player settings, and it’s not entirely clear how well the proposed approach scales to more complex multi-agent or high-dimensional RL environments. The theoretical discussion of rotational dynamics is compelling, but it would be even stronger if the authors provided quantitative analyses or ablations showing how Lookahead affects convergence rates or equilibrium quality in larger systems.

### Questions
- The paper shows clear benefits in two-player and small-scale MARL settings. Do you anticipate any challenges or modifications needed for scaling the VI-based Lookahead approach to environments with many agents or continuous control tasks?

- The analysis decomposes learning dynamics into symmetric and antisymmetric components. Could the authors elaborate on whether this decomposition can inform adaptive algorithms — for example, automatically adjusting the update step or damping based on the observed rotational strength?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents an operator-theoretic viewpoint on actor-critic learning in MARL to show that AC methods exhibit rotational dynamics during learning, resulting in cycling around equilibria. They use a VI framework to study MARL as an equilibrium-seeking problem, and adopt the lookahead method for VIs, resulting in an algorithm which mitigates rotational dynamics. They test this method on some classical MULTI AGENT benchmarks.

### Strengths
1. The work is well-motivated. Instability is a well-known but understudied phenomenon in MARL, and the VI perspective sheds a light most MARL practitioners have probably not seen before. The work also presents a good overview of current literature with respect to VIs in the appendix.
2. The proofs of off-policy AC methods as VIs appears correct in Lemma 1 and Appendix C (Have not checked Appendix D).
3. The results for LAMARL applied to MADDPG and MATD3 are convincing, especially RPS and MP.

### Weaknesses
1. What are the error bars in Figures 3 and 4? This should be clearly noted such that the statistical significance of these results can be understood.
2. Why were more off-policy baselines not compared against? The Soft Actor Critic is generally considered a superior OP-AC baselines to the ones presented.
3. The benchmarks are relevant, but in my opinion not sufficient. MPE and classical games are necessary to indicate the mitigation of rotational dynamics, but what about more complex multi-agent benchmarks such as SMAC, Mamujoco, of Overcooked?

### Questions
1. The authors claim that an extension to on-policy methods is simple. Why was this not tried? Can the authors provide more insights, perhaps through an ablation, that is indeed feasible?

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
4

### Summary
This paper discusses actor-critic methods in single and multi-agent RL settings. The work adopts the variational inequality viewpoint formulation of the problems to adapt the lookahead VI algorithm and actor-critic methods to the Markov game setting and mitigate cycling behavior due to rotational dynamics. Experiments on rock-paper-scissors, Matching pennies and Multi-Agent Particle environments illustrate the performance of the proposed methods in mitigating cycling behavior.

### Strengths
1. The paper is well-organized. 
2. The VI viewpoint is interesting to be further explored in MARL.

### Weaknesses
Overall this paper does not provide any substantially new contributions for the following reasons: 

1. **VI formulation.** This formulation is well-known in game theory. It has also been exploited in Markov games (see e.g. Giannou et al. 2022). 

2. **Algorithms using VI formulation for Markov games.** The algorithms for VIs to address divergence issues of gradient dynamics are well known and well studied. They have also been introduced and studied in the context of Markov games, see e.g. the paper below among others. 

Anagnostides, I., Panageas, I., Farina, G., & Sandholm, T. (2024, March). Optimistic policy gradient in multi-player markov games with a single controller: Convergence beyond the minty property. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 9, pp. 9451-9459).

3. **Theory.** Beyond the formulation which is not genuinely new, the paper does not provide any theoretical convergence guarantees for the proposed algorithms (unlike some existing works, see examples above).

4. **Experiments.** Simulations are also very limited in scope. Given that there are new substantially new theoretical contributions, one would expect a more comprehensive evaluation of the proposed algorithms on more challenging benchmarks to support the proposed algorithms: 
- Simulations are conducted for small scale toy problems such as Rock-Paper-Scissors and Matching Pennies, MPE is also small scale.
- The considered settings do not even seem to include state transitions, while the title and the preliminaries and notation insist on the RL/MARL setting. The evaluation is mostly on matrix games. 
- It is not clear why we need AC methods in these settings, especially the simple ones. 
- It is well known that algorithms such as extragradient and lookahead mitigate the divergence behaviors of gradient descent dynamics. The experiments provide little additional insights. 

5. **Clarity.** The presentation lacks clarity in many parts of the paper and this makes reading the paper difficult, writing needs substantial improvements: 
- What is exactly setting 1? Is the paper back to single agent RL after introducing Markov games in section 2? The setting and the operators are not explained, I guess some TD learning like algorithms are considered. 
- Actor-critic convergence and convergence in mixing competitive 2 player zero-sum RL involve two separate set of theoretical questions as the rotational dynamics already show up for matrix games (independently of the dynamic RL setting). The game formulation for actor-critic methods (in single-agent RL) is unclear. If there is no game, what do you mean exactly by rotational game-like dynamics. I think the paper would benefit from stating the policy optimization problem and the Markov game problems separately and then introduce actor-critic methods for each one of them separately, then analyse the convergence behavior of AC methods for each one of the settings. 
-  l. 239: ‘The above lemma shows setting 1 has game structure’: what is the considered game here? Actor-critic methods are used to solve policy optimization problems which are optimization problems, once can see them as solving a bilevel optimization formulation where an actor optimizes the expected return whereas a critic minimizes a policy evaluation error. It is unclear what the paper means by game here. Appendix C looks a little clearer but the main part requires entire rewriting in my opinion (sections 3, 4) and new developments.  
-  In Markov games there is no observation space (l. 138-140) in the classical setting unless you consider a partially observable setting which does not seem to be the case in the rest of the paper.  
- The introduction is too descriptive and does not sufficiently emphasize the gaps in the literature that the paper is addressing.

### Questions
- Can you clarify the novelty with respect to prior work? 
- Can you clarify what is game setting considered in section 3 for single-agent RL?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The pervasive instability in competitive actor–critic RL stems less from algorithm design and more from the optimization dynamics. From a variational inequality perspective, the authors show that both single-agent actor–critic and multi-agent systems are equilibrium-seeking games whose Jacobians naturally induce rotational dynamics near equilibria, which causes the cycling. Then the authors introduce Lookahead-(MA)RL, a computationally efficient VI-based optimizer applied directly in the joint parameter space. Experiments on matrix games and competitive MPE benchmarks demonstrate improved stability and convergence over GD-style baselines, without changing agents or objectives.

### Strengths
- This paper offers a clear framing by identifying rotational dynamics as the fundamental cause of instability in competitive RL and formalizing this through a variational inequality perspective.

- This paper is theoretically rigorous, establishing a direct link between rotations in actor–critic methods and the non-symmetric Jacobian / complex eigenpairs structure.

- This paper proposes Lookahead-based VI methods as a practical and computationally efficient solution that can scale across actor-critic variants, single- and multi-agent cases with both on- and off-policy implementations, demonstrating consistent empirical superiority and stability on multiple tasks.

### Weaknesses
- The baseline comparisons are limited, failing to include state-of-the-art MARL approaches.

- The environments are mostly toy games (RPS / MP / simple MPE), missing more complex and realistic MARL benchmarks (e.g., SMAC, football, ...).

- Missing ablation or sensitivity study for the key hyperparameters ($k^{(j)}$, $\alpha$ , ...).

### Questions
- Why aren't standard evaluation metrics like exploitability or NashConv included in experiment results?

- Why in MPE only 5 seeds are used when the environment is more complex?

### Soundness
2

### Presentation
2

### Contribution
3
