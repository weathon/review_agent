# Balancing Extremes: Exploiting the Performance Spectrum from Best to Worst in Multi-Agent Systems

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Coordinating exploration and avoiding suboptimal equilibria remain central challenges in cooperative multi-agent reinforcement learning (MARL). We introduce BEMAS (Balancing Extremes in Multi-Agent Systems), a decentralized and proximity-aware framework that exploits the performance spectrum that naturally emerges as agents learn at different rates. During training, agents exchange bounded local messages to identify their best and worst neighbors via phase-aware TD-error scoring: a curiosity score to encourage coordinated exploration and a performance score to guide exploitation. BEMAS couples two shaped signals: (i) optimism, an intrinsic bonus equal to the optimistic action-value gap, with respect to the best neighbor; and (ii) pessimism, a relative-entropy-based repulsion that discourages imitation of the worst neighbor. A schedule down-weights optimism and up-weights pessimism over training, and execution is fully decentralized with no communication. We establish boundedness of the shaping terms and add a Bayesian stability regularizer that limits policy surprise, resulting in stable updates. Across a standard cooperative MARL benchmark, BEMAS proves superior performance compared to baselines, with ablations isolating the contributions of optimism and pessimism. Motivated by group learning theory, the proposed framework provides a simple mechanism that moves toward the best peers and repels weak behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes BEMAS, a decentralized and proximity-aware reward-shaping framework for cooperative multi-agent reinforcement learning (MARL). The core idea is that agents can exploit the performance spectrum within their neighborhood: learning from the best peers (optimism) and avoiding the worst (pessimism). BEMAS includes: a peer-relative action-value gap bonus (optimism) encouraging exploration when outperforming the best neighbor; a KL-divergence–based repulsion (pessimism) penalizing similarity to the worst neighbor’s policy; a Bayesian stability regularizer (based on Dirichlet beliefs and Bayesian surprise) to stabilize updates; a phase-aware schedule that transitions from optimism to pessimism over training.

### Strengths
1. The idea of conbining optimism and pessimism for MARL is novel and intuitive, which I believe can emerge multi-agent cooperative behaviours. 
2. The presetation of this paper is good and easy to understand, with clear equations and figures. 
3. The design of BEMAS is sound, with all expected parts included. 
4. Clear proofs of boundedness for each shaping term (logarithmic optimism, KL-based repulsion, Bayesian surprise) is rigor and seems correct.

### Weaknesses
My major concerns about this paper lies in the experiment part. 
1. The number of baselines chosen is insufficient. I suggest the authors compare BEMAS with other decentralized methods and optimism and pessimism based methods, such as [1] and [2] 
2. The experiments are only conducted on MPE environments with 3 scenarios. It is better to evaluate BEMAS on other partially observable environments. 

Minor concerns
1. Is $T_{exp}$ a hyperparameter? If so, it is important to conduct ablation experiments on this parameter, since it control the influences of the two key parts of BEMAS. 

[1] Conditionally Optimistic Exploration for Cooperative Deep Multi-Agent Reinforcement Learning, UAI 2023
[2] Tactical Optimism and Pessimism for Deep Reinforcement Learning, NeurIPS 2021

### Questions
See Weaknesses.

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
4

### Summary
This paper proposes BEMAS, a decentralized MARL framework that exploits both best and worst performing peers during the training. The key idea is to balance optimism and pessimism through two reward-shaping signals. 1)  An optimism bonus that rewards agents when their Q values outperform those of their best neighbors, encouraging exploration. 2) A pessimism penalty based on the KL divergence from the worst-performing neighbors, discouraging imitation of poor strategies. and 3) A phase-based schedule controls the relative weight of optimism and pessimism across training, while a Bayesian stability regularizer mitigates instability from moving peer references. The approach remains fully decentralized during execution, relying only on local communication during training, which has potential scalablity towards larger number of agents. Empirical results on cooperative predator–prey grid-world benchmarks demonstrate that BEMAS improves both sample efficiency and training stability over IQL and PED-DQN baselines.

### Strengths
1. Originality: This paper introduces an interesting conceptual framing in MARL by explicitly leveraging both the best and worst performing peers through optimism and pessimism shaping. While optimism and pessimism have individually appeared in prior literature, their coupling within a decentralized, proximity-aware reward shaping framework is original. The phase-based scheduling and the integration of a Bayesian stability regularizer add further novelty and theoretical grounding.
2. Quality: The paper is technically solid, with clear mathematical formulations and boundedness proofs for all shaping terms. The authors provide rigorous theoretical analysis ensuring stability and bounded rewards, complemented by ablation studies that systematically isolate the effects of each component.
3. Clarity: The manuscript is well-written, logically organized, and supported by diagrams that effectively communicate the method’s mechanisms. Mathematical notation is consistent, and appendices provide comprehensive proofs and experimental details.
4. Significance: The work makes a potentially valuable contribution to the broader MARL community by proposing a general, decentralized shaping mechanism that enhances both exploration and stability without centralized training, which might inspire new directions in cooperative learning and stability regularization in distributed systems. While the experiments are somewhat narrow in current scope, the conceptual and methodological contributions are potentially significant and broadly applicable.

### Weaknesses
1. The evaluation is restricted to a single class of cooperative grid-world predator and prey environments, which are relatively simple and low-dimensional, which limits confidence in the generality and scalability of BEMAS to more complex or continuous MARL benchmarks. Testing on at least one higher-dimensional or partially observable domain would significantly strengthen the empirical claims.
2. The baselines used IQL and PED-DQN do not represent the state of the art in cooperative MARL in fully decentralized training manner. More recent decentralized or reward-shaping methods such as IPPO, I2Q, or influence-based exploration models) would provide a stronger and fairer comparison. Without these, the performance improvements have risk of overstating the practical advantage of BEMAS.
3. The optimism–pessimism transition is controlled by a manually designed, piecewise-linear schedule. The choice of hyperparameters appears ad hoc and is not empirically justified. A sensitivity analysis or an adaptive scheduling mechanism would help clarify whether performance is robust or heavily dependent on tuning.
4. This approach relies on bounded local message exchange during training, but the paper does not quantify the communication overhead or test robustness under limited or noisy communication.
5. The theoretical analysis focuses on boundedness and stability but does not address convergence properties or potential negative effects of non-stationary peer references. Explicitly discussing when the shaping might mislead learning would make the theoretical contribution more complete.

### Questions
Based on the weaknesses I described above, I have some questions for authors to disscuss.
1. Could the authors clarify how BEMAS would scale to larger or more complex environments, such as continuous control tasks or high-dimensional cooperative benchmarks (e.g., SMAC or MPE)?
2. Are there specific computational or communication bottlenecks that would arise in such settings? Providing either empirical results or scalability analysis could strengthen confidence in generalization.
3. Could the authors provide the performance of IPPO and I2Q algorithms on predator-prey scenario?
4. How would BEMAS behave if the communication radius ρ were smaller or dynamic? Clarifying this could strengthen the decentralization claim.
5. The proofs establish boundedness and stability, but do not address convergence to optimal joint policies. Can the authors discuss whether BEMAS inherits or violates any known convergence properties from standard TD-learning under decentralized settings?
6. Could the optimistic and pessimistic signals ever create conflicting gradients or lead to oscillations? Empirical evidence or theoretical discussion would help.
7. Since the motivation draws from group learning and social dynamics, could the authors comment on whether the BEMAS framework could extend to competitive or mixed-motive environments, beyond purely cooperative ones?
8. Were all methods trained under equal computational budgets such as the same number of updates, training steps, and network capacity? Clarifying fairness of comparison would be helpful.

I look forward to the discussion during the rebuttal period and hope the authors can provide further clarification and evidence to address the raised concerns.

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
5

### Summary
The paper proposes BEMAS, a decentralized multi-agent reinforcement learning framework that uses local communication during training to identify best and worst-performing neighbors. It introduces two reward-shaping signals: an "optimism" bonus for outperforming the best neighbor and a "pessimism" penalty for having a policy similar to the worst neighbor. A Bayesian stability regularizer is added to prevent policy divergence. The method is evaluated on a cooperative predator-prey grid-world and is shown to outperform IQL and PED-DQN baselines.

### Strengths
This paper studies an important problem in co-MARL. The idea of identifying the best and worst neighbor is novel.

### Weaknesses
1. Methdological Flaws: Both the curiosity and performance scores are derived from the TD-error magnitude. This creates a circular dependency: the quality of the shaping signal depends on the quality of the learned Q-function, which is itself being trained using the shaping signal. Therefore, the shaped reward can be unbounded and lead to unstable feedback loops.

2. Lack of Theoretical Understanding: The design of optimistic and pessimistic shaping term should be explained further, especially their interpretation when all agents are neighbors. To be specific for example, why we should penalize the similar behavior of one agent to another even though they are in different local states?

3. The empirical results seem inadequate, as they consider only one single simple environment.

4. The presentation can be improved. Figure 1 uses $a$ to represent agents instead of actions as in the main body. Section 4.4 should appear earlier. Equation 4 seems to be wrong.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BEMAS, a method that helps agents learn by looking at the best and worst performers in their local neighbourhood. It gives agents an optimistic bonus for outperforming their best neighbour (which encourages exploration) and a pessimistic penalty that encourages them to act differently than their worst neighbour (which helps them avoid bad strategies). The system is scheduled to focus on optimism early in training and then ramp up the pessimism later. A stability penalty is also added to prevent the agent's strategy from changing too suddenly. The authors prove these reward signals are stable. Experiments on a predator-prey game show BEMAS learns faster and achieves better or more consistent results than other methods. Further tests show that the pessimism signal, combined with the stability penalty, is the main reason for its strong performance late in training.

### Strengths
- Clear design that is easy to implement and reason about
- Fully local messaging during training; no comms at execution
- Boundedness results for each shaping term; shaped returns are well‑posed
- Good ablation narrative (optimism helps early & pessimism + surprise stabilise later)

### Weaknesses
- The optimistic “Q‑gap” compares different agents’ Q‑values in different states, which may be poorly calibrated and noisy
- TD‑error EMAs as curiosity/reliability proxies are not validated; could correlate weakly with true usefulness or competence
- Benchmarks/baselines are limited (single grid world; no QMIX/VDN/MAPPO etc.)
- Limited parameter sensitivity analysis ​
- No study of communication robustness

### Questions
- How do you calibrate cross‑agent Q‑values for the optimism gap? E.g. did you try advantage or z‑score normalisation?
- Any empirical correlation between $\delta$ and actual competence/return?
- Have you considered the sensitivity of results to various key parameters 
- I'd like to see ablations on neighbourhood radius  and on message dropouts
- Can you add at least one standard MARL benchmark and baselines like QMIX/VDN or MAPPO?

### Soundness
3

### Presentation
3

### Contribution
3
