# Robust Adversarial Policy Optimization Under Dynamics Uncertainty

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Reinforcement learning (RL) policies often fail under dynamics that differ from training, a gap not fully addressed by domain randomization or existing adversarial RL methods. Distributionally robust RL provides a formal remedy but still relies on surrogate adversaries to approximate intractable primal problems, leaving blind spots that potentially cause instability and over-conservatism.
We propose a dual formulation that directly exposes the robustness–performance trade-off. At the trajectory level, a temperature parameter from the dual is approximated with an adversarial network, yielding efficient and stable worst-case rollouts within a divergence bound. At the model level, we employ Boltzmann reweighting over dynamics ensembles, focusing on more adverse environments to the current policy rather than uniform sampling. Two components act independently and complement each other: trajectory-level steering ensures robust rollouts, while model-level sampling provides policy-sensitive coverage of adverse dynamics.
The resulting framework, robust adversarial policy optimization (RAPO) outperforms robust RL baselines, improving resilience to uncertainty and generalization to out-of-distribution dynamics while maintaining dual tractability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes RAPO (Robust Adversarial Policy Optimization), a dual-based framework for robust reinforcement learning under dynamics uncertainty. Starting from a KL-constrained robust MDP, the authors derive a dual formulation introducing a temperature variable that balances performance and robustness. RAPO uses two complementary mechanisms: an adversarial network (AdvNet) to amortize the dual variable estimation at the trajectory level, and Boltzmann reweighting across an ensemble of dynamics models for model-level robustness. Experiments on Walker2d and a quadrotor payload task show that RAPO improves out-of-distribution robustness compared to prior works while maintaining in-distribution performance.

### Strengths
- The paper provides a well-motivated unification of distributional robustness (dual formulation) and adversarial training through trajectory-level and model-level mechanisms, both controlled by KL budgets.
- The dual-level design (AdvNet + Boltzmann reweighting) is intuitive and provides a fine-grained control between local (trajectory) and global (model) robustness.
- The paper is clearly structured, with theory, algorithms, and experiments aligned. The authors articulate trade-offs (e.g., robustness vs. performance degradation) clearly and transparently.

### Weaknesses
- The experiments primarily assess robustness to environmental parameter shifts (e.g., mass, friction, inertia) rather than active adversarial perturbations. Since RAPO explicitly claims adversarial robustness, it would be valuable to include tests under learned or adaptive adversarial agents, as used in RARL [1], QARL [2], ROSE [3], to evaluate resilience against deliberate attacks.
- The latest robust RL baseline compared is RARL (2017), while several more recent algorithms [2, 3, 4] provide stronger and more diverse perspectives on robustness. Without these comparisons, it is difficult to judge whether RAPO represents a genuine advance over current state-of-the-art.
- Figures 1 and 2 could benefit from larger fonts for readability. 
- Conceptually, RAPO can be seen as combining two existing ideas: distributional robustness via dual tilting and weighted domain randomization within a PPO loop. While this integration is elegant, it may be viewed as incremental unless stronger empirical gains or theoretical guarantees are provided over existing robust RL frameworks.
- The dual formulation is motivated by avoiding the limitations of sample-based adversarial training, where finite sampling may fail to cover the entire ambiguity set, leaving blind spots or inducing over-conservatism. However, recent methods such as ROSE [3] tackle the same issue using Stein variational policy gradients to approximate the worst-case distribution. A comparison or discussion of how RAPO’s dual approach differs from or improves upon these variational methods would strengthen the related work and clarify the novelty claim.



#### [1] Lerrel Pinto et al., Robust Adversarial Reinforcement Learning, ICML 2017
#### [2] Aryaman Reddi et al., Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula, ICLR 2024
#### [3] Juncheng Dong et al., Variational Adversarial Training Towards Policies with Improved Robustness, AISTATS 2024
#### [4] Takumi Tanabe et al., Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification, NeurIPS 2022

### Questions
- The dual decomposition and Boltzmann reweighting are both well-established concepts. Could the authors highlight what specific theoretical or algorithmic innovation is unique to RAPO beyond integrating these elements within PPO?

### Soundness
3

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
4

### Summary
This work introduced robust adversarial policy optimization (RAPO) which solve the dual of the robust RL problem that collapse an infinite-dimensional search space down to a scalar within the KL budget, which can help provide sufficient coverage over challenging scenarios and corner cases (i.e., out-of-distribution dynamics), followed by using boltzmann sampling to steer more visitation toward low-return/challenging regions.

### Strengths
* The approach is designed following very clear line of thoughts, e.g., why solving the dual problem and the need for AdvNet and Boltzman sampling.
* Sufficient theoritical analyses were provided to support the design choices and insights toward various properties of the approach, e.g., convergence of the ensemble estimation, the existence of stationary/saddle points and robust value drop bounds.
* Experiments clearly ablates the AdvNet and Boltzman sampling components respectively, which empirically justifies the design of the approach.

### Weaknesses
* From the reviewer's point of view, this work is also closely related to the distributionally robust RL work [1-3 below for example] in general, which shared similar high-level objectives of addressing out-of-distribution scenarios. It could be worthy to discuss the connections and distinctions to that line of work.
  * It would be interesting to compare against some of those works in the experiment as well, if applicable.
* The reviewer appreciated that the authors compared to robust RL baselines including RARL and EPOpt. It would be interesting to see how the approaches that directly solve the primal would contrast. For example, RNAC and Gleave et al. as cited in the paper.
* The robust RL environments the authors used in the experiments are typical, while the reviewer is also curious how this work could also facilitate exploration efficiency in larger state-action spaces (with little-to-mild robustness needed in terms of friction/inertia changes). For example, how RAPO would perfom in higher degree-of-freedom humanoid or hand manipulation (Adroit) environments, or more precise tasks like object picking with robot manipulators.

[1] Ramesh, Shyam Sundhar, et al. "Distributionally robust model-based reinforcement learning with large state spaces." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[2] Shi, Laixi, et al. "The curious price of distributional robustness in reinforcement learning with a generative model." Advances in Neural Information Processing Systems 36 (2023): 79903-79917.

[3] Liu, Zijian, et al. "Distributionally Robust -Learning." International Conference on Machine Learning. PMLR, 2022.

### Questions
One clarification question -- in AdvNet, $m$ samples needed to be obtained for the given $(s,a)$ (page 4 line 201, and Alg. 2 line 1). The reviewer was wondering how this could be done practically in a more realistic setup, e.g., with an actual robotic manipulator does it imply that the approach would need the manipulator to be reset to the same state again and again to obtain $m$ samples of the next states?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses the robustness of reinforcement learning (RL) policies to shifts in dynamics distributions post-training. The proposed method employs a dual formulation of value maximization under distributional uncertainty that reduces distributional sampling to a parametric optimization and a Boltzmann reweighting scheme to prioritize difficult training examples. Experiments support the method's generalization to out-of-distribution dynamics.

### Strengths
- The dual formulation approach is interesting and gives insight into how efficient distributional policies can be trained.
- The paper is organized well.

### Weaknesses
### Motivation
- The motivation of the dual formulation is unclear. The stated reason, that a finite set of training distributions leaves blind spots or is otherwise not comprehensive, is somewhat contradicted by the later inclusion of the ensemble method. If exponential tilting is sufficient to ensure robustness, why is explicit sampling used both in the AdvNet and reweighting components?
- The use of the word "adversarial" is a little confusing, since Section 4.2 states that aleatoric uncertainty is out of scope. Generally, in adversarial RL,  perturbations are assumed to be chosen at test time such that the resulting mistakes minimize the target's reward; this changes the problem from epistemic to aleatoric in that the resulting perturbation is definitively not a natural occurrence. 

### Distinction from prior work
- The stated weakness of existing Monte Carlo methods is that they are unable to fully capture the uncertainty set via sampling. The intuition behind random sampling is that, given an infinite number of samples, the full distribution will be captured, so the problem is not in theory but in practice. With that in mind, the practical aspects of the method (i.e. sample complexity, performance in low-occurrence distributions) should be examined empirically.
- The paper states that $\eta^*$ functions as a robustness-performance knob that is absent in primal formulations, which seems untrue. Many methods in adversarial RL use some variation on the general form $(1-\lambda)V(\cdot) + \lambda V^{adv}(\cdot)$, where $\lambda$ is a robustness temperature. See [1] as an example.

### Experiments
- The paper shows comparisons to two outdated robust baselines, dating from 2016 and 2017. There are many peer-reviewed methods published more recently that would serve as a stronger comparison.

[1] Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane S. Boning, Cho-Jui Hsieh: Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations. NeurIPS 2020

### Questions
- Is there a concrete example of a domain where the $\eta^*$ dual solution captures critical dynamics that discrete sampling would not? Intuitively, it seems that most or all realistic domains are "smooth", i.e. dynamics do not vary wildly across small distances. This seems supported by the ablation in Figure 2; one would expect failures of the naive solution to have a more "jagged" shape were this not the case. 
- What is the distinction of the dual function from prior work? By reading the paper, one can understand that it provides robustness guarantees beyond the finite sample set. Is this proven, and are there other qualities that can be stated?

### Soundness
3

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
The paper proposes a new algorithm RAPO that steers the trajectories towards low return ones, resembling the trajectories drawn from adversarial kernel. This  makes the policy robust.

### Strengths
The idea is good and motivation is intuitive. However my main concern is that the work is not compared with [1] which seems very similar in setting and motivation.

### Weaknesses
The current approach seems very similar to [1] where they also do pessimistic sampling, can authors compare and contrast with [1]. 



[1]@inproceedings{
gadot2024bring,
title={Bring Your Own (Non-Robust) Algorithm to Solve Robust {MDP}s by Estimating The Worst Kernel},
author={Uri Gadot and Kaixin Wang and Navdeep Kumar and Kfir Yehuda Levy and Shie Mannor},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=UqoG0YRfQx}
}

### Questions
Q1) The paper considers sa-rectangular uncertainty sets which are very conservative. Can authors comment if this approach can be extended to s-rectangularn [2] or non-rectangular uncertainty sets [3,4]. 





[2]@inproceedings{
kumar2024efficient,
title={Efficient Value Iteration for s-rectangular Robust Markov Decision Processes},
author={Navdeep Kumar and Kaixin Wang and Kfir Yehuda Levy and Shie Mannor},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=J4LTDgwAZq}
}

[3] @inproceedings{
kumar2025nonrectangular,
title={Non-rectangular Robust {MDP}s with Normed  Uncertainty Sets},
author={Navdeep Kumar and Adarsh Gupta and Maxence Mohamed ELFATIHI and Giorgia Ramponi and Kfir Yehuda Levy and Shie Mannor},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Xx0cJGXU7n}
}

[4]@misc{li2025policygradientalgorithmsrobust,
      title={Policy Gradient Algorithms for Robust MDPs with Non-Rectangular Uncertainty Sets}, 
      author={Mengmeng Li and Daniel Kuhn and Tobias Sutter},
      year={2025},
      eprint={2305.19004},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2305.19004}, 
}

### Soundness
2

### Presentation
3

### Contribution
2
