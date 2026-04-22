# One Policy Learns Them All: Synergizing  Prior-Guided Exploitation and Online Exploration in Curriculum Based MARL

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Existing offline-to-online (O2O) multi-agent reinforcement learning (MARL) methods typically employ offline prior policies for warm-start initialization but are susceptible to distributional shifts and structural consistency constraints.
On the other hand, prior-guided cold-start conditions, albeit of more practical interests, require a subtle synergy between utilizing prior-collected samples and self-exploring the state-action space.
In this paper, we propose DUCE, a dual-track curriculum MARL algorithm that balances exploitation and exploration to ensure efficient, stable cold-start training. 
The curriculum designs include:
(1) an externally configured task-difficulty curriculum that alternates between performing prior and online policies with probabilistic scheduling, progressively reducing the prior-guidance horizon to transition tasks from easy to hard, and
(2) an internally evolving policy optimization curriculum that imposes a decaying offline RL regularizer on the online loss, enabling a smooth shift from conservative prior reliance to exploration-driven training.
Extensive experiments on challenging StarCraft multi-agent challenge (SMAC) v1/v2 tasks demonstrate that DUCE achieves faster convergence and higher asymptotic performance, and consistently outperforms state-of-the-art warm-start baselines.
Importantly, DUCE is agnostic to the architectures of priors (e.g., rule-based or RNN).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The main idea of the paper is combining some approaches from Offline Multi-Agent Reinforcement Learning (primarily CFCQL) and guided policy learning (primarily JSRL) to develop a more sample efficient MARL framework in an online setting, where the policy is guided using both a regularization loss based on an expert policy and jump-starting from intermediate states obtained by the same expert policy to improve exploration. Both these guidances are scheduled to decrease over the course of training.

I would summarize the main contributions compared to prior work as:
1. Applying the previously suggested JSRL in a multi-agent setting.
2. Specifically, the combination of JSRL and the previously suggested CFCQL.
3. Using a linear schedule for previously constant weight of the regularization term in the existing CFCQL loss for offline-MARL.
4. Introducing a probability of guidance p_guide in the JSRL algorithm, which at the start of an episode determines whether guidance shall be used at all. The default for "vanilla" JSRL would be p_guide=1.

### Strengths
The paper presents a novel framework for prior-guided multi-agent reinforcement learning which combines an expert policy both to regularize the learned policy towards a desired behavior and help learning by jump starting trajectories. The combination of these methods is well motivated and the proposed scheduling reasonable. The overall motivation is sound and the research topic of interest to the RL community.

### Weaknesses
## Regarding main contributions

The abstract and sections 1, 6, A4 claim higher asymptotic performance of the proposed method compared to SOTA methods. However, figures 4 and 10 show non-convergence of the performance of other methods such as ft-QMIX.

The application of JSRL in a multi-agent setting, even in the specific context of SMAC, is not novel, see "SC-MAIRL: Semi-Centralized Multi-Agent Imitation Reinforcement Learning", Brackett et al. This literature is missing.

Section 6.3 claims for p_guide that "When the values exceed 0.7, substantial performance fluctuations emerge during the early training phase." However, in Figure 6, the blue line for p_guide=1 is at all stages of training the best or close-to-best performing variant. Since p_guide=1 is from my understanding the base setting for JSRL, I do not see where it is shown that this contribution is beneficial. Furthermore, the JSRL paper also contains an ablation called "JSRL-Random Switching", where the guidance length is randomly varied and not monotonically decreasing, which conceptually sound similar to the p_guide introduced here.

From my understanding, the primary difference to the standard CFCQL loss is linearly scheduling the weight alpha of the regularization, which is constant in CFCQL. I consider both the design choice as a constant in an offline setting for the original CFCQL algorithm and decaying it in an online setting as done in the proposed paper well motivated. Non-constant weighting of a regularizing loss based on for example an expert policy in an offline-to-online setting is however not a novel contribution, see as one example "Adaptive Behavior Cloning Regularization for Stable Offline-To-Online RL", Zhao et al.
Given that introducing a schedule to a regularization loss is of limited novelty, and is only novel in the context of CFCQL, I would have expected a strong and convincing evaluation of a variety of schedules. However, the scheduling is only evaluated in one scenario in figure 8, only the parameters alpha_start and T_alpha are varied and no comparison is made to other schedules, despite hinting at different schedules in the appendix. Furthermore, the authors themselves write that the length of the decay period T_alpha has limited impact on the overall training efficiency and that the choice of alpha_start has limited influence on the overall training efficiency (both section 6.4). This raises the question how important this contribution is.

In general, the evaluation is performed on different scenarios within the SMAC environment. However, I consider this somewhat limited. In particular, the authors only evaluate on discrete action spaces. Previous work such as CFCQL has used SMAC, MPE and MA-MuJoCo in their evaluations.

## Regarding presentation and notation

In section 3, the definition of domain of pi is given as o_t x a_t. However, o_t and a_t are not defined.

Observation II claims that Fig. 2c shows Q_tot of JSRL-CFCQL to "exhibit sharp early fluctuations before stabilizing", but JSRL-QMIX has much larger shaded area whereas JSRL-CFCQL is as narrow as ft-QMIX?

All regarding Equation (1), which is the core loss introduced by the authors:
It is not explained what a_-i notation represents. From CFCQL I expect this to be the joint action without the action of agent i, but this should be clearly expressed.
$E_{a \sim \eta}[Q(s,a)]$, a should likely be bold because I have interpreted it to be a joint action.
The notation $E_{\pi_i} (\pi_i(s)/\eta_i(s))$ is unclear. My best guess is $E_{a \sim \pi_i(\cdot|s)} (\pi_i(a|s) / \eta_i(a|s))$. If this is correct, there are several issues which I do not see adressed:
First of all, since samples are drawn from $\pi$, but division happens through probability of $\eta$, if any sample is never selected by $\eta$, this expected value is not defined. Secondly, without further tricks, I would expect that one must be able to evaluate $\eta(a|s)$. Being able to calculate $\eta(a|s)$ is a stronger requirement than being able to sample $a \sim \eta(\cdot|s)$, the general requirement for a (black box) stochastic policy. Yet, the paper claims that it addresses "prior policies without assumptions on their form or structure". Maybe this statement should be relaxed, or the authors should clarify why this is not a concern. I am aware the CFCQL mitigates this by training a VAE to mimic the behavior policy, but whatever is done by the authors should be explicitly stated.

### Questions
I would ask the authors to clarify my questions also mentioned in Weaknesses regarding handling behavior policies for which a probability density function is not straightforward to calculate. I would appreciate if they could further clarify the points regarding notation. 

*Suggestions regarding current weaknesses*:

I recommend repeating the main experiments with higher step number such that performance of all methods saturates (or deteriorates if this happens) or removing the claim regarding better asymptotic performance.

I suggest citing the work mentioned in Weaknesses regarding JSRL in a MARL context and discussing the differences to this work.

It would be interesting to further analyze how similar the previously suggested JSRL-Random Switching curriculum is to the proposed p_guide<1, since both introduce random ordering and in general extend the analysis of this contribution.

Similarly, I would recommend extending the analysis on different schedules for the regularization loss, especially compared to a variety of constant factors, and clearly showing the benefit on a variety of tasks as this is a claimed contribution.

Given the broad claims of applicability, especially that the proposed framework is defined on arbitrary action spaces and not just discrete ones, I would suggest further analysis to its benefits on continuous action spaces, or since this may be difficult within the time frame, clarify the limitations in evaluation.

### Soundness
1

### Presentation
2

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
This paper proposes an offline-to-online MARL approach, involving two types of learning curricula run in parallel: one external, changing the difficulty of tasks as represented by the length of required planning by the learned policy; the other, by gradually shifting the training loss from offline-RL loss to purely online loss. 

The paper demonstrates that the combination of these two strategies produces policies which outperform strong baselines on challenging multi-agent coordination tasks.

### Strengths
As far as I am aware, this paper is indeed an initial study into cold-start offline MARL algorithms, and the contributions of experimenting with diverse sources of prior policies are indeed novel. 

The contribution of the paper is well motivated and justified. In particular, I liked section 4, where the authors present qualitative results supporting the motivation for their proposed method. 

Experiments are comprehensive and include reasonable ablations. Results on the challenging SMAC tasks seem to outperform strong baselines.

### Weaknesses
Some minor concerns and questions arise when reading the paper. 

Line 309: “To strengthen these warm-start baselines, their online training is allowed to leverage offline data, with 30% of training samples drawn from the offline buffer”: Does this actually strengthen the baseline, or rather induce detrimental performance since this data is from a different distribution to the policy? It might be helpful to consider techniques for including offline prior data in RL training for this type of baseline (eg. RLPD). 

Nit picks: 

- Line 348: text seems cut (missing start of sentence)
- A slight concern on the side of novelty is that the scheduled weighting of the QMIX and CFCQL losses should probably be considered a scheduled hyperparameter adjustment, rather than a curriculum.

Other concerns follow in the questions section below. If these concerns and questions are answered adequately during the rebuttal period, I am happy to raise my score accordingly.

### Questions
1. Section 3 (preliminaries) - as suggested by the reward definition, and the definitions of Q functions in later sections - does the algorithm proposed in this paper only deal with reward shared among all agents (i.e. cooperative scenarios only)? 
2. What are the prior policies used for the variants of JSRL?
3. Why are only sub-optimal policies considered? Why not work with expert policies, if they are available?
4. In eq. 1, is the Q function in the CFCQL loss a single-agent Q function or the total Q function for all agents, similar to $Q_{tot}$? If it’s the former, doesn’t the mixing between these two losses cause issues with scaling of gradients and stability of training?
5. Are the prior policies used for guidance in the external curriculum identical to the ones used as behavior policies for the CFCQL loss?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a framework for training multi-agent RL problems when making use of an informative prior policy. This prior is not necessarily optimal, but may provide structured guidance that assists in exploration. The proposed method, DUCE, does not use warm-start techniques and instead initializes from a random policy. DUCE operates by building off the jump-start RL framework, and starts episodes by first following the trajectory defined by a prior guidance policy. A curriculum is established where the guidance policy is replaced by the live policy at increasingly early intervals. At the same time, an "offline RL" auxiliary loss is computed on a fixed dataset, and the weighting of this loss is decreased over the training interval. Experiments are conduced over the SMAC domain, and DUCE is shown to outperform the main baselines.

### Strengths
The ideas presented in this paper are justified and empirically show an improvement. The paper provides numerical analysis on the pitfalls of naive baselines, including reproducing the "warm-start collapse" phenomenon where warm start policies show an initial drop in performance. Over the standard benchmarks, the proposed method shows a consistent improvement. Ablations are presented on the two core proposals (prior guidance and offline RL).

### Weaknesses
- The proposed method introduced significant complexity. In general it is undesirable to introduce new hyperparameters, and the proposed method can be seen as a "bag of tricks" combining JSRL and the offline CFCQL, along with a tuned curriculum that modulates their effects over training. The resulting method does not appear immediately practical as it introduces additional complexity that must be tuned.
- The paper's novelty is not immediately apparent. The two adjustments presented have been detailed in previous works, and the paper does not present any concrete analysis on the *cross-interaction* of these two adjustments.
- The loss function in equation 1 combines two styles of TD loss. This could be strengthened by an ablation to just use the same TD loss (but over an offline vs. online state distribution).
- It is unclear how much of the insights in this paper are specific to multi-agent RL, vs. offline-to-online RL methodology in general. The proposed algorithm does not include any specific multi-agent components. To strengthen the multi-agent specificity, it may be beneficial to highlight performance on single-agent tasks as well, and/or show explicit analysis that specific phenomena only appear in the multi-agent setting.

### Questions
- Do the curves in Fig 1 come from real data? I would be worried that the trends presented in the stylized graph are not neccessarily reflected in a real experiment, or are domain dependent.
- How much does the gain from DUCE depend on the specific prior policy?
- On page 5, it is stated that h_0 is set to the average steps required by the prior policy to solve the task. However, it is later states that the prior policies are designed to be suboptimal and have a < 100% success rate. In that case, how is h0 selected?
- In equation 1, there are three Q networks referenced -- Q, Q_tot, and Qbar. Qbar is defined as the target network, but what is the difference between Qtot and Q?
- In equation 1, how is pi(s) calculated, when pi is originally defined as pi(a|s)?
- Is there a way to measure the "diversity" of states visited by the guiding policy vs. the cold start policy? This form of analysis may improve the paper to describe concrete phenomena rather than potentially domain-specific or black-box proposed improvements.

Minor:
- It is unclear/undefined what "guidance step size" in the top of page 5 means.
- The paper would be strengthened by an explicit explanation of what JSRL and QMIX do.
- An explicit algorithm box would improve clarity, including which hyperparameters need to be tuned per-task.

### Soundness
2

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
2

### Summary
This paper proposes DUCE, which is curriculum on both task difficulty and policy optimization to enable a cold-start training in MARL.  In particular, DUCE utilizes similar curriculum on task difficulty inspired by JSRL and improves it to be adjusted by online performance feedback. Besides, DUCE also utilizes CQL loss to mitigate distribution shift and proposes to linearly decay the CQL loss with time $t$ for better balances between exploitation and exploration.

### Strengths
In spired from JSRL [1] for SARL, this paper proposes a cold-start O2O MARL algorithm, which is different from previous hot-start MARL methods.

This paper proposes a curriculum loss, which gradually degrades the weight of offline loss beyond JSRL.

The performance of the proposed method is much better than baselines on training efficiency and final results.

### Weaknesses
### Here are some concerns that may limit the contribution.
### 1. Confusion about Section 4 or Figure 2

1. What are the guidance policy and exploration policy in JSRL? It is important to show:
   1. The performance of the guidance policy $\pi^g$, which is trained well in offline dataset.
   2. The performance of the exploration policy $\pi^e$.
   3. The performance of the combined policy $\pi$.
2. What is the strategy for guide-step sequences, curriculum, and random-switching? 
   1. Better to show both fails in MARL.
3. It is better to show that the guidance policy is significantly better than the exploration policy at the early stage, which is the central claim of JSRL.
4. Since JSRL originally belongs to single-agent RL, it is better to test how much benefit JSRL can provide to "6h vs 8z medium" by centralizing the agents to be single-agent before comparing with MARL.

### 2. Curriculum on task difficulty

1. The curriculum method for determining guide-step sequences is translated from JSRL, while the improvement of this method is insufficient. Here is the reason:
   1. The curriculum method in JSRL can be naturally applied in MARL.
   2. The only difference is that JSRL degrades the guide step based on a fixed rule, while this paper is based on experience feedback.
   3. The curriculum method in this paper doesn't consider any multi-agent property, like
      1. different agents may be suitable for different strategies for determining guide-step sequences.
      2. different agents may be suitable for different curriculum parameters.

### 3. Curriculum on offline loss

1. Utilizing offline CFCQL loss to mitigate the unlearning problem is first proposed in OVMSE [2].
2. The description from line 307 to line 309 seems to have a problem:
   1. Of course, OVMSE applies offline CFCQL loss in Offline Value Memory (OVM) for $Q_{OVM}$.
   2. However, $Q_{OVM}$ is applied to update $Q_{tot}$ partially weighted by $\lambda_{memory}$.
   3. Finally, the agent is updated via the joint action-value function $Q_{tot}$ in ft-QMIX.
   4. Thus, the policy is also influenced by offline loss, which is a similar idea in this paper. 

3. Besides, OVMSE also gradually degrades the offline loss weight by annealing scheduling $\lambda_{memory}$, which is similar to the curriculum design on policy loss in this paper.
### 4. Baselines

The selected baselines are all RL-based methods. However, there is also another type of baseline on the generative model, such as Decision Transformer [3] (DT). This kind of method allows offline pretraining and online finetuning, which is O2O. Please compare with the work SO2-MADT [4], which is also an O2O work on MARL with code provided in the paper. 

### 5. Scenario Selection

While the scenarios in SMAC v1 are comprehensive, the scenarios in SMAC v2 are similar. Is there any standard for the scenario selection beyond "widely recognized"? Could you please provide diverse tasks, such as different and asymmetric numbers of agents?

[1] Uchendu, I., Xiao, T., Lu, Y., Zhu, B., Yan, M., Simon, J., ... & Hausman, K. (2023, July). Jump-start reinforcement learning. In *International Conference on Machine Learning* (pp. 34556-34583). PMLR.

[2] Zhong, H., Wang, X., Li, Z., & Huang, L. (2024). Offline-to-Online Multi-Agent Reinforcement Learning with Offline Value Function Memory and Sequential Exploration. *arXiv preprint arXiv:2410.19450*.

[3] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, *34*, 15084-15097.

[4] Shah, A. B., Wen, Y., Chen, J., Wu, X., & Fu, X. (2024, October). Safe Offline-to-Online Multi-Agent Decision Transformer: A Safety Conscious Sequence Modeling Approach. In *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 12400-12407). IEEE.

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
