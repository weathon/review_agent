# ENTER THE VOID: EXPLORING WITH HIGH ENTROPY PLANS

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Model-based reinforcement learning (MBRL) offers an intuitive way to increase the sample efficiency of model-free RL methods by simultaneously training a world model that learns to predict the future. These models constitute the large majority of training compute and time and they are subsequently used to train actors entirely in simulation, but once this is done they are quickly discarded. We show in this work that utilising these models at inference time can significantly boost sample efficiency. We propose a novel approach that anticipates and actively seeks out informative states using the world model’s short-horizon latent predictions, offering a principled alternative to traditional curiosity-driven methods that chase outdated estimates of high uncertainty states. While many model predictive control (MPC) based methods offer similar alternatives, they typically lack commitment, synthesising multiple multi-step plans at every step. To mitigate this, we present a hierarchical planner that dynamically decides when to replan, planning horizon length, and the commitment to searching entropy. While our method can theoretically be applied to any model that trains its own actors with solely model generated data, we have applied it to Dreamer to illustrate the concept. Our method finishes MiniWorld's procedurally generated mazes 50\% faster than base Dreamer at convergence and in only 60\% of the environment steps that base Dreamer's policy needs; it displays reasoned exploratory behaviour in Crafter, achieves the same reward as base Dreamer in a third of the steps; planning tends to improve sample efficieny on DeepMind Control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose to train Dreamer(/like) world-model-based-agents by acting with an exploration policy that is inferred in real time by planning with the world model.
The main contribution is the usage of latent-transition-model entropy to direct the policy towards novel states.

### Strengths
The methods seems rather simple.

Presentation is rather clear.

I like the direction that makes the planner adaptive, deciding when to spend compute for replanning, and when to simply follow the existing plan.

### Weaknesses
Soundness:
1. It is not clear to me that the method is sound. In stochastic environments, maximizing the sum of individual transition entropies amounts to searching for the most noisy trajectories, which is opposed to exploration, which should look for novel transitions. The authors acknowledge this in lines 176-180. 
2. In det. environments, the entropy of the learned transition model can be used as a pseudo-metric for the novelty of the state. If we limit this method to det. environments, it's not clear to me why we should take the entropy instead of the variance, where the variance directly produces theoretically motivated and well investigated UCB exploration (see [1]).
2. Even in det. envs., It's not clear to me why the entropy of the learned transition model should be a better metric for novelty than the disagreement of the model ensemble (i.e. plan2explore). This seems to me to suggest problems in the evaluation setup. Can the authors explain why plan2explore can be expected to fail in maze? Isnt it essentially the same method, only operating based on a theoretically sound metric for novelty (ie a metric that can seperate aleatoric from epistemic uncertainty)?
3. The method seems unnecessarily myopic. Why not learn the intrinsic-reward (the entropy in this case) in a value function, train a policy with respect to this value function, and produce an agent that is able to explore in the direction of novel states, even if these states are not reached in the rollout-length? This is a rather standard method, which again amounts simply to UCB-exploration (see [1] for more details).

Empirical evaluation:
1. A very small number of seeds is used in evaluation, resulting in no statistical significance of the results in most enivornments.
2. Stat. signif. is measured as STD. I don't think STD is the right metric, as it relates to the deviation between the seeds, not the deviation of the mean (ie stat. signif. of the performance gain). I would suggest 2 SEM (i.e. ~95% Gaussian CI).
3. Missing baselines: [1] provides a structured and better theoretically motivated method for planning-for-exploration in MBRL, which is applicable to Dreamers. With the same entropy based estimates for epistemic uncertainty, is there any reason to use the planner proposed in this paper over [1]?
4. Presentation: I would suggest changing the legends to the following: *PPO*, *Dreamer* (previously "no-plan"?), *Dreamer + Our planner* (previously "using plan"), *Dreamer + Plan2Explore* (or P2E for short) for clarity, to make it easier to follow from a glance what is actually compared.

[1] Oren, Yaniv, et al. "Epistemic Monte Carlo Tree Search." ICLR 2025.

### Questions
Comments:
1. I dont understand (/agree with) the seperation of intrinsic reward into "retrospective and anticipatory" methods. If we have access to the model, the one "correct" (or "complete) way to take intrinsic-reward into account is by learning the intrinsic reward inside a value function (retrospective?) and using planning / search to combine the value predictions and the intrinsic reward predictions (anticipatory?) to find an even-better-for-exploration action. If we don't learn the reward in a value function, it seems to me we are myopic - ie suboptimal - for no reason. If we don't have access to the model, we can only use the value function that learned the intrinsic reward. So isn't the better seperation between model-based and model-free methods (i.e. planning vs. no-planning?).
2. line 088: MCTS has been extended to stochastic and cont. envs. (sampled MZ and stochastic MZ [1,2]).
3. Missing reference for PPO (line 049).
4. Line 096 towards -> toward.

[1] Hubert, Thomas, et al. "Learning and planning in complex action spaces." International Conference on Machine Learning. PMLR, 2021.

[2] Antonoglou, Ioannis, et al. "Planning in stochastic environments with a learned model." International Conference on Learning Representations. 2021.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an extension of the Dreamer model-based RL method to improve exploration of rarely visited states. The authors propose to generate $C$ standard (i.e. greedy) trajectories and then use the ones with the highest entropy of the transition predictor (prior) for training the policy. They also propose another meta-controller, trained to maximize both reward and entropy, that decides randomly whether this process shall be used or not. Experiments on DMC show in 2 out of 6 environment a significant advantage, but on 9 Crafter environments there is no significant improvement observable.

### Strengths
The topic is very interesting. Last ICLR another paper [1] showed impressive improvements in exploration when AlphaZero plans optimistically w.r.t. epistemic uncertainty of the involved neural networks, but only showed minor to no improvement for learned models like MuZero (in the appendix). Better exploration with a Dreamer algorithm (which learns the model) would be of great interest to the community, and the presented idea of using internal prior entropy as a stand-in of epistemic uncertainty is (to the best of my knowledge) novel and interesting.

**References**

[1] Oren et al. (2025). "Epistemic Monte Carlo Tree Search". Proceedings of the International Conference on Learning Representations. URL https://arxiv.org/abs/2210.13455

### Weaknesses
I recommend to reject the paper in its current form, because (i) incomplete or missing formalism, description and intuition, (ii) insufficient distinction between aleatoric and epistemic uncertainty, (iii) insufficient reasoning why the prior neural network should produce reliable entropy estimates for unseen states, and (iv) results that are inconclusive. In detail:

1. I have read section 3 and 4 multiple times, and I am still not sure what the authors exactly propose to do. This is partly because the formalism is incomplete. Section 3 needs a clear introduction of MDP and POMDP. Much of the math in Section 4 is incomprehensible (at least to me). For example: I assume the idea is that IG in Equation 3 is minimized by RSSM training (as the "KL divergence loss can also be interpreted as the model's information gain"). Large prior entropy means large prior standard deviations $\sigma_p$ (for Gaussians), leading to *smaller* IG. But how does choosing trajectories (or more precisely random outcomes in the trajectories) that artificially lower the IG incentivize the agent to explore less visited states in the environment? What is the max operation in Equation 5 over? To "choose the [trajectory] with the highest entropy"? This biases the imagined environment, but how does it affect the choice of actions in the environment towards epistemic uncertainty? 
Another example of unclear formalism is the following discussion on $\hat p$, which conditions on different terms every time it is mentioned. What is this distribution? A thought experiment or something that is learned somewhere? What are the states $s_t$ that is reasoned over, the true environment states? But those do not depend on $h_{0:t}$. Finally, I did not understand why a meta-planer is even necessary. Assuming that the authors reasoning on lowering the IG in correct, why stochastically do it only every now and then? Why not do it all the time (called MPC style later)?
2. In the second half of Section 4.1 the authors seem to develop an argument  when the prior entropy is a good stand-in for epistemic uncertainty (I believe) and when not. However, I find it hard to understand why this should be the case in the first place. As the authors note, high prior entropy will *after sufficient training* on a data set correspond to high aleatoric uncertainty. But why should it be related to epistemic uncertainty about the agent's knowledge of the environment? This is what exploration (in the environment) should be based on, as only epistemic uncertainty is reducible with more samples. Or did I misunderstand what the authors call "exploration"?
3. All presented arguments only hold if the neural network produces the "correct" prior entropy. This will only be the case in or near the training set. However, exploration is about finding states that have *not* been explored, that is, which are not in (or even near) the training set. So why do the authors expect the prior entropy to be a good exploration signal? The paper misses a discussion on this.
4. While there are a lot of experiments, and the discussion of the results often contextualizes them correctly ("gains are modest but consistent"), it must be observed that only in ```walker_walk``` and ```hopper_hop``` the proposed methods shows a significant advantage over the "no plan" baseline.  In particular the ```Crafter``` experiments do not show **any significant** improvement. Given that the paper's motivation is somewhat hard to follow (at least for me), this is not a sufficient level of evidence to claim that the proposed method has any systematic effect whatsoever. Without clearly separated standard deviations (or a more rigorous measure of statistical significance), none of the paper's claims can be relied on.

### Questions
- See the questions above.
- What does the "state" in Equations 4 and 5 refer to? The input of the entropy $h_t$?

### Soundness
2

### Presentation
1

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
Model features an exploration method for model-based reinforcement learning from high-entropy exploration. Authors propose to focus on exploring the states with high-entropy stochastic part in learned state representation. To do so, authors use a train-time planing: 
- In planing stage, algorithm generates N imaginary trajectories using low-level actor (that is  learned as in vanilla Dreamer V3), chooses one with highest cumulative entropy over learned hidden state representation's stochastic parts.
- Throughout episode, algorithm uses high-level PPO-based agent which decides wether we should change plan at current step or continue execution.

### Strengths
- Paper proposes a novel exploration algorithm for model-based reinforcement learning, applying it to Dreamer V3. Proposed algorithm generalizes well as long as learned world model uses RSSM as backbone
- Aside from proposition of exploration target, paper proposes usage of PPO-based high-level trainable controller that dynamically decides when to change current plan during training and replanning based on imaginary trajectories that generated with low-level RL agent. While planning is not novel for MBRL itself, such hybrid approach may have high potential in future research and can be applied to other exploration techniques in model-based setup.

### Weaknesses
- Weak experimental base. In past years several model-based exploration techniques were published (i.e. [1](https://proceedings.neurips.cc/paper/2021/hash/cc4af25fa9d2d5c953496579b75f6f6c-Abstract.html), [2](https://arxiv.org/pdf/2310.07220), [3](https://arxiv.org/pdf/2112.01195)) that utilize Dreamer as the backbone. It's hard to understand how well proposed method works without direct comparison.

### Questions
Could you please clarify your argumentation for choosing entropy of stochastic part of learned state representation as the main exploration objective? It is not intuitively understandable how exploration of states with high-entropy stochastic part benefits exploration. As written in paper (lines 176-182), high-entropy stochastic part is usually present for states with high transition function uncertainty regardless of how well state is already explored (which is understandable as the main purpose of stochastic state part in Dreamer is to represent possible uncertainty in transition function), and it's unclear that such high-entropy stochastic part will be present for not explored states (especially with low uncertainty). That's why entropy-based exploration usually target to maximize entropy over states visited during episode: $H(s | \pi)$ rather than uncertainty in transition function.

### Soundness
2

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
The paper proposes an inference time planning layer for model based reinforcement learning that exploits the world model beyond training to drive purposeful exploration. Building on DreamerV3, the authors generate short horizon imagined rollouts from the current latent state and select the plan whose latent prior has the highest cumulative entropy, which they argue anticipates informative states rather than rewarding novelty only after the fact. A lightweight reactive hierarchical controller trained with PPO learns when to commit to a chosen plan versus replan, with a shaped meta reward that combines environment return and latent prior entropy over a window. The paper motivates entropy seeking via an information gain view of the Dreamer KL objective, discusses pitfalls from aleatoric uncertainty and hidden epistemic uncertainty, and proposes practical mitigations by conditioning candidate plans on a greedy actor before ranking by entropy. Experiments on MiniWorld mazes, Crafter, and DeepMind Control show faster exploration, improved sample efficiency, and equal or better final performance compared to Dreamer without planning and a Plan2Explore baseline. An ablation that replans every step demonstrates that commitment is important and outperforms myopic MPC style replanning.

### Strengths
- Clear conceptual framing that links latent prior entropy to information gain in Dreamer and uses it to steer exploration proactively rather than retroactively.
- Practical, model agnostic planner that can wrap existing Dreamer style agents without retraining the actor or replacing the backbone. The gating policy is simple and the squared threshold trick reduces excessive replanning.
- Balanced objective at the meta level that encourages both task progress and coverage of uncertain latent regions, leading to reasoned rather than random exploration.
- Experiments across three regimes with distinct challenges. Results report five seeds on MiniWorld and DMC and three on Crafter, include a strong model based baseline in Plan2Explore, and show consistent gains in sample efficiency. The ablation demonstrates that commitment matters compared to myopic MPC style replanning.
- Implementation details such as inputs to the meta controller, sequence length for meta advantages, and seeds are documented, and a reproducibility statement is provided.

### Weaknesses
- The information gain argument motivates seeking high prior entropy, but the claimed min max coupling of world model learning and exploration is more of an interpretation than a concrete change to the Dreamer objective. The method modifies data collection and planning, not the training loss, and this distinction should be made explicit.
- Prior entropy conflates epistemic and aleatoric uncertainty. The paper mitigates this by conditioning on greedy actor proposals and considering reward, yet a direct comparison to ensemble disagreement or epistemic proxies at equal compute would strengthen the case.
- Compute overhead at inference time is nontrivial due to generating 64 imagined rollouts per step and training a PPO meta policy. The wall clock budget is matched for Crafter and DMC, but there is limited profiling of per step latency and throughput relative to Dreamer alone.
- Omission of tuned model free pixel baselines is understandable, yet it leaves open how the proposed planner compares when the best DrQ or SAC variants are properly tuned under the same wall clock and seed budgets.
- Some design choices appear sensitive and are not fully explored, such as the rollout horizon, the number of candidates, and the weight implicit in the meta reward that balances return and entropy. A small sweep or sensitivity analysis would increase confidence in robustness.
- The conclusion mentions instabilities from inflating the KL objective and recommends reinforcing the model, but the main text does not detail the exact KL weighting schedule or regularization that was used. This missing detail hinders reproducibility.

### Questions
- How sensitive are results to the candidate count N and the imagined horizon H used for entropy accumulation, both in terms of performance and wall clock cost per environment step?
- Can you clarify whether any explicit change in the Dreamer KL coefficient or loss weighting was used in practice? The conclusion mentions inflating the KL objective. Please specify schedules and values if applicable.
- How does the method compare to Plan2Explore when matching compute at a finer granularity, for example the same number of imagined rollouts per step and the same world model updates? If ensembles are permitted for Plan2Explore under the same wall clock, do your gains persist?
- How robust is the method to model bias early in training when the world model is inaccurate? For example, does the entropy signal overprioritize parts of the latent space that are falsely uncertain due to poor reconstructions?

### Soundness
3

### Presentation
3

### Contribution
3
