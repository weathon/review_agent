# Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
This study investigates behavior-targeted attacks on reinforcement learning and their countermeasures. Behavior-targeted attacks aim to manipulate the victim's behavior as desired by the adversary through adversarial interventions in state observations. Existing behavior-targeted attacks have some limitations, such as requiring white-box access to the victim's policy. To address this, we propose a novel attack method using imitation learning from adversarial demonstrations, which works under limited access to the victim's policy and is environment-agnostic. In addition, our theoretical analysis proves that the policy's sensitivity to state changes impacts defense performance, particularly in the early stages of the trajectory. Based on this insight, we propose time-discounted regularization, which enhances robustness against attacks while maintaining task performance. To the best of our knowledge, this is the first defense strategy specifically designed for behavior-targeted attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new state-perturbation attack against a trained RL agent, aiming to manipulate the agent into behaving according to a target policy chosen by the attacker. The main idea is to use imitation learning techniques, such as GAIL, to learn the attack policy from demonstrations of a target policy, so that the attacker does not need to access the agent's policy directly. Further, the paper proposes a new defense adapted from the SA-PPO algorithm, using time-discounted regularization to train a smoothed policy.

### Strengths
This problem of adversarial behavior maniplation through state perturbation has been considered recently. The main contribution of the paper is showing that the attacker's problem can be viewed as a MDP with carefully defined reward and transition functions derived from the agent's MDP, and utilizing behavior imitation to avoid white-box access to agent's policy. Another contribution of the paper is the obsevation that the state changes in the early stages of trajectories have a great influence on the agent's overall performane in continuing tasks and the time-discounted regularization scheme derived from that.

### Weaknesses
1. While the paper does not need white-box access to the agent's policy, it requires demonstrations from the target policy, which might not be easy to get in practice without strong domain knowledge, including the environment and the agent's policy. In particular, behavior manipulation includes reward minimization as a special case, treating a reward minimization policy as the target policy. 
2. Algorithm 1 is a straightforward adaptation of GAIL, and it can be directly derived from the objective (2), without Theorem 5.1 and the discussions in Section 5. Why are those discussions useful? 
3. The defense objective (10) does not make sense to me. From the agent's perspective, it should just try to maximize its own reward. Why should it try to minimize the attacker's gain? This only makes sense in the zero-sum setting. Further, since behavior manipulation includes reward minimization as a special case, the agent should consider the worst-case scenario to achieve robust defense. In that sense, there is no need to distinguish target behavior manipulation and reward minimization from the agent's perspective. 
4. The observation that the agent's policy is more sensitive to early stages of trajectories looks like an easy consequence of using a discount factor in continuing tasks, and this should apply to the reward minimization case as well. However, this result does not apply to episodic tasks and when the discount factor is very close to 1, which are commonly considered in the previous work, such as SA-PPO.  
5. The evaluation is not very convincing. The paper does provide baseline results for other defenses except in the Meta-World environment. Further, it ignores important recent defenses such as:
- Li et al., Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error, ICML 2024.
- Sun et al., Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations, ICLR 2024.
- Yang et al., DMBP: Diffusion Model-Based Predictor for Robust Offline Reinforcement Learning against State Observation Perturbations, ICLR 2024.

### Questions
1. Algorithm 1 is a straightforward adaptation of GAIL, and it can be directly derived from the objective (2), without Theorem 5.1 and the discussions in Section 5. Why are those discussions useful? 
2. Why should the agent consider target behavior manipulation? It seems that the agent should always consider the worst-case scenario to achieve robust defense, and there is no need to distinguish target behavior manipulation and reward minimization from the agent's perspective. 
3. Is time-discounted regularization still useful when gamma is close to 1, as typically considered in previous work?

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
3

### Summary
This paper introduces behavior-targeted attacks, which steer a victim RL agent’s policy toward an adversary-desired behavior rather than simply reducing reward. It proposes (1) Behavior Imitation Attack (BIA) — an imitation-learning–based attack operable in black-box or no-box settings, and (2) Time-Discounted Regularization Training (TDRT) — a defense that suppresses early-trajectory policy sensitivity via time-weighted regularization.

### Strengths
1. Innovative attack under limited access: BIA elegantly converts the problem into an imitation-learning formulation, removing dependence on victim parameters.

2. TDRT’s time-discounted regularization is motivated by a provable bound (Theorem 6.1) and empirically validated.

3. Experiments span multiple continuous-control and grid environments, with clear comparisons against strong baselines (ATLA-PPO, SA-PPO, RAD-PPO, WocaR-PPO).

4. The paper is well-structured, with a precise threat model, proofs in appendices, and reproducibility statements including algorithm pseudocode.

### Weaknesses
1. Diffusion model based defenses[1][2] have been proposed recently to fight against state adversarial perturbations. Could the author compare them with the proposed defense?

2. The proposed method currently do not scale to image based input RL environments such as Atari games.

[1] Z. Yang and Y. Xu. DMBP: Diffusion Model–Based Predictor for Robust Offline Reinforcement Learning against State Observation Perturbations. ICLR, 2024.

[2] X. Sun and Z. Zheng. Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations. ICLR, 2024.

### Questions
1. I would like to see a discussion between the behavior imitation attack and the backdoor attack. This should be an important related work in this work.

2. I would like to see some results on diffusion model-based defenses against behavior imitation attacks.

3. Could the proposed methods extend to a large state space environment, such as images based environment, such as Atari games? I know it is a challenging extension, but do the authors have any potential ideas on this?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors propose behavior imitation attack (BIA), a new class of behavior-targeted attack in DRL (where the adversary's goal is to manipulate the victim policy in such a way that it benefits the adversary instead of pure reward minimization) via adversarial manipulations to state observations. Unlike prior work, BIA leverages imitation learning and replaces the state observations of the victim with the falsified state generated by the adversary policy and works under black-box or no-box settings. The authors also present a defense strategy, Time-Dependent Robust Training (TDRT), against such attacks. TDRT introduces a regularization term to suppress the sensitivity of action outputs to state changes. Experiments on Meta-World, MuJoCo, and MiniGrid show that BIA can effectively manipulate victim policies, and TDRT outperforms existing adversarial training baselines in robustness-performance trade-offs.

I believe that the theoretical insights combined with the novelty is enough for acceptance.

### Strengths
S1. First black-box/no-box behavior-targeted attack: The introduction of BIA allows for attack generation with extremely limited victim access, which aligns with realistic threat models. 

S2. Theoretical analysis: The authors provide a theoretical basis for both the attack and defense strategies proposed in the paper. I also find the motivation behind TDRT simple and elegant. 

S3. Strong empirical evaluation: Although most of the experiments and valuable ablation studies are pushed to the appendix, results are comprehensive and multiple baselines are compared using different environment settings.

### Weaknesses
W1. The defense method is referred to as both Time-Discounted Robust Training (in introduction) and Time-Discounted Regularization Training (in later sections). The naming should be unified throughout the paper.

W2. The proposed attack may not scale to high-dimensional state spaces, which was also admitted by the authors in the limitations section.

### Questions
Q1. What is the computational cost of BIA compared to white-box attack baselines?

Q2. Is TDRT (or policy smoothing) vulnerable to reward minimization attacks? Can we combine adversarial training with policy smoothing to equip agents with a defense that can cover different attack types?

Q3.  When I click to repository in Appendix I, I get the following error: "The repository is expired". The authors should ensure that the code is accessible.

### Soundness
4

### Presentation
3

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
This paper proposes TDRT, an adversarial training method for mitigating adversarial behavior manipulation, along with a novel black-box adversarial behavior manipulation method, BIA. BIA leverages adversarial imitation learning to train the adversarial policy using demonstrations of the adversary's target behaviors under the State-Adversarial MDP (SA-MDP) setting. TDRT then integrates BIA into the adversarial training process to learn a robust policy against behavior-targeted attack. Experiments show BIA and TDRT are comparable with previous works, SA-RL and SA-PPO, separately.

### Strengths
1. This paper is well written and easy to follow.
2. The motivation for developing black-box behavior-targeted attacks is clearly analyzed.
3. Experiments demonstrate the effectiveness of BIA and TDRT.

### Weaknesses
1. The novelty of this paper is limited. Firstly, regarding BIA, it is clear that BIA is equivalent to SA-RL; both approaches involve adversarial policy learning within the SA-MDP framework. The primary distinction is that BIA requires the adversary to provide several demonstrations of the target behavior, whereas SA-RL necessitates the adversary to develop a reward model for that behavior. Moreover, Theorem 5.1 in BIA closely resembles Lemma 1 from Zhang et al. (2020b), with the only difference being that the adversary's reward function in BIA is derived from the demonstration samples.
Secondly, concerning TDRT, it is also evident that TDRT is equivalent to SA-PPO. Theorem 6.1 is nearly identical to Theorem 5 from Zhang et al. (2020b), differing primarily in that TDRT employs a discounted form while SA-PPO uses a maximum form.

2. Due to the limited novelty, the performance of BIA and SA-RL, and the performance of TDRT and SA-PPO, are almost the same. Although the authors compare the clean task performance of TDRT and SA-PPO and show that TDRT achieves better clean task performance, it may be mainly due to the different hyperparameter sets. The authors should provide a clearer and fair comparison of the best performance of both SA-PPO and TDRT.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
