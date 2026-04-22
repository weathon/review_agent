# Improving Online Reinforcement Learning via Behavior Prior Distillation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Existing behavior prior reinforcement learning (BPRL) algorithms predominantly rely on offline pre-training, where a behavior cloning model is learned from offline datasets, and policy priors are used to guide the online fine-tuning of the agent. However, the limited quality of offline datasets often hinders the ability to provide high-value policies that can effectively guide policy updates. The absence of expert trajectories significantly impairs online policy learning, leading to low sample efficiency and suboptimal performance. To address these challenges, we depart from conventional behavior prior approaches and propose a Bidirectional Behavior Prior Distillation (B2PD) algorithm. B2PD leverages action-value priors to guide a conditional variational autoencoder (CVAE) in generating a high-value behavior support set. The resulting expert behavior priors are further distilled into the agent, effectively reducing inefficient exploration and enabling stable policy optimization, while establishing a bidirectional knowledge flow mechanism. Experimental results on across both state- and pixel-based environments demonstrate that B2PD significantly improves both sample efficiency and overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes an off-policy actor critic algorithm for online reinforcement learning. Departing from the standard approach of distilling a prior from offline data, the method relies on a behavior prior distilled from the data collected online, and guided by the critic. In practice, the algorithm seems to combine different techniques and ideas:
- entropy regularization as in SAC
- a multi-modal behavior prior (CVAE)
- a random-shooting procedure for computing action targets while distilling the prior
- noise injection over actions as in TD3

The combination of these techniques produces the proposed algorithm, B2PD, which is evaluated extensively in state-based and visual online RL (across standard Mujoco and Pybullet continuous control environments). The submission is concluded by an ablation visualizing entropy curves on a single task, and by a detailed Appendix.

### Strengths
- The goal of designing a sample efficient online RL algorithm remains relevant.
- The experimental evaluation is sufficiently broad and includes several relevant baselines.
- Empirically, the method seems to perform rather well, at a modest computational overhead.

### Weaknesses
- The core idea of the method is not justified formally, to the best of my understanding. Despite a few propositions, it's unclear why one should train both a policy and a multi-modal prior on the same data. Behavioral priors are almost exclusively deployed in asymmetric settings, in which the prior and the policy are exposed to different data sources and objectives: in the most common setting, the prior is trained on offline, task-agnostic data, while the policy is trained online on a different downstream task. In this case, the benefits of a prior which was exposed to more data is clear. In this submission, both policy and prior are trained on the same data, with similar objectives (maximizing Q-values, with some regularization). Can the authors provide a formal justification of why a prior is necessary in these settings? If the issue is simply multimodality, can we directly train a multi-modal policy? 
- Aside from the core contribution, the method and its components appear not to be principled, despite formal arguments:

(i) The method is presented in the standard max-entropy framework, but it is not clear why. In fact, the method is also applied on top of TD3 in experiments. Is there any fundamental synergy between the proposed method and the max-entropy framework? If not, what is the significance of Section 3.2?

(ii) Standard (linear) online RL admits an optimal deterministic policy. In this case, the benefit of a multi-modal action distribution is unclear. This is of course not the case for entropy-regularized RL, but given that the algorithm is also combined with TD3, the question still stands.

(iii) Proposition 3.3 appears to be wrong in general MDPs (i.e., consider an MDP with a constant rewards, or rewards matching the log-prob of a Gaussian over actions in the max-entropy case).

(iv) To the best of my understanding, Propositions 3.4-3.6 are the standard soft policy iteration results from SAC. Their relevance to this method is not clear, and the original results are not referenced.

(v) The SDA noise scheduling mechanism appears to produce actions from the sum of two Gaussians, which is in turn Gaussian. As the variances of the two are connected by Eq. 8, this seems equivalent to scaling the variance specifically for the action used for computing value targets, instead of a TD3-style noise injection.

- The experiments do not help disentangling which of the proposed components (summary) is inducing the algorithm's performance. Ideally, each of the component which deviates from SAC should be ablated independently.

### Questions
## Minor issues and questions:
- Line 31 blames the issues of off-policy algorithm on the Bellman equation, which "leads to ineffective exploration". I think this is a strong mischaracterization: exploration is a fundamental problem in RL, and the Bellman equation does not induce it in any particular way. Can the authors further comment on this?
- Section 3.1 directly begins with an Assumption, which should perhaps be introduced in text.
- Assumption 3.1: $G$ and $G_\omega$ are used interchangeably.
- Why is Proposition 3.3 is surrounded by round brackets?
- Line 208: is Fujimoto 2019 the right reference for CVAEs?
- Section 4.1 lacks references for nearly all baselines.

## Conclusion
In my opinion, while the method performs rather well, it is also overcomplicated and not principled. Formal results are not directly relevant to explaining the method's performance, and several components seem to be redundant (e.g. injecting noise on top of a max-entropy policy). For these reasons, I currently recommend rejection. Disentangling why the method works, either empirically or formally, would constitute an important priority in my opinion.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
**Paper summary**: This work studies how to apply a pre-collected dataset to online RL performance improvement. Specifically, they focus on behavior prior RL (BPRL), which trains a behavioral cloning policy from a pre-collected dataset, thereby distilling it into an online RL policy. Like offline RL, they suffer from chronic limitation, which heavily relies on the quality of the pre-collected dataset. To alleviate this issue, this paper introduces a bidirectional behavior prior distillation (B2PD) algorithm. The main idea is simple: 1) train the CVAE policy with Q guidance and 2) distill it into an online RL policy with a simple RL objective. They show the justification of their algorithmic choice using a Toy example and ablation study. Additionally, B2PD outperforms the selected baselines over both the state- and pixel-based environments, including seven MuJoCo, four PyBullet, and four DMControl tasks.

---
**Summary of review**: Overall, the paper is clearly written and relatively easy to follow. However, the motivation and methodological exposition lack clarity, and several parts of the paper contain elements that reduce its overall polish and completeness.
From a technical standpoint, using a CVAE to model a multimodal policy distribution and distill it into the actor is interesting, yet the approach still relies on rather strong assumptions and feels somewhat limited in conceptual scope. In practice, comparisons with more modern generative modeling techniques such as flow matching or diffusion steering would strengthen the argument for its novelty and effectiveness. Furthermore, the theoretical analyses appear to be moderate extensions of existing results rather than introducing fundamentally new insights. In summary, the contribution is closer to an engineering refinement than a conceptual breakthrough. To sum up, I therefore assign an initial score of 4, while leaving room for possible adjustment pending clarifications and additional justifications in the author response.

### Strengths
**Writing**
- The paper is well-structured and relatively easy to follow.
- The authors clearly articulate the fundamental limitation of behavior-prior RL methods and justify why addressing this issue is both timely and important for improving online RL performance.
- Key assumptions are clearly stated, helping to delineate the theoretical scope and strengthen the paper’s transparency.

**Methodology**
- The proposed solution is conceptually simple, but it shows powerful performance; in addition, some parts could be easily integrated into standard RL frameworks.
-  The inclusion of prior distillation and SDA presents a good engineering design aimed at stabilizing optimization while reducing inefficient exploration.

**Experiments**
- This work includes diverse experiments, spanning $7$ MuJoCo, $4$ PyBullet, and $4$ DMControl tasks, with consistent hyperparameters and 10 random seeds per environment.
- Both state-based and pixel-based settings are tested.
- The appendix includes sensitivity analyses for key hyperparameters and thorough ablation studies, reinforcing the reproducibility of results.

**Theoretical support**
- The theoretical analysis provides a reasonable degree of mathematical justification for convergence and Q-value smoothness.
- These analyses help to confirm the soundness and stability of the optimization dynamics.

### Weaknesses
**Writing**: weak motivation and framing
-  While the problem setup is valid, the justification for transitioning to a purely online RL paradigm is not entirely convincing. The paper may overstate the generality of online-only applicability without fully addressing hybrid or offline-to-online alternatives.
- The discussion lacks a comparative reflection on when offline priors are still beneficial or how the proposed method complements existing pretraining pipelines

**Methodology**
- The proposed SDA noise scheduling is designed with two constants (0.2 and 1.5) whose motivation or sensitivity is not well explained.
- Compared to baselines, the authors do not analyze computational overhead quantitatively. 
- Behavior-prior generation heavily relies on the assumption of a well-trained Q-function with broad coverage. Theoretical guarantees may fail under biased or sparse datasets; this limitation should be acknowledged more explicitly.

**Theoretical depth**
- Propositions 3.4–3.6 mainly restate established RL principles with incremental extensions.
- The analysis is sound but does not introduce fundamentally new theoretical insights. The reviewer thinks that moving some incremental theorems to the appendix and expanding more empirical reasoning would improve focus.

**Experiments**
- The reviewer thinks that the toy example illustrates behavior qualitatively but does not convincingly link trajectory patterns to sample efficiency or exploration coverage. Quantitative measures of state-space visitation or gradient signal analysis would better support the claims.
- The authors ask, `Why does behavior prior distillation outperform entropy-driven exploration?', but provide only empirical evidence rather than a logical explanation.
- The baselines are somewhat outdated. More recent algorithms, such as TD7 [1], Mr.Q [2], or other modern behavior-prior representation methods, should be considered.
- There is no comparison to recent or alternative exploration strategies, for example, intrinsic reward [3], diversity-driven [4], distributional RL [5], or other RL on prior data (RLPD) [6]. Similarly, while the related-work section cites [7-9], these works are not included in experimental comparisons or deeper discussions.
- There is no main table to grasp overall performance across all benchmarks and tasks. The reviewer thinks that it would be better to provide the main summary table, consolidating all benchmark results and averaged performance across tasks.

**Miscellaneous**
- Formatting and typesetting issues appear, e.g., garbled characters in Figure 1 and Figure C.1.
- Notational inconsistencies:
   - The discount factor $\gamma$ is stated as $\gamma \in (0,1)$, through theoretically it can be $\leq 1$.
   - Some variables (\theta, \phi, \tau) are not introduced clearly on first use. 
   - Section organization could be smoother. In the experimental section, there is a solved research question by a toy example.
   
**References**

[1] S. Fujimoto, et al. For SALE: State-Action Representation Learning for Deep Reinforcement Learning. NeurIPS 2023.

[2] S. Fujimoto, et al. Towards General-Purpose Model-Free Reinforcement Learning. ICLR 2025.

[3] N. Chentanz, et al. Intrinsically motivated reinforcement learning. NeurIPS 2004 

[4] D. Pathak, et al. Curiosity-driven exploration by self-supervised prediction. ICML 2017.

[5] W. Dabney, et al. Distributional Reinforcement Learning with Quantile Regression. AAAI 2018.

[6] P. Ball, et al. Efficient Online Reinforcement Learning with Offline Data. ICML 2023.

[7] H. Zang, et al. Behavior Prior Representation learning for Offline Reinforcement Learning. ICLR 2023.

[8] G. Spigler. Proximal Policy Distillation. arXiv 2024.

[9] M. Nakamoto, et al. Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. NeurIPS 2023.

### Questions
- What is the computational cost of training and sampling from the CVAE, particularly as the number of sampled priors $H$ increases? Please discuss its bottleneck or limitations that might arise when scaling to an image-based environment.
- Related to hyperparameter sensitivity ablation, are there practical tuning heuristics or observed failure modes when these parameters are mis-set?
- Could the authors provide a direct comparison or analytical discussion with recent approaches, as mentioned in the weakness section?
- Table E.1 and Figure E.1 compare KL-divergence vs. MSE objectives for distillation. Do the authors have principled guidelines for choosing between them depending on the policy’s stochasticity or parameterization?
- To what extent can the CVAE generate actions outside the replay buffer support?
Does it meaningfully encourage exploration beyond previously seen behaviors, or mainly reinforce high-value regions already represented in the buffer?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes B2PD: Bidirectional Behavior Prior Distillation for online RL. Rather than relying on offline pretraining, B2PD builds a behavior prior online using a CVAE and establishes a two-way knowledge flow:
(1) Action-Value Prior Distillation (AVPD) trains the CVAE with guidance from a learned $Q$ so that it samples a high-value support set;
(2) Behavior Prior Distillation transfers the best actions from that set back to the actor via a KL-style anchor loss.
The method also introduces a Standard Deviation Aware (SDA) noise schedule to stabilize soft $Q$ targets and provides tabular convergence for policy evaluation, improvement, and iteration. On MuJoCo, PyBullet, and DMControl (including pixel-based DrQ-v2 settings), B2PD improves sample efficiency and final returns over TD3/SAC and prior-guided baselines, with ablations and a toy study that visualize reduced inefficient exploration.

### Strengths
Online priors without offline data. A generative prior creates diverse, high-value anchors and distills them into the actor.

Bidirectional design. AVPD guides the CVAE with $Q$; prior anchors regularize the actor, reducing aimless entropy-driven exploration.

Stability add-on. SDA smooths value targets and reduces variance.

Broad evaluation. Consistent gains across 16 tasks, including pixel-based settings.

Ablations and toy study. Each module’s effect is isolated; exploration becomes more targeted over training.

### Weaknesses
Dependence on value quality. AVPD uses a learned $Q$ to steer the CVAE. Failure modes under biased $Q$ or sparse rewards are not deeply diagnosed.

Anchor selection sensitivity. The $\arg\max$ over $H$ sampled actions may be sensitive to $Q$ noise; the compute vs. robustness trade-off for $H$ is underexplored.

Theory scope. Convergence is shown in tabular settings, not with function approximation or under distribution shift.

Pixel baselines. Visual experiments mainly use DrQ-v2; more backbones would strengthen claims of modality robustness.

### Questions
$Q$ uncertainty. Have you tried ensembles or variance-aware filters to avoid over-optimistic AVPD targets, especially early or in sparse-reward tasks

Anchor budget $H$. What is the trade-off between $H$, wall-clock time, and final return Do you observe diminishing returns beyond $H=10$

SDA schedule. How sensitive are results to the constants in the SDA equation and to excluding noise from the entropy term Could per-dimension scaling collapse exploration

CVAE underfit. If the CVAE misses high-value modes, how quickly can B2PD recover Any diagnostics to detect support gaps

Pixel-based generality. Does B2PD transfer to other visual backbones (for example DrQ-v3, RAD, PI-SAC) without retuning

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
4

### Summary
This paper addresses the sample inefficiency of online RL by proposing Bidirectional Behavior Prior Distillation (B2PD), which replaces conventional offline behavior priors with dynamically generated high-value policy guidance. B2PD trains a CVAE using Q-value-guided gradients to produce a diverse support set of actions. The highest-value actions from this set are distilled into the agent policy, reducing inefficient exploration. ). Experiments on continuous control tasks (MuJoCo, PyBullet, DMControl) demonstrate improved sample efficiency and performance over baselines like SAC and TD3.

### Strengths
- Novel way for combining generative modeling and RL.
- Theoretically grounded contributions
- Comprehensive and rigorous evaluation

### Weaknesses
- In Figure 2, B2PD(w/o AVPD) converges faster (at 100K steps) to the states with high reward than B2PD (at 150K steps). This result seems to show that the AVPD module is even harmful for effective state exploration.
- Learning under value-guided offline policy distribution has been widely researched in offline RL before. For example, the advantage-weight CVAE model [1], and the advantage-conditioned CVAE model [2]. But None of them are referenced in the main text or compared in experiments. 
- Actually, the main contribution is the Q-value prior distillation loss in Eq. 10, where the CVAE decoder is constrained to decode actions with high Q value. This method is quite similar to LAPO [1] mentioned above, which also encourages the CVAE to consider the state-action pair with high advantage. Thus, I think the novelty is quite limited.
- Meanwhile, considering the ablation study of weight $\xi$ presented in Figure F.2, the effectiveness of Q-value prior distillation loss seems unclear and unstable, where sometimes the returns are even worse than without this loss.

### Questions
- Can the author provide a detailed pseudo-code in the appendix? Is both the CVAE and critic model trained in offline stage?
- Instead of retraining a new policy from scratch under the offline RL–trained or BC–trained policy, why not directly fine-tune the offline RL–trained policy in the online stage using Offline-to-Online (O2O) algorithms? I believe that O2O algorithms are more effective technique.
- More studies about the effectiveness of Q-value prior distillation loss should be conducted. For example, how will the policy perform under the CVAE trained by [1][2]?

[1] LAPO: Latent-Variable Advantage-Weighted Policy Optimization for Offline Reinforcement Learning. NeurIPS 2022

[2] A2PO: Towards Effective Offline Reinforcement Learning from an Advantage-aware Perspective. NeurIPS 2024

### Soundness
2

### Presentation
2

### Contribution
2
