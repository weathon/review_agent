# Guided Actor-Critic: Off-Policy Partially Observable Reinforcement Learning with Privileged Information

- Decision: Reject
- Scores: 4, 4, 4, 4, 8

## Abstract
Real-world decision-making systems often operate under partial observability due to limited sensing or noisy information, which poses significant challenges for reinforcement learning (RL).
A common strategy to mitigate this issue is to leverage privileged information—available only during training—to guide the learning process. While existing approaches such as policy distillation and asymmetric actor-critic methods make use of such information, they frequently suffer from weak supervision or suboptimal knowledge transfer. 
In this work, we propose Guided Actor-Critic (GAC), a novel off-policy RL algorithm that unifies privileged policy and value learning under a guided policy iteration framework. 
GAC jointly trains a fully observable policy and a partially observable policy using constrained RL and supervised learning objectives, respectively.
We theoretically establish convergence in the tabular case and empirically validate GAC on challenging benchmarks, including Brax, POPGym, and HumanoidBench, where it achieves superior sample efficiency and final performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel off-policy reinforcement learning algorithm—Guided Actor-Critic (GAC)—for solving partially observable Markov decision processes (POMDPs) with privileged information. The method integrates a "guider" policy, which has access to the full state, and a "learner" policy, which only receives partial observations, within a jointly trained framework.

### Strengths
**Clear problem formulation and motivation.**: 

The paper clearly articulates the challenges of leveraging privileged information during training in POMDPs and precisely identifies the limitations of the two existing paradigms—privileged policy learning and privileged value learning: the former may lead to suboptimal distillation, while the latter provides only indirect supervision. 

**The core idea of GAC.**

Jointly optimizing the guider and learner under a KL-divergence constraint is relatively straightforward. It successfully extends the guiding principle of Guided Policy Optimization (GPO) from an on-policy setting to an off-policy one. The approach appears intuitively sound and practical.

### Weaknesses
**About Lemma and Method.**
In general partially observable Markov decision processes (POMDPs), the optimal policy under full observability (MDP) is typically unattainable—and often infeasible—under partial observability. This distinction is critical both theoretically and in practice.
In an MDP, the optimal policy \(\pi^*_{\text{MDP}}(a|s)\) is a function of the true state \(s\) and can make decisions directly based on complete state information.
- In a POMDP, the agent cannot observe the true state \(s\); instead, it must base decisions on the observation history \(\tau = (o_0, a_0, \dots, o_t)\), or equivalently, on the belief state \(b_t(s) = P(s_t = s \mid \tau)\). The optimal POMDP policy is thus \(\pi^*_{\text{POMDP}}(a|\tau)\) (or \(\pi^*(a|b_t)\)).
- Because the belief state generally contains strictly less information than the true state (i.e., \(I(b_t; s_t) < H(s_t)\)), the optimal return in a POMDP is usually **strictly lower** than that of the corresponding MDP:
  \[
  J^*_{\text{POMDP}} < J^*_{\text{MDP}}.
  \]

Therefore, **attempting to make a partially observable policy imitate the fully observable MDP-optimal policy is inherently misaligned**, as the latter may be unrealizable under partial observability and could even induce irrational behavior in the learner (e.g., "braking early" without perceptual evidence of an obstacle).

Two examples :

1. **Texas Hold’em Poker**:  
   - A "God’s-eye-view" policy knows all players’ hole cards and can make theoretically optimal betting/folding decisions.  
   - A real player, however, only knows their own cards and public information, and must infer opponents’ ranges. Imitating the God’s-eye policy would lead to overconfidence or incorrect inferences, reducing win rates.

2. **Occlusion in autonomous driving**:  
   - A full-state policy might brake early because it "knows" a pedestrian is hidden behind a large vehicle.  
   - A partially observable policy that blindly mimics this behavior—without sensory evidence—might brake unnecessarily in clear scenarios, compromising ride comfort or even causing rear-end collisions.
Given this, Theorem 3.4—which relies on that assumption—and the subsequent method described in Section 3.2 appear to be theoretically unsound. 

I believe that in some partially observable environments (maybe contain some noisy observations), it is impossible to achieve the same policy as in the fully observable MDP. That said, I do not deny the empirical improvements the authors have achieved through engineering optimizations; however, the theoretical assumption underlying their approach seems somewhat inappropriate.

**About related works.**

I concern that "since supervision is provided indirectly through the RL objective, it may be less sample-efficient than methods that
leverage direct expert supervision."  In some environments, such as autonomous driving, direct policy distillation may fail to recover the optimal policy, whereas effective value-function guidance can succeed.  As Figure 1, SAC-asym outperforms TGRL in most cases.

**About Experiment.**

The experiments compare against SAC-asym (which, aside from the proposed method, appears to perform the best). However, this set of baselines seems relatively weak. Relevant approaches such as leverages privileged information in the value
function **UAAC**[1] (maybe an SAC-based variant), representation learning methods **Believer**[2] are not included in the comparison.

[1] Unbiased asymmetric reinforcement learning under partial observability. AAMAS 2022.

[2] Learning belief representations for partially observable deep rl. ICML 2023.

### Questions
Whether the additional training resources incurred by the two policies have been evaluated. I think the paper's approach leans toward practical computational efficiency,  it is necessary to evaluate the additional resources and time required.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Guided Actor–Critic (GAC), an off-policy algorithm for RL with training-time privileged information under partial observability. The key idea is a guided policy iteration scheme that co-trains a guider policy and a learner policy under a KL alignment constraint. Empirically, GAC is benchmarked on Brax with noisy observations, POPGym memory tasks, and HumanoidBench manipulation with tactile signals available only during training. The authors report better sample efficiency and final returns than asymmetric SAC, TGRL/teacher-student, RMA, and on-policy GPO-clip.

### Strengths
1.The paper tackles the important and practical problem of sample-efficient learning in partially observable environments using training-only privileged information.

2.The paper is well-written and situates itself perfectly within the existing literature, with a clear motivation and a thorough related work review.

### Weaknesses
1.The final GAC algorithm is extremely complex, requiring six separate neural networks (a guider policy, a learner policy, two guider Q-networks, and two learner Q-networks) and five distinct loss functions (two Q-losses, two policy losses, one $\alpha$-loss). This high complexity is a barrier to reproduction and adoption.

2.The tabular analysis suggests a KL-aligned learner (distillation-like), but the practical method adds an auxiliary RL loss for the learner and separate learner critics. Ablations indicate the auxiliary learner RL loss is important for performance; separate Qs are often helpful but not universally critical. This weakens the practical relevance of the “pure distillation” narrative.

3.Although the paper formally relates GAC to MPO, it provides no empirical MPO-style baseline (with/without privilege), leaving it unclear whether gains stem from asymmetry + guidance or from a generic KL-regularized update.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Guided Actor-Critic (GAC) for POMDPs that leverages training-time privileged information. The method co-trains a privileged guider policy $\mu(a\mid s)$ and a partially observable learner policy $\pi(a\mid o)$ via guided policy iteration: (i) guided evaluation applies a modified Bellman backup that includes a KL term between $\mu$ and $\pi$; (ii) guided improvement updates $\mu$ toward a $Q$-weighted version of $\pi$ and then distills $\mu$ into $\pi$ via KL. The paper provides tabular convergence and a practical deep RL algorithm with automatic KL temperature and clipped double Q. On Brax with observation noise, POPGym memory tasks, and HumanoidBench where tactile signals serve as privileged info, GAC improves sample efficiency and final performance over asymmetric SAC, TGRL, and GPO-clip, and analyzes failure modes at high noise and target-KL sensitivity.

### Strengths
Unified use of privileged info: combines privileged policy learning and privileged value learning in a single off-policy framework, improving sample efficiency.

Clear algorithmic structure: guided evaluation and improvement are well specified; the learner receives direct supervision from the guider plus RL signals.

Theory plus practice: convergence in the tabular case and a practical deep variant with automatic KL control and clipped double Q.

Broad benchmarks: consistent gains on Brax (noisy partial observations), POPGym (memory), and HumanoidBench (tactile privileged info).

Insightful analysis: identifies a high-noise failure case and studies the effect of the target KL range.

### Weaknesses
Function-approximation theory gap: convergence results are tabular; there is no error-propagation or stability analysis with neural critics and replay.

Reliance on value quality: the guider update uses $Q$-weighted reweighting; bias in $Q$ under partial observability may misguide $\mu$ and $\pi$.

Target-KL tuning sensitivity: despite auto-temperature, performance can degrade for extreme target KL and very noisy observations.

Baselines: more competitive off-policy privileged baselines (e.g., MPO-style privileged variants) would sharpen the study.

Ablations: limited diagnostics on how much gains come from KL supervision vs learner RL, and on guider capacity assumptions.

### Questions
$Q$ robustness: have you tried ensembles or uncertainty-aware weighting in the $Q$-weighted improvement of $\mu$ to reduce bias in early training

Guider capacity: how sensitive are results to the architecture of $\mu$ and to the assumption that $\Pi_{\mu}$ can drive the KL in the improvement step near zero

Replay coupling: since data are collected by $\mu$ while $\pi$ is supervised, do you observe distribution-shift issues in the learner critic If so, any remedies beyond target networks

Memory modeling: on POPGym, what recurrent modules are used for $\mu$ and $\pi$ Are the architectures symmetric If not, how much does this matter

HumanoidBench tactile: can you quantify how often $\pi$ imitates $\mu$ vs deviates when tactile info was critical for $\mu$ but unavailable to $\pi$

### Soundness
3

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
The paper provides an approach for combining asymmetric actor-critic and teacher-student policies for leveraging privileged information in POMDPs. The paper presents experiments on various benchmarks and provides analysis that demonstrates the convergence of their actor-critic approach.

### Strengths
- Combining asymmetric actor-critic with a teacher-student approach. To provide a stronger supervision signal to the actor.
- Adding the divergence between the teacher and student policy to the value function backup seems like it might lead to the teacher policy taking actions that keep the agent in regions where the student agent can imitate the teacher. It appears to be a novel idea, although related to [1] and [2].

[1] Nguyen, Hai, et al. "Leveraging fully observable policies for learning under partial observability." arXiv preprint arXiv:2211.01991 (2022).

[2] Messikommer, Nico, et al. "Student-informed teacher training." arXiv preprint arXiv:2412.09149 (2024).

### Weaknesses
- It is not entirely clear why the provided method is superior to other similar approaches, for example, [1], which also leverages a teacher policy alongside a symmetric actor-critic approach.
- The epsilon variable needs to be tuned as well; it's not very clear how sensitive the method is to such a variable
- no ablation studies

### Questions
- Can the authors provide ablation studies of the different parts of their algorithm? It's difficult to disentangle the effect of each module as the paper currently stands
- Can the authors compare their algorithms with other methods combining teacher/student policies and asymmetric actor-critic, such as [1] ?
- Why can't your approach be off-policy?

I would be willing to raise my score if the authors can provide the required experiments.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a policy optimization algorithm that leverages privileged information at training time. Specifically, a "guider" policy with access to privileged information, is trained to find high rewards while being softly constrained to the unprivileged learner policy. The learner policy is trained to imitate the guider. The paper establishes some theoretical backing for this method, and evaluate it on several benchmarks where it outperforms other privileged information baselines.

### Strengths
- Very nicely written and presented. The paper does a good job of setting up the problem, discussing the body of related works, explaining the method and necessary theory, and then showing the strengths and limitations of the method 
- theoretical backing seems solid, although with the caveat that the KL divergence is actually met
- empirical section is strong. the authors compare against a variety of privileged info baselines, and in different domains.

### Weaknesses
I don't see any major weaknesses, but there are some minor ones to round out the paper. 

- Discussion of "on-policy-ness" of using a guided policy to generate data for the learner. I could see in early stages, the privileged guider is useful. But eventually, it seems like a good idea to anneal to just the learner policy to generate data to avoid off policy issues, even if the guider is being softly constrained to act like the learner. Could there potential convergence issues or unstable training dynamics since the guider is likely always going to be a bit "off policy" wrt learner? 
    - In Section 4.4, there is an experiment where KL divergence fails to converge. But what if you just train the learner on a convex combination of the guider and learner state action distribution, and anneal towards the learner over time?

- It would be nice to have some more task-specific  qualitative analysis / interesting behaviors / visual figures in the experimental section. 
- Missing some references. The seminal work is Vapnik's SVM+[0]. Missing some privileged model-based rl works as well. Would like to see Vapnik properly cited in main work, and MBRL works could be supplemental in Appendix A.

0. A new learning paradigm: Learning using privileged information
1. The Wasserstein Believer: Learning Belief Updates for Partially Observable Environments through Reliable Latent Space Models (ICLR24)
2. Informed POMDP: Leveraging Additional Information in Model-Based RL (RLC25)
3. Privileged Sensing Scaffolds Reinforcement Learning (ICLR24)
4. TWIST: Teacher-Student World Model Distillation for Efficient Sim-to-Real Transfer (ICRA24)

### Questions
See weaknesses above. Overall I am fairly positive about this paper, although this is subject to change depending on the author's response and other reviewers' concerns.

### Soundness
3

### Presentation
4

### Contribution
3
