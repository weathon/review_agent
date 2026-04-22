# Revisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Value-based approaches such as DQN are the default methods for off-policy reinforcement learning with discrete-action environments such as Atari. Common policy-based methods are either on-policy and do not effectively learn from off-policy data (e.g. PPO), or have poor empirical performance in the discrete-action setting (e.g. SAC). Consequently, starting from discrete SAC (DSAC), we revisit the design of actor-critic methods in this setting. First, we determine that the coupling between the actor and critic entropy is the primary reason behind the poor performance of DSAC. We demonstrate that by merely decoupling these components, DSAC can have comparable performance as DQN. Motivated by this insight, we introduce a flexible off-policy actor-critic framework that subsumes DSAC as a special case. Our framework allows using an m-step Bellman operator for the critic update, and enables combining standard policy optimization methods with entropy regularization to instantiate the resulting actor objective. Theoretically, we prove that the proposed methods can guarantee convergence to the optimal regularized value function in the tabular setting, matching prior work. Empirically, we demonstrate that these methods can approach the performance of DQN on standard Atari games, and do so even without entropy regularization or explicit exploration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates off-policy actor-critic methods for discrete action tasks. Specifically, it empirically examines the effect of entropy regularization in both the critic and actor training processes, as well as the impact of variations in the objective function. The empirical results demonstrate how the inclusion of the entropy term influences performance in discrete action settings.

### Strengths
- Off-policy actor-critic methods for discrete action tasks remain relatively under-explored, making this study a timely and interesting contribution.

- The finding that entropy regularization in the critic is not necessary for discrete action tasks is particularly noteworthy and potentially useful for practitioners.

I agree that DQN variants currently dominate discrete action tasks, and thus exploring off-policy actor-critic methods in this domain is a meaningful research direction. The insight regarding the limited necessity of entropy regularization in the critic could offer practical implications for future algorithm design.

### Weaknesses
- The presentation lacks clarity in several sections, making the paper somewhat difficult to follow.

- The reported performance remains weak and still far from the current state of the art.

Regarding Equation (3), while it is referred to as the “natural policy gradient (NPG),” I do not believe this terminology is appropriate. Although both the solution in Equation (3) and NPG can be derived from reward maximization under a KL-divergence constraint, the solution represents a non-parametric solution to the reinforcement learning problem (Peters et al., 2010; Abdolmaleki et al., 2018). In contrast, NPG refers to the gradient direction of policy updates defined via the Fisher information matrix. While I acknowledge the conceptual connections, I am not convinced that Equation (3) should be described as NPG.

Additionally, the term “proposed methods” appears to refer to algorithms based on the objective functions shown in lines 251–262. However, these algorithms have already been introduced in prior studies, and it may not be appropriate to refer to them as “proposed.”

The experimental results are reported based on the parameters $\zeta$ and $\tau$. However, this notation is not intuitive, and I found it necessary to reread multiple sections to fully understand the meaning of the results.

The authors conclude that the investigated objective function achieved performance comparable to that of DQN. However, since DQN itself is far from state-of-the-art methods such as BTR [R1], the performance of the investigated off-policy algorithms also appears to be limited. Therefore, the reported results are not particularly promising.

Reference:
[R1] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC. Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. ICML 2025.

### Questions
- Although the authors investigated the objective function based on SPMA and NPG, I am not sure why SPMA is employed as the baseline. NPG is much more well-known than SPMA, and the performance of NPG-based methods appear to be better than those based on SPMA. Please explain why the analysis based on SPMA was necessary.

- I believe there are additional reasons why DQN variants dominate in the discrete action domain, beyond the effects of entropy regularization and the KL-divergence constraint. One important factor is that, in the discrete action setting, the Bellman optimality operator can be directly applied because the maximum over actions can be explicitly computed. This is not feasible in the continuous action domain.
This fundamental difference may explain why DQN variants are typically preferred for discrete action tasks, whereas off-policy actor-critic methods are more suitable for continuous action domains. Actor-critic methods are inherently indirect, as they rely on policy iteration, in contrast to value iteration methods such as DQN, which can directly approximate the optimal value function.
However, this important discussion appears to be missing from the paper. I would like to ask the authors for their thoughts on this point.

### Soundness
2

### Presentation
1

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
This paper proposed a generalized actor critic framework extending discrete SAC with a focus on discrete actions. The authors considered various possibilities such as different entropy coefficients for critic and actor, NPG/SPMA + forward/reverse KL divergence in the greedy step. The authors empirically surveyed these choices and the number of actor updates as well as auto coefficient tuning to give concrete conclusions on what combination might be more preferrable. Theoretical justifications were also provided with the same rate as prior work.

### Strengths
This paper is a nice summary of entropy regularized actor critic in discrete actions. The proposed general off-policy actor-critic framework is sufficiently large as a result of carefully distilling prior work. I work in entropy regularized RL and I appreciate that the authors comprehensively and accurately listed related papers along the way of developing their method. Decoupling of actor/critic entropy coefficient, NPG/SPMA for greedy step, the number of actor updates have all been discussed and experimented. The conclusions are to the point and of interest to the community.

### Weaknesses
Compared to the technical sections (though techniques from prior works are heavily used), the first section fails to deliver a convincing motivation. For example, in the abstract:
```
Common policy-based methods are either on-policy and do not effectively learn from off-policy data or have poor empirical performance in the discrete-action setting. 
```
claims like this seem to be overreaching and are not well supported by the paper. Did the authors take methods like Greedy-AC/InAC/TAWAC [1-3] into account? These methods have shown good performance in the discrete case as well and do not necessarily introduce entropy. Motivating entropy and therefore DSAC as a starting point for the proposed general family is reasonable, but it is not well connected to the claim that existing methods are flawed in discrete actions. I do think discrete-action AC is important, but I believe the current story needs to be modified to avoid criticizing all existing ACs. The motivation can come from motivating e.g. RL for healthcare that necessitates combinations of treatment. 

**References:**\
[1] The In-Sample Softmax for Offline Reinforcement Learning, ICLR 2023\
[2] Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement, ICLR 2023\
[3] q Exponential Family Policy Optimization, ICLR 2025

### Questions
1. M-DQN has their actor-critic version [1], have you compared to it? 
2. have the authors tried other actor critic methods like GreedyAC/InAC/TAWAC to see how they perform against DQN?
3. the authors claimed SPMA-RKL$(\zeta, \tau)$ is a novel method, but I can't find it in the experiments
4. in Figure 2 why DSAC-auto-ent$(\tau, \tau)$ is bad on the left column but good on the right? 


**References:** \
[1] Implicitly Regularized RL with Implicit Q-values

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a well-organized framework for discrete action off-policy actor–critic that decouples the actor and critic entropies, spans choices like KL direction (FKL/RKL), update style (NPG/SPMA), and m-step evaluation, and recovers several prior families as special cases. The empirical story leans on using this framework to diagnose DSAC and related design knobs—showing, for instance, that turning off critic entropy while keeping actor entropy helps a lot—and offering practical guidance on KL direction and entropy strength for discrete control. The theory is modular but for the tabular setting. It gives bounds that help reason about the design space but don’t address the non-linear function-approximation regime. The empirical results are competitive with DQN in many games, though notable gaps remain (e.g., Pong).

### Strengths
- This is an interesting paper that proposes a nice framework that can recover some prior families of algorithms. I am always happy to see well-organized papers that unify certain methods in a good framework.

- The core empirical results and discussions focus on using the proposed framework—which decouples the critic and actor’s entropy regularization—to examine DSAC and choices like KL direction, strength of entropy, etc. These provide nice insights with practical implications for using off-policy actor-critic methods in discrete settings. In that sense, I see the value of the paper for applying these methods to problems. It improves our understanding in this setting. I am personally in favor of papers that improve our understanding of existing methods in practice.

- The theoretical analysis in tabular settings seem to imply that the proposed framework and its bounds can match existing bounds as special instances. This has the added benefit of unifying different theoretical results / bounds under a single result, which I again appreciate.

### Weaknesses
- Theoretical results are restricted to the tabular setting, which is unrealistic. This kind of matches the existing theory, but still the setting of the paper is "discrete action" but not "tabular", so it is a bit disappointing.

- The paper does not necessarily position itself as a novel algorithm/approach compared to baselines; the contribution is more diagnostic/unifying than algorithmically new. This does not personally bother me, since I appreciate papers that improve our understanding. However it must be said. 

- Vanilla DQN is perhaps okay for making the paper's statement. However, when it comes to Atari, I would have liked seeing a stronger DQN-variant. Looking at Rainbow, we can see how much better Rainbow or even Distributional DQN, Dueling DDQN etc. are compared to vanilla DQN. This makes me wonder if the results we see here would still hold against a much stronger DQN-variant than vanilla DQN.

- There is no explicit Related Works section, connecting the paper into existing research.

### Questions
1. You state that Vieillard et al. (2020a) gives an $O(\frac{1}{K})$ for the $\zeta = \tau$ and $m=1$ case, but it is not immediately clear to me what bound we would get from your result for this setting. Can you confirm that you match this? If not, which term is the culprit here?

2. It seems that no matter what you have tried, DSAC or any other actor-critic variant cannot match DQN in Pong, and the gap is substantial. In your opinion, what could be the reason?

3. What was the reason for using vanilla DQN in your comparisons? Please correct me if you have not used vanilla DQN. There are much stronger DQN variants for Atari, so this makes the relative performance comparisons suspect.

### Soundness
3

### Presentation
3

### Contribution
3
