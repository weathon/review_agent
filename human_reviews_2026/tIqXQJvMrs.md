# USE: Enhancing Mixed-Motive Cooperation via Unified Self and Collective Rewards

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Mixed-motive cooperation requires agents to balance individual and collective rewards, often leading to tension between self-interest and cooperation. 
Conventional methods typically treat individual and collective rewards as completely independent, 
solving mixed-motive cooperation by maximizing their weighted sum (sometimes with auxiliary constraints). Because maximizing either the individual or the collective reward alone is sufficient to increase the weighted sum, such a design might lead to incorrect credit assignment and converge to overly selfish or altruistic policies. 
To address this, we propose a novel method named Unifying Self and collEctive rewards (USE). 
USE decomposes the individual reward into an independent part, unaffected by others, and a dependent part, shaped by interactions with others, then correlates the individual reward with the collective reward via the dependent part, as both the dependent part and collective reward arise from the interaction (including cooperation or betrayal) of agents. This coupling transforms maximizing individual and collective rewards into intrinsically correlated objectives, so that optimizing one implicitly promotes the other, reducing the risk of overly selfish or altruistic convergence. 
We conduct extensive experiments in mixed-motive cooperation tasks, demonstrating the effectiveness of USE. 
Interestingly, we find that the correlation between individual and collective rewards, to a certain extent, reflects the cooperative tendency of the agents. 
Our code is available at https://anonymous.4open.science/r/QPC-B6FD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces USE (Unifying Self and collEctive rewards), a novel method to enhance cooperation in mixed-motive multi-agent environments where balancing self-interest and collective good is crucial. Instead of treating individual and collective rewards as separate objectives, which can lead to free-riding or exploitation, USE unifies them by first decomposing an agent's individual Q-value into an "independent" component (from solitary action) and a "dependent" component (from interaction). The core of the method is the "Value Link" (VLink), which is the derivative of the collective Q-value with respect to the agent's dependent Q-value, effectively measuring if an agent's interactive behaviors are beneficial or harmful to the group. By integrating this VLink into the policy gradient update, the algorithm rewards pro-social actions and penalizes detrimental ones, thus aligning individual and collective goals to foster more effective and fair cooperation.

### Strengths
1. The empirical evaluation is comprehensive.  The authors structure their experiments around six distinct research questions, which helps to systematically investigate the method's performance, generalization, and fairness.  The included ablation studies offer supporting evidence for the utility of the designed components, such as the VLink mechanism and the hypernetwork.

2. The paper is well-written and generally easy to follow.  The use of the "Cleanup" game in the introduction is helpful for illustrating the core challenges of the mixed-motive setting, such as the free-rider problem.  This helps clarify the problem setting for the reader before the main technical contributions are detailed.

### Weaknesses
1. The paper's use of a centralized training (CTDE) paradigm is a notable limitation. It assumes access to global information that may be impractical in the realistic, decentralized scenarios that mixed-motive games aim to model. This makes the approach feel less scalable and potentially less applicable than methods designed for a fully decentralized (DTDE) setting, such as IAI, MOA and LASE [1].

2. The visualization in Figure 8 is puzzling.  The dependent (dep) module, which takes the local observation ($o_t^i$) as input, appears to only assign meaningful non-zero Q-values to the explicit interaction actions ("fire" and "clean").  However, since the agent's policy is also conditioned on $o_t^i$, the observation should contain sufficient signal to learn Q-values for other actions (e.g., movement) via the dep module as well.  The result suggests that only the independent (ind) module learns these values, which is strange. A possible, but undiscussed, explanation could lie with the hypernetwork. Since it also takes $o_t^i$ as input, is it possible that it learns to generate weights that effectively nullify or "gate" the `dep`  module's output for non-explicitly-social actions, routing the learning credit entirely to the `ind` module in those cases? And the hypernetwork architecture appears to draw inspiration from methods like QMIX (Figure 2 in USE is similar to Figure 2 in QMIX). However, in QMIX, the hypernetwork typically takes the global state $s_t$ as input to generate weights. What is the rationale for using $o_t^i$ over $s_t$ or $s_t^i$, and what would be the performance implications of changing this input?

3.  The process of "pruning" the global state $s_t$ to create the agent-specific state $s_t^i$ could introduce significant variance and misleading signals. For instance, if another agent eats an apple within the agent's field of view, the effect of pruning would be that the agent observes itself taking an action, and the apple simply vanishes from the state. It is unclear how prevalent such scenarios are and what impact this source of variance has on training stability.

4.  The paper presents quantitative results but falls short in providing a qualitative analysis of the agents' converged strategies. The fairness analysis, for example, reports a low Gini coefficient, suggesting equitable reward distribution. However, this seems to contradict the intuitive optimal strategy in Cleanup, which would be a fixed division of labor (e.g., one agent cleans, others harvest). Such a fixed-role strategy should result in a high Gini coefficient for extrinsic rewards. If, instead, agents are rotating roles, the paper does not explain how the USE mechanism overcomes the powerful learning signal that harvesting apples provides a much higher immediate reward than cleaning. Since the critic's TD-error objective would favor harvesting for both the `dep` (via observation) and `ind` (via state) inputs, a clearer explanation is needed for how the VLink signal consistently and successfully counteracts this greedy pressure to prevent a "tragedy of the commons" where all agents default to harvesting.

[1] Kong, Fanqi, et al. "Learning to balance altruism and self-interest based on empathy in mixed-motive games." Advances in Neural Information Processing Systems 37 (2024): 135819-135842.

### Questions
The mathematical notation for Q-value subscripts (e.g., "tot", "ind") is inconsistent across the paper (e.g., line 877 with \text{} vs. 239 without).

### Soundness
3

### Presentation
4

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
This paper introduces USE (Unifying Self and collEctive rewards), a novel framework for improving mixed-motive cooperation in multi-agent reinforcement learning (MARL), where agents must balance self-interest and collective welfare. Unlike previous approaches that treat individual and collective rewards as independent and simply optimize their weighted sum, USE decomposes each agent’s individual Q-value into an independent component (self-driven) and a dependent component (arising from interactions with others). It then correlates the dependent component with the collective Q-value through a differentiable measure called the Value Link (VLink), aligning both objectives at the value-function level. This coupling ensures that improving individual returns naturally promotes collective outcomes, mitigating problems like free-riding and exploitation. Experiments across standard social-dilemma benchmarks (Prisoner’s Dilemma, Chicken Game, Cleanup, Harvest, and Allelopathic Harvest) show that USE consistently outperforms prior methods in both performance and fairness. The study also demonstrates that the VLink can serve as a measurable indicator of agents’ cooperative tendencies.

### Strengths
1. The paper targets a long-standing weakness in mixed-motive MARL — the naive weighted-sum treatment of individual vs. collective rewards — and replaces it with a structured correlation mechanism. This represents a meaningful theoretical step beyond existing “reward reweighting” or “social influence” methods, which still assume separable objectives.
2. The Individual Dependence Decomposition (IDD) neatly separates the self-driven and interaction-driven components of each agent’s Q-value, providing an interpretable and modular formulation. The Individual Collective Coupling (ICC) with the derivative-based Value Link (VLink) offers a mathematically clean and differentiable way to quantify how an agent’s interaction behaviour affects group value — a novel analytic handle on cooperation.
3. By embedding the cooperative coupling term $g_i = \partial Q_{tot}/\partial Q_i^{dep}$​ directly into the policy gradient, the method avoids ad-hoc reward shaping and maintains theoretical coherence with standard MARL optimization. The inclusion of $A_i + \lambda g_i$​ preserves convergence guarantees, a rigor often missing in “social incentive” literature.
4. USE consistently outperforms six strong baselines (PPO, MAPPO, IAI, MOA, SLI, AGA) across diverse environments — from simple matrix games to long-horizon social dilemmas and adversarial mixed-motive tasks (Allelopathic Harvest). Results are robust to noise, scarcity, and population scaling (10 agents), suggesting generalization rather than over-fitting to specific dynamics.
5. The VLink offers a quantifiable proxy for cooperative tendency, observable through its correlation with prosocial actions (“clean”) and inverse correlation with antisocial actions (“fire”). The decomposition yields visualizable Q-value maps distinguishing dependent vs. independent contributions — rare interpretability in multi-agent RL.
6. The paper maintains consistency between intuition, architecture, and gradient formulation, supported by convergence analysis and boundedness proofs. The overall narrative — from social-dilemma motivation to algorithmic realization — is coherent and well-structured.
7. The authors systematically remove or alter key components (−VLink, −Hyper, −TLoss, etc.) and show predictable, interpretable degradation — reinforcing that each design choice contributes functionally rather than heuristically.

### Weaknesses
1. The method hinges on decomposing the individual reward into two components—“independent” vs. “interaction-dependent”. This decomposition requires the ability to prune other agents’ influence (for the independent part) and accurately capture the local interaction effects (for the dependent part). In many real-world or large-scale multi-agent systems such clean separation may not hold or may be very noisy.
2. The architecture uses a hypernetwork to fuse independent and dependent Q-values, and the Value Link (VLink) uses a partial derivative $\partial Q_{tot} / \partial Q_i^{dep}$​ to compute an agent’s contribution. These components assume differentiability, smoothness and manageable dimensionality. In very large, heterogeneous, non-stationary agent populations, or when observation spaces are huge, these assumptions may break down. Moreover, the experiments—while reasonably broad—still reside in more controlled simulated environments (e.g., “Cleanup”, “Harvest”, “Allelopathic Harvest”). Real world mixed-motive settings (with asynchronous agents, partial observability, communication issues) may behave quite differently.
3. By strongly coupling individual and collective rewards via the dependent component, there is a risk of suppressing innovative selfish behaviour that could benefit the group in the long term but doesn’t immediately align with collective Q-value. In other words, the approach may favour “safe cooperation” over “risky but beneficial divergence”. The paper does not deeply probe this trade-off in environments where individual novelty leads to collectively beneficial emergence.

### Questions
> 1. Please address the weaknesses.

> 2. The idea of structuring an individual agent's Q-value into two parts, i.e., $Q_i = f_{hyper}(Q_i^{ind}, Q_i^{dep})$, is novel in the literature of mixed-motive MARL. However, this idea has emerged in [1] that derives an additive form to aggregate $Q_i^{ind}$ and $Q_i^{dep})$, underpinned by Coalitional Affinity Game and prove the validity of this shaping, with connection to stable cooperation. The proposed $Q_i = f_{hyper}(Q_i^{ind}, Q_i^{dep})$ is a weighted form, generalizing the result from [1]. For this reason, I believe it should be with a discussion in this paper between the proposed structure of $Q_i = f_{hyper}(Q_i^{ind}, Q_i^{dep})$ and the linear shaping in [1]. Further, this could also underlie this paper with a game-theoretical foundation.

[1] Wang, Jianhong, Yang Li, Yuan Zhang, Wei Pan, and Samuel Kaski. "Open Ad Hoc Teamwork with Cooperative Game Theory." In International Conference on Machine Learning, pp. 50902-50930. PMLR, 2024.


> 3. Please address the following technical questions:

(1) The update rule (Eq. 14) uses $\nabla_{\theta_i} J = \nabla_{\theta_i} \log \pi_i(A_i + \lambda g_i)$.  How sensitive is training to the hyperparameter $\lambda$?  Does a fixed λ = 1 generalize across tasks, or does it require tuning?  The ablation (Appendix B.4) shows large performance drops for λ > 1 or = 0 — is there any theoretical guidance for its choice?

(2) Since $Q_{\text{tot}}$​ is computed from all agents’ $Q_i$​ (Eq. 6), doesn’t calculating $g_i^t = \partial Q_{\text{tot}}/\partial Q_i^{(\text{dep})}$​ back-propagate gradients through other agents’ critics, implicitly sharing parameters across agents during centralized training?  How does USE prevent information leakage or gradient interference between agents in decentralized execution?

(3) The convergence “proof” (Appendix C.1) relies on boundedness and non-zero weighting of $A_i+\lambda g_i$​.  But $g_i$ is itself learned and may be unbounded (e.g. with an etremely large value in practice) under certain activation functions.  Can the authors formally guarantee stability (e.g., in the sense of stochastic approximation theory) under function-approximation noise?

(4) Figure 7 claims that $g_i^t$​ correlates with “clean” vs. “fire” actions, but can $g_i^t$ ever flip sign within a single trajectory due to noise or network instability?  If yes, how does the algorithm prevent mis-classification of cooperative actions during training?

### Soundness
3

### Presentation
3

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
The core objective of this paper is to address the issues of inaccurate credit assignment and objective conflict in multi-agent mixed-motive cooperation tasks. Conventional methods typically optimize by maximizing the simple weighted sum of individual and collective rewards, $R = \lambda R_{individual} + (1-\lambda) R_{collective}$. The authors point out that this design can lead agents to converge to suboptimal policies, skewed towards either extreme selfishness or extreme altruism, failing to achieve a true dynamic balance between the two objectives. To overcome this limitation, the paper proposes the Unified Self and Collective Rewards (USE) method. USE aims to deeply fuse individual and collective interests, rather than simply superimposing them, through a novel reward structure or policy objective. This enables agents to more effectively learn an equilibrium strategy that both maximizes their own returns and promotes overall collaboration. The method emphasizes establishing an intrinsic link between individual and collective behavior in the reward signal design, guiding agent actions and enhancing collaborative performance and efficiency in mixed-motive environments such as the Prisoner's Dilemma or resource sharing tasks.

### Strengths
The core advantage of the USE method lies in its ability to overcome the fundamental flaws of goal conflict and inaccurate credit assignment in mixed-motive multi-agent collaboration. Traditional weighted-sum approaches merely perform an external trade-off between two independent objectives, often leading policies to converge to suboptimal extremes of either pure selfishness or altruism. In contrast, USE proposes a more elegant solution: through structured decomposition and unified individual rewards (for instance, decomposing individual rewards into dependent and independent components), it establishes an intrinsic linkage between self-interest and collective contribution within the reward signal itself. This intrinsic coupling ensures that while agents maximize their newly defined utility, their policy gradients are automatically aligned with the collective optimum, thereby accurately crediting behaviors that contribute incrementally to team outcomes. Ultimately, this mechanism enables USE to identify a Pareto-improving equilibrium strategy, which in experiments demonstrates both higher collective efficiency (total return) and sufficient preservation of individual interests, significantly enhancing collaborative performance in mixed-motive environments.

### Weaknesses
W1: In mixed-motive scenarios, the assumption of being able to access the private rewards (or utility functions) of other agents significantly limits the applicability of many existing multi-agent algorithms in real-world settings.

W2: The performance difference of the algorithm across different seeds is too large; the algorithm's stability seems to be lacking.

W3: https://anonymous.4open.science/r/QPC-B6FD The link is invalid/empty.

### Questions
Q1: Equations 1 and 2 mention the necessity of using the global state. This is inappropriate under the assumption of limited agent observation (i.e., a Partially Observable environment), as each agent can only acquire its local observation and should not be able to perceive all environmental information beyond its sensory range, even if the information regarding other agents is excluded. If this is indeed the case (i.e., using the global state is required), then your problem definition should be reformulated as a Markov Game with Global Observation. Furthermore, for the experimental comparison, were the baseline methods you adopted also permitted to access global or non-local environmental information extending beyond their local observation space during the observation phase?

Q2: Regarding MAPPO's significantly weaker fairness performance compared to USE in PD and CG environments, what is the primary reason for this disparity? Additionally, can you provide the training curves for individual rewards of MAPPO and USE in PD and CG?Whether all compared methods (including baselines) utilize the environment's raw reward for fairness evaluation?

Q3: Can you provide a visual comparison of the Q-value accuracy and stability achieved by USE versus baseline algorithms?

Q4: Can you provide a heat map illustrating the change in the cooperation rate of the USE algorithm under different parameter combinations of the Prisoner's Dilemma (PD) game?

Q5: Your method's training is based purely on a self-play training paradigm. If your opponent were to be switched to another algorithm, how would its performance be? Would USE be exploitable? For instance, if the opponent were an Always Defect strategy or an RL-driven one like PPO, a simple verification experiment could be performed in the Prisoner's Dilemma (PD)

### Soundness
2

### Presentation
3

### Contribution
3
