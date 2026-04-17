# Extending Differential Temporal Difference Methods to Episodic Problems

- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Differential temporal difference (TD) methods are value-based reinforcement learning algorithms that have been proposed for infinite-horizon problems. They rely on reward centering, where each reward is centered by the average reward. This keeps the return bounded and removes a value function’s state-independent offset. However, reward centering can alter the optimal policy in episodic problems, limiting its applicability. Motivated by recent works that emphasize the role of normalization in streaming deep reinforcement learning, we study reward centering in episodic problems and propose a generalization of differential TD. We prove that this generalization maintains the ordering of policies in the presence of termination, and thus extends differential TD to episodic problems. We show equivalence with a form of linear TD, thereby inheriting theoretical guarantees that have been shown for those algorithms. We then extend several streaming reinforcement learning algorithms to their differential counterparts. Across a range of base algorithms and environments, we empirically validate that reward centering can improve sample efficiency in episodic problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Differential temporal difference (TD) methods, which rely on reward centering via the average reward, are generally inapplicable to episodic problems because centering can alter the optimal policy. This work proposes a generalization of differential TD to overcome this limitation, making it applicable to both discounted and undiscounted episodic contexts. The core mechanism involves defining a differential terminal value and proving that the modification preserves optimal policy invariance by mapping the generalized update onto the framework of potential-based reward shaping. The authors also establish a theoretical equivalence between the algorithm and a form of linear TD with a state-action-independent output-level bias unit. Empirical results in streaming deep RL environments (MinAtar, MuJoCo) suggest that the proposed generalization, integrated into Stream Q($\lambda$) and Stream AC($\lambda$), improves sample efficiency compared to uncentered baselines.

### Strengths
- The writing in this paper is clear, and I can easily grasp the information the authors intend to convey.
- The performance experiments in the paper demonstrate that the method has some effectiveness.

### Weaknesses
- I believe the novelty of this paper is very limited; it merely incorporates commonly used terms from reward shaping methods into differential Q Learning, without offering any new innovation.
- The paper does not provide any theoretical or experimental analysis that offers insight; the experiments are almost entirely performance evaluations.
- The amount of work presented in this paper is modest, and I believe the contribution is not sufficient to meet the bar for ICLR.

### Questions
- Could you provide some valuable insights through theoretical or experimental analysis to justify the necessity of the method proposed in the paper?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper revisits differential temporal difference (TD) learning, a class of value-based reinforcement learning (RL) algorithms traditionally applied to continuing infinite-horizon problems. Differential TD methods rely on reward centering, subtracting the average reward from each observed reward to stabilize learning and remove constant value offsets. However, this reward transformation is not invariant to optimal policies in episodic tasks, because subtracting a constant shifts the value of trajectories of different lengths unequally. As a result, differential TD could distort the policy ranking in problems with termination. 
The authors propose a generalization of differential TD that extends its applicability to episodic reinforcement learning, both discounted and undiscounted. By reinterpreting reward centering through the framework of potential-based reward shaping, they show that adding a specific terminal correction preserves the ordering of optimal policies in episodic environments. Moreover, they demonstrate the equivalence between the proposed generalization and a TD update with an explicit bias unit, a state- and action-independent output parameter that functions as a learnable baseline. This equivalence allows the generalized differential TD to inherit theoretical guarantees from linear TD methods.
Overall, the work provides both theoretical and empirical evidence that reward centering can be safely and beneficially applied in episodic reinforcement learning.

### Strengths
The central contribution, viewing differential TD as a special case of potential-based reward shaping with terminal correction, is elegant and rigorous. The analysis in Sections 3–4 clearly shows how to define a terminal differential value so that subtracting the average reward does not alter the policy ranking. The reinterpretation of differential TD as TD with a bias unit (Eq. 1–5) is particularly insightful, bridging a gap between continuing and episodic settings.
The equivalence proof allows the algorithm to inherit established convergence properties of linear TD, giving the proposal strong theoretical grounding.
The paper balances formal reasoning with practical implications. For example, the grid world examples (Fig. 1–2) are simple yet clearly demonstrate when and why centering helps (notably under “painful” dense reward conditions). The potential-based reward shaping analysis (Section 3) naturally connects to the algorithmic implementation (Algorithm 1) and the bias-unit interpretation (Section 4). The theoretical argument that episodic problems can be viewed as continuing problems with state-dependent discounting (Appendix D) is a clean conceptual unification.
The authors evaluate across diverse scales and settings. In almost all settings, differential versions improved or matched baseline performance, while never performing worse, supporting the claim that reward centering is safe and often beneficial.
The proposed changes are minimal and easily applicable: Only one additional scalar parameter (b) and one step-size are introduced. The approach can be applied to existing algorithms “as-is,” requiring no structural change. This makes it highly practical for resource-constrained or streaming RL settings, where buffer-free online learning is essential.
Appendices A–D list exact hyperparameters, search ranges, and algorithmic details.
The pseudocode (Algorithm 1) is clear and unambiguous. The paper also promises to release source code upon acceptance.

### Weaknesses
While the reinterpretation is elegant, it largely builds on existing principles (potential-based reward shaping, bias-augmented TD, and differential TD) rather than introducing fundamentally new algorithms or analyses.
The main theoretical novelty lies in recognizing how to apply these concepts to episodic settings, but the resulting mathematics is straightforward.
The tabular experiment (Figure 1) is limited to two simple grid worlds.
Although the results serve their illustrative purpose, including more challenging episodic tasks (e.g., stochastic transitions or larger state spaces) could provide stronger validation.
The additional step-size parameter (eta) introduces a hyperparameter tuning burden.
While this is acknowledged (Sec. 6), more analysis on its sensitivity and its relationship to alpha or gamma would improve the reader’s understanding

### Questions
1.	Clarify the theoretical connection between Section 3 and 4.
While the potential-based shaping and bias-unit views are both valid, a more direct derivation linking the two would help readers see how the bias term implements the shaping correction in function approximation.
2.	Add computational analysis.
While the method introduces negligible overhead, including runtime or memory comparisons would concretely support claims of “minimal cost.”
3.	Extend to policy-based methods.
Since reward centering parallels baseline subtraction in actor-critic methods, exploring whether similar bias-units could be applied to the critic in PPO or A2C would be a natural extension.

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
3

### Summary
The paper proposes a modification of differential TD methods based on reward centering in episodic problems. First, the authors provide the modification in the differential TD algorithm in the presence of the terminal state. Further modification in the learned loss is done. Finally, the authors evaluate their proposed differential TD algorithm (under episodic reward scenarios) in the tabular setting and Atari and Mujoco scenarios with the integration of the actor-critic algorithm. Experiment results show that the efficacy of reward centering is prevalent in the 'painful' reward setting, where the reward is not normalized.

### Strengths
1) The motivation to use reward centering makes sense. Reward centering can facilitate fast adaptation of the base RL algorithm under unnormalized rewards.

2) The paper is not hard to follow. I believe the authors thoroughly extended the reward centering to the condition of episodic reward.

### Weaknesses
1) My biggest concern about the paper is its significance. The method can be efficient in some tailored scenarios, but the gap becomes low in the sparse reward scenario. Furthermore, there are myriad of methods that extend over Q-Learning and the AC algorithm. However, the paper only includes Q-Learning and actor-critic, which can be seen as outdated. More relatively recent methods [1,2,3] should also be compared.

2) I am not sure of the contribution of the paper. The author's work extends differential TD learning from an infinite-horizon scenario to the scenario where there terminal state exists, which feels like a straightforward extension. Neither does the significance of the result, connected to 1).

3) I also do not understand the motivation for extending differential TD learning methods to episodic tasks. Reward centering can be effective in no non-terminal state. However, if the process have terminal state, what is the efficacy of using complex reward centering instead of advantage function?

4) Enhanced exploration method [4] or efficient update through experience replay [5] also greatly contributes to the sample-efficient update of reinforcement learning methods. The gain of the proposed episodic version can be diminished under the enhanced exploration or replay-based update.

***References***

[1] Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, ICML 2018

[2]  Rainbow: Combining improvements in deep reinforcement learning, AAAI 2018.

[3] Asynchronous Methods for Deep Reinforcement Learning, ICML 2016.

[4] Exploration by Random Network Distillation, ICLR 2019.

[5] Revisiting Fundamentals of Experience Replay, ICML 2020

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

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
The paper studies extension of differential TD-learning to episodic case. In the infintie horizon problem, average reward is often estimated and subtracated in the TD-learning udpate, which the authors call reward centering mechanism. This becomes a problem in episodic case as subtracting constant reward can cause early termination or non-terminating behavior. The invariance is achieved by considering a proper reward shaping at the terminal state. The paper approximates the value function with a differential value function and an additional bias unit which does not depend on state-action, and reflects the free variable in the reward shaping method.

### Strengths
1. The paper considers an unexplored problem in the community, which is average reward learning in episodic case. While the community has focused on the infinite horizon case, the paper considers a different scenario which is finite horizon case (episodic).

2. The authors manage to derive a new algorithm in average reward learning under episodic scenario. The derivation which relies on reward shaping seems to new.

2. The claims are supported by experimental results. The proposed method seems to be effective in the streaming learning scenario.

### Weaknesses
1. The authors do not propose a theoretical convergence result for their update equation.


2. The mean-squared formulation and the parametrization $V(s;w,b)=V^{\Delta}(s;w)+b\approx v^{\pi}(s)$ is not straightforward. Moreover, the motivation for the reparameterization of $\hat{b}=(1-\gamma) b$ and $\hat{\eta}=\eta(1-\gamma)$ is not clear.

3. skepticism on the problem formulation from theroetical perspective: The challenge of average reward problem is in its unboundedness of $\sum_{t=0}^{\infty} r_t$. Nonethelss, if we consider an episodic case, we always have bounded sum of rewards. With boundedness, we maybe possible to guarantee many nice properties as in the discounted infinite horizon problem. Therefore, it is questionable, whether considering an episdoic problem in the average reward is meaningful in theoretical sense.

### Questions
1. The intuition on introducing additional bias unit is not clear ($V^{\Delta}(s;w)+b$). 
2. In line 242, can the authors provide more detail on the choice of $\eta$ and $b_0$?

### Soundness
3

### Presentation
2

### Contribution
2
