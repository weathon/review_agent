# Revisiting Value Estimation in Policy Gradient Methods

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Temporal-difference (TD) estimation is a central component of value estimation in reinforcement learning. However, its role within policy gradient methods has not been systematically understood. In this work, we introduce a framework grounded in the notion of well-posedness, which provides a rigorous formulation of TD estimation across a wide range of control problems and enables more accurate value estimation. Through extensive empirical studies, we further show that when policy optimization is properly formulated and combined with an appropriate bootstrapping strategy, even the vanilla policy gradient algorithm can reliably solve the problem. These findings indicate that deep reinforcement learning methods can be made more robust and interpretable with proper problem formulation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper notes that TD approximation is not guaranteed to converge to a unique solution when the ergodicity assumption is broken, and make the case that this is potentially a common problem in continuous spaces. The paper formalizes this notion in terms of well-posedness, that the solution of TD learning depends on how we assign value to episode terminations. The authors present two compelling examples of where this can be a problem, and a formal result that clarifies a sufficient condition for well-posedness. Empirical results demonstrate that a simple learning-free terminal value function can sometimes lead to better RL performance.

### Strengths
This paper presents a useful and, to the best of my knowledge, novel theoretical description of convergence in TD learning in terms of well-posedness. This work motivates its result well, describing the importance of accurate TD learning and a potential area of weakness. The formal result is strong and clear, and I have not seen it written in this language before. Also, I thought the two examples provided (the continuous one, and the simple description of how TD without termination can be under-determined) were very clever and clear. Overall, I think framing the TD learning problem this way is a valuable contribution to the field.

### Weaknesses
TD learning may be ill-posed when the values from terminal states are unconstrained or not defined, and the authors present two examples where this may be the case. However, in an episodic task with time-limits, states that are terminal for one episode may appear as part of a transition tuple in others. Therefore their values should be pinned by the bootstrapping update — and using a different “terminal reward” than value for those states could cause its own issues. This is, I would argue, the more common and realistic case. The absence of discussion about this feels like a big gap to me.

I’m not weighing this much because this is primarily a theory paper. But in the environments used for the empirical results, it’s likely that terminal states in one episode appear as intermediate states in another. And so it’s not clear that the modest improvements come from there, if the problems you tested on are also well-posed.

### Questions
Am I correct in understanding that TD is only ill-posed if some terminal states do not appear as non-terminal states elsewhere? I believe this paper would be much more complete with a discussion of this setting.

I believe that your result holds even in the case that s_T does also appear as a non-terminal state, but can you confirm this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates TD-based value estimation in policy gradient algorithms. They consider a case where there is no grounding for the value function (e.g., finite-length trajectories that don't reach a terminal state), observing that the Bellman equation has multiple solutions in such cases. They propose an alternate bootstrap target for truncated trajectories, show that it is well-posed, and validate the modification empirically.

### Strengths
* The well-posedness observation is really interesting, as it is relevant to cases where finite-length rollouts are collected from some starting state distribution. This setup is relatively common in deep RL, where an algorithm alternates between separate trajectory collection and batch-update phases. To my knowledge, it hasn't been widely acknowledged how this setup can readily break the standard assumptions made on the underlying MDP.

* The proposed modification does address the well-posedness concern in that it anchors the value estimates. The modification is somewhat heuristic, but the authors were p front about how the choice introduces an assumption on how the rewards must be consistent with the true values, and noted that this is (approximately) the case in the class of problems they considered.

### Weaknesses
* The demonstration of the three bootstrapping schemes was only done in one environment. Notably, the settings were changed from the defaults without discussion as to why. The novel "Reward" terminal modification was not investigated further, as section 6 only seems to compare PPO with the ill-posed target and VPG with the well-posed "None" scheme.

* There are considerable concerns around the empirical methodology. Only 5 seeds were used, which has been repeatedly shown to not be enough to make a proper statistical comparison for the claims being made (e.g., Henderson et al., 2017; Colas et al., 2018; Patterson et al, 2023; Patterson et al, 2024). Further, there is substantial variability in many of the plots, calling the statistical significance of the results into question—the shaded regions represent the standard deviation which is not a measure of confidence.

A minor concern that stood out but did not impact my score:

* The paper repeatedly suggests that the role of the value function in an actor-critic setup is not well understood. As worded, I'm not sure this is true, as hinted at by the paper citing evidence that the accuracy of the value estimates matter. In addition to being a baseline, values also help form an estimate of the return if TD is used, after which it is substituted where the return should be in the policy gradient theorem. Here, it is clear what role it is playing, and why it is beneficial for it to be accurate. Now, there is opportunity for handling settings when value-learing is ill-posed. That is where this paper makes a contribution, but statements suggesting that it's "unclear how the value estimation module, specifically the temporal-difference (TD) method (Sutton, 1988), operates within deep policy gradient methods" feel wrong without further specifics.

### References

* Henderson, P., Islam, R., Bachman, P. Pineau, J., Precup, D., Meger, D. (2018). Deep Reinforcement Learning that Matters.
* Colas, C., Sigaud, O., Oudeyer, P. (2018). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.
* Patterson, A., Neumann, S., White, M., White, A. (2023). Empirical Design in Reinforcement Learning.
* Patterson, A., Neumann, S., Kumaraswamy, R., White, M., White, A. (2024). The Cross-environment Hyperparameter Setting Benchmark for Reinforcement Learning

### Questions
* The well-posedness concerns seem orthogonal to their role within policy gradient methods. For example, the proposed modification is relevant to pure value-based methods as well, in setups where the value function similarly isn't anchored. I feel this makes the repeat emphasis on policy gradient methods somewhat imprecise. Could the authors clarify the choice to contextualize this in policy gradient methods?

* Can the authors elaborate the concern around time-independence? While the truncated return with no terminal reward is time-dependent, when it's applied to a non-finite-horizon problem, it is akin to optimizing over a moving window into the future from each state onward. While Figure 1 shows it performing poorly, it's unclear whether this is inherently problematic or a specific interplay with aggressive time limits. The remaining results with VPG, which I believe is using the "None" scheme, seems to be performing okay.

* Can the authors comment on the statistical significance of the results, and whether 5 seeds are enough for the conclusions drawn?

* If VPG uses a critic, advantage functions, and extra clipping for stability, is it even "vanilla" anymore? I initially assumed that VPG would be much closer to REINFORCE, as it is akin to direct application of the policy gradient theorem.

### Soundness
1

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
2

### Summary
This paper presents a re-evaluation of policy gradient methods, arguing that the accuracy and stability of value function (critic) estimation are far more crucial to performance than the specific policy (actor) update mechanisms, such as the clipping in PPO. The paper provides a theoretical framework for “well-posedness”, and its analysis and empirical validation. Empirically, the authors show that a modified Vanilla Policy Gradient algorithm, when paired with the theoretically-appropriate bootstrapping method for a given domain, can match or even outperform highly-tuned PPO baselines on challenging benchmarks.

### Strengths
- This paper challenges a core assumption in modern deep RL. The community has largely focused on the actor as the key to stability. This work provides a compelling argument that the critic and its proper formulation are the real bottleneck.
- The "well-posedness" framework, particularly the characterization of the "Critic" bootstrap as an under-determined system (Section 4), is insightful and provides a satisfying explanation for instabilities that many practitioners have likely observed.
- The experimental design is good and results are positive.

### Weaknesses
- The proposed reward bootstrapping method, while shown to be effective on ManiSkill3 tasks, assumes the reward landscape approximately reflects the ground-truth value function. This assumption holds for the selected manipulation tasks, which have well-shaped, dense rewards. Many important domains feature sparse rewards.
- The VPG implementation (Equation 10) is not quite vanilla. This "clipping" of negative advantages is a major deviation from true VPG. This modification is spiritually very similar to PPO's clipping, as it prevents the policy from being updated based on "bad" actions, which is known to stabilize training.

### Questions
-  The Reward bootstrapping method assumes the reward landscape is a good proxy for the value landscape. Could you elaborate on the specific failure modes of this method?
- What challenges would you have when applying the well-posedness framework and the proposed bootstrapping schemes to tasks with large, discrete action spaces?
- The paper builds on [1], which suggests more value updates are key. Your VPG uses 50 value steps. How does the number of value updates interact with well-posedness? Would an ill-posed method converge to the correct solution if you simply ran 1000 value updates instead of 50?

[1] Tao Wang, Ruipeng Zhang, and Sicun Gao. Improving value estimation critically enhances vanilla policy gradient. In ICML, 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper re-examines fundamentals of value estimation in continuous state space MDPs. It identifies a potential issue of the standard approach of using critics for value estimation and proposes an alternative for bootstrapping. Experiments in standard continuous control benchmarks show the benefits of the approach when combined with vanilla policy gradients compared to PPO.

### Strengths
- I appreciate the direction of the paper: Revisiting the fundamentals of the learning algorithms we use, in this case, value estimation in policy gradient methods. Better understanding the foundations can lead to more profound insights. 

- In general, theory and experiments are clearly explained. The experiments are reasonably chosen to demonstrate the claims.

### Weaknesses
- I have some concerns about example 4.1 and the discussion of uniqueness of the value function.
The paper mentions that in continuous state spaces, the Bellman equation may not have a unique solution.
As far as I know, the uniqueness of solutions to the Bellman equation comes from the Banach fixed point theorem for contraction mappings. This theorem only requires that the (function) space under considerations is a complete metric space. These conditions are satisfied by the set of bounded, measurable functions equipped with the supremum norm. Thus, applying the Banach fixed point theorem would tell us that iteratively applying the Bellman equation would give us the unique value function in the limit.

This is consistent with example 4.1 because the alternative solution given is unbounded, which is not in the set of bounded, measurable functions. So, if we start with a bounded value function estimate, we will stay within the bounded set when applying the Bellman equation and recover the unique solution, the zero function. That is, we should not be encountering any issues.

Overall, I am unconvinced of the ill-posedness of the value estimation problem.


- Experimental sections: Some experiments are only run for particular environments but it would strengthen the claims if they were run across all the ones considered. For example, evaluating the bootstrapping strategy (None, Reward or Critic) was done in Fig.5 for the IsaacLab experiments but it would be more convincing if the three were run for the other environments. The reward bootstrapping seemed like a major part of the paper. These experiments could be added to the appendix at least.

- A minor point: I feel like the organization of the paper could be improved and there seems to be a slight lack of focus. There seems to be two main threads. One about the ill-posedness of value estimation with critics and an alternative using reward bootstrapping. A second about using vanilla policy gradient as an alternative to PPO. These felt slightly disjointed at times and perhaps they could be woven together more smoothly in the paper. Alternatively, focusing on one or the other could be good.

### Questions
- Line 228: There ia discussion of time-outs in RL environments and how bootstrapping is handled. An alternative that is notably absent is augmenting the state space with the time variable for finite-time MDPs. This can be a better alternative when the time index is important to the policy [1].


- Line 252: It is claimed that the proposed reward bootstrapping reflects the ground-truth $V^\pi$. Could this be confirmed experimentally? 
- Also, it would be interesting to have a theoretical analysis. Are there conditions we could put on the reward and dynamics so the proposed method would be guaranteed to be effective? 
The opposite is clear; there are simple, finite MDPs where reward boostrapping would fail to produce the correct optimal policy.


- Line 341: Clipping to eliminate updates on negative advantages was introduced. This seems like a nontrivial addition. If this is not a commonly-used technique, some more investigation would be welcome. For example, how does ths fare compared to entropy regularization, which is common addition preventing $\log \pi$ from going to $-\infty$. 

Summary of review: \
This paper has some interesting ideas and explores some directions that I would be excited to see developed further. I am particularly interested in further investigations into value functions in continuous state spaces or alternatives to critic boostrapping. Currently, I am unconvinced of some of the main claims concerning ill-posedness of the value function and I think the experiments could be expanded to more strongly support the claims.
I would lean towards rejection at present.




Suggestions and other thoughts (no impact on score):
- It seems like there could be other interesting phenomena related to value functions in continuous state spaces. For example, [2] shows that a quite simple MDP with continuous states can lead to very complex value functions.
Perhaps it would be interesting to see if this complexity can lead to pitfalls in policy optimization? 

- This paper [3] also explored how value estimates can impact the policy optimization path and may be interesting to consider. For example, perhaps choosing R(s)/(1-\gamma) as a terminal reward may be optimistic (greater than the true value function), which could be beneficial for exploration by increasing the policy's entropy.

- Is there a reason the estimator is called the vanilla policy gradient rather than REINFORCE (with baseline)? 


[1] "Time Limits in Reinforcement Learning" Pardo et al. 

[2] "The Gambler's Problem and Beyond" Wang et al.

[3] "Beyond variance reduction: Understanding the true impact of baselines on policy optimization" Chung et al.

### Soundness
2

### Presentation
3

### Contribution
2
