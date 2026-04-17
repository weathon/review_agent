# Maximum-Entropy Exploration with Future State-Action Visitation Measures

- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Maximum entropy reinforcement learning motivates agents to explore states and actions by providing intrinsic rewards proportional to the entropy of some distribution. In this paper, we study intrinsic rewards proportional to the entropy of the discounted distribution of state-action features visited during future time steps. This approach is motivated by two results. First, we show that this new objective is a lower bound on the standard objective providing intrinsic rewards proportional to the entropy of the discounted distribution of state-action features visited during full trajectories, i.e., starting from initial states. Second, we show that the distribution used in the intrinsic reward definition is the fixed point of a contraction operator. The intrinsic reward can therefore be computed off-policy. We quantify and compare the exploration effectiveness of different maximum entropy objectives. Experiments highlight that the new objective leads to feature exploration concurrent to the alternative methods. In expectation over trajectories, features are typically visited less often, as suggested by the lower bound, but over individual trajectories, features are visited more often than the concurrent approaches. All methods lead to similar control performance on the considered benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new maximum entropy reinforcement learning (MaxEntRL) objective that encourages exploration by maximizing the entropy of conditional discounted future visitation distributions over state-action features. Unlike classical MaxEntRL approaches that either maximize policy entropy or the entropy of marginal visitation measures, this work focuses on the conditional visitation measure starting from each state-action pair. The new objective is shown to be a lower bound on the standard marginal visitation entropy objective, thereby retaining desirable exploration incentives while potentially simplifying computation. The conditional visitation distribution is the unique fixed point of a contraction operator, enabling off-policy estimation of intrinsic rewards via N-step bootstrapping rather than expensive on-policy rollouts. On the algorithmic side, the authors adapt soft actor-critic  to include this new intrinsic reward, learned through an auxiliary visitation model trained with a cross-entropy objective. Empirical evaluation on MiniGrid environments compares three exploration strategies-policy entropy, marginal visitation entropy, and the proposed conditional visitation entropy — using both exploration metrics (feature entropy over episodes and within trajectories) and task performance. Results show that the new objective improves within-trajectory exploration and sometimes matches or outperforms marginal visitation methods in terms of entropy, while maintaining similar control performance.

### Strengths
1. The paper is well-structured, with clear definitions of visitation distributions and intrinsic reward functions.
2. The lower bound theorem (Thm 3.2) provides a theoretical justification,  relating the conditional and marginal objectives.
3.  The contraction property (Thms 4.2–4.5) is a strong technical result that underpins the off-policy learning advantage.
4. The approach provides a middle ground between purely on-policy entropy methods and more complex state-visitation-based objectives.
5. Off-policy compatibility addresses a well-known bottleneck of previous state-visitation entropy methods, which are typically on-policy and sample-inefficient.

### Weaknesses
1. All experiments are conducted on MiniGrid, which is discrete, small-scale, and relatively simple compared to widely used continuous control benchmarks (e.g., Mujoco, DM Control Suite). It is unclear how the method scales in high-dimensional continuous state-action spaces, especially regarding density estimation.
2.  The implementation drops importance weighting and other terms, introducing bias in the visitation model estimation. The practical impact of this bias is not fully analyzed.
3. While exploration improves (particularly within trajectories), final task returns are comparable to baseline MaxEntRL methods. t remains to be shown whether improved exploration translates into clear performance benefits in more challenging settings.
4.  While the method is simpler than marginal visitation estimation, the additional training of a visitation model still adds overhead. No detailed runtime or sample efficiency comparison is provided.

### Questions
1. How does the method perform on environments with large or continuous state-action spaces (e.g., HalfCheetah, Ant)? Which parts of the algorithm (e.g., density estimation, model learning) would need to be adapted ?
2. The paper shows the theoretical lower bound but does not empirically analyze how close the conditional and marginal objectives are during training. Can you provide such an analysis ?
3. How significant is the approximation bias introduced by neglecting importance weights in Eq. (12) ? Could variance reduction techniques help retain theoretical guarantees ?
4. Could the feature space Z be learned jointly with the policy (e.g., using contrastive learning or predictive representations) instead of being predefined?
5. How does the training time and sample efficiency compare to marginal visitation methods, especially when scaling up N or the environment complexity?

### Soundness
3

### Presentation
3

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
1. The author replace the entropy term H during full trajectories into H during future time steps.
2. Use this entropy term as intrinsic reward and propose a new method
3. "All methods lead to similar control performance on the considered benchmarks."

### Strengths
The author provide a theoretical analysis for the proposed method.

### Weaknesses
The paper is poorly written and contains unclear language and disorganized structure.

1. **Use of Eq (4) for Intrinsic Reward Formulation:**  
   Why is Eq (4) chosen as the intrinsic reward formulation? How can this formulation be equivalent to the MaxEntRL or the particle-based entropy estimation method (Hazan et al., 2019) mentioned in line 14?

2. **Explanation of Variables and Distributions:**  
   The explanation of variables and distributions is inadequate. What do \( \bar{s} \) and \( \bar{a} \) represent? What is the feature function \( h \)?  
   Furthermore, the term \( q_\pi(z|s, a) \) appears multiple times in Section 2.2, but its definition is only provided in Section 3. It seems these are different entities, which causes confusion. It would be helpful to provide the definition earlier and give an intuitive explanation for clarity.

3. **Experimental Results on Grid Experiment:**  
   The experimental results on the Grid experiment do not demonstrate superior performance. The results do not provide sufficient evidence to support the claims of the paper.

4. **Baseline Comparison:**  
   The baseline includes only the SAC algorithm, but there are many exploration methods that should be considered, such as ICM and NGU. For entropy-driven methods, the particle-based entropy estimation method should also be included as a baseline.

Overall, the paper is poorly written, making it difficult to extract useful information. The method, algorithm, and contributions lack clarity. I recommend rejecting this paper and suggest that it be revised and resubmitted later after addressing the above issues.

### Questions
See Above.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new intrinsic reward for exploration in RL, which combines the commonly used action entropy and another entropy term on the conditional states visitation. Implementing this intrinsic reward requires to model the state visitation, here obtained through a TD-like methodology in a discounted setting. The intrinsic reward can be incorporated into any discounted RL algorithm, e.g., SAC, for improved exploration on the state space. The resulting method is evaluated against vanilla SAC and another baseline maximizing the entropy of the marginal state visitation in some sparse-rewards environment from MiniGrid.

### Strengths
- The paper provides an elegant unification of different "MaxEntRL" approaches by formalizing the intrinsic reward with a separate feature space, which can alternatively be the action space for standard action entropy incentives or the state space for state entropy exploration;
- The paper provides an original version of the state entropy exploration objective that only looks at the entropy of the future discounted state (or state-action) distribution conditioned on the current state;
- The paper provides a few theoretical results showing that the introduced intrinsic reward is the fixed point of a contractive operator and that it constitutes a lower bound of the more common marginal state visitation entropy objective.

### Weaknesses
- The paper seems to mischaracterize existing state entropy algorithms as inherently on-policy. While several of the existing implementations are used on-policy, they can be easily adapted to work off-policy;
- After reading the paper and looking at experimental results, why conditional entropy shall be preferred to marginal entropy is largely unclear to me;
- The paper does not much to clarify why MiniGrid has been chosen to compare the performance of conditional visitation entropy with prior work, which typically consider much more challenging domains, such as Mujoco or Atari;
- The paper circumvent the page limit by placing all the plots in the appendix, although almost a full page of the main paper is left blank, so that I would consider this mostly as a poor formatting decision rather than a style violation.

**EVALAUTION**

While the paper is original and promising, I think the paper lacks strong conceptual and empirical ground to support the proposed intrinsic reward. Advancements over prior works shall be clarified to meet the bar for acceptance.

### Questions
Other than the weaknesses mentioned above, I have two additional comments the authors may consider in their response.

1) Is conditional visitation better than marginal in some sense?

After reading the paper, I am not convinced that conditional visitation is better than marginal for the purpose of pre-training a policy to be used for efficient learning of a downstream task. If the abstract objective is to induce even visitation of the states (and actions), I would argue that the past also matters. Let us look at this simple example: We have two rooms (left and right) connected by a corridor. The agent starts in the middle of the corridor with the goal to maximize the conditional entropy. Let's say the agent start exploring the left room before going back to the corridor. Now, since the objective is conditioned on the current state and action and only looks at the future, visiting the right room or the left room again is equivalent. Thus, a policy that visits the left room repeatedly is optimal for the conditional objective. Instead, a policy maximizing the marginal state visitation, will necessarily randomly choose between going left or right when in the middle. If the policy is history-based, it shall choose to go left and right depending on what has been previously explored. In this example, the marginal entropy looks much more aligned with the abstract exploration objective. I am wondering if there are other examples where the conditional entropy is better.

2) On-policy vs off-policy methods

The marginal state entropy can also be optimized off-policy. In various implementations, e.g., see Abbeel et al., 2021 or Seo et al., 2021, the objective is cast into an intrinsic reward that can be optimized with SAC, more or less as it is done in the experiments here. So i do not understand why the paper claims that the conditional visitation opens the door to off-policy methods while marginal state entropy doesn't. Perhaps a more supported claim would be to note that conditional entropy is the fixed point of a contractive operator, while a similar result is not known for marginal state entropy.

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
2

### Summary
The paper introduces a new MaxEntRL intrinsic reward based on the conditional state-action visitation probability and the conditional state visitation probability.  This objective extends classical one where Shannon entropy of policy is considered (feature space is action space). The paper proposes an off-policy learning algorithm integrating this reward into Soft Actor-Critic (SAC). Empirical results on MiniGrid environments, comparing three exploration strategies (policy entropy, marginal visitation, conditional visitation) are provided.

### Strengths
Integrating visitation distributions into MaxEntRL is an interesting idea. There are theoretical results (e.g contractive properties and KL lower bounds) and numerical experiments on MiniGrid environment.

### Weaknesses
1. The function $h$ is central but under-specified. How to choose or parameterize it for high-dimensional states is unclear
2. Only small, discrete MiniGrid environments are tested. No continuous-control task are considered.  The current experiments rely on hand-crafted discrete features (agent position) limiting generality.
3. There is no intuitive explanation of why the lower bound of Th. 3.2  is meaningful for exploration or what properties it preserves. What is L in this theorem. It would be good to specify if it is an absolute constant. 
4. The paper doesn't provide any theoretical performance or sample-complexity advantage from using different maximum entropy objectives. All the theoretical results obtained in the paper are generally trivial.

### Questions
1. There are no assumptions in the formulations of the main theorems. Do I understand correctly that the theorems are always fulfilled for any spaces of states and actions?
2. The authors report that “all methods lead to similar control performance.” What is the practical impact in this case?

### Soundness
2

### Presentation
2

### Contribution
2
