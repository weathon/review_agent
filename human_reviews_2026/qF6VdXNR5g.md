# Strategic Deception in Deterministic Markov Decision Processes via Value Differences

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
We investigate the design of autonomous deceptive agents capable of deceiving observers while executing tasks in deterministic and complex environments. Recent research has introduced an intent recognition model based on Q-differences and the ambiguity model (AM), which selects actions ambiguous over reward functions using pre-trained Q-functions to mislead observers. However, we identify that AM fails to achieve effective deception in deterministic Markov decision processes (DMDPs) because the strategy of maximizing entropy at each step leads to a large number of ineffective deceptive behaviors in the later stages of the task when the intention has been revealed. To address this problem, using the existing intent recognition based on state value differences (V-differences), we propose the concept of the last deceptive state (LDS), a method to compute the optimal LDS, and two \textit{V}-differences-based Deceptive Models (VDMs). VDMs plan deceptive trajectories in DMDPs, moving beyond the geometric constraints of traditional path planning. Experiments in path planning domains demonstrate that VDMs achieve stronger deception and outperform AM across key metrics, including trajectory cost, deceptiveness, and steps after LDS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a two-stage approach for performing deceptive planning over finite, deterministic Markov decision processes (MDPs), where the planning agent's objective is to navigate to its true goal while deceiving an external observer that is trying to infer the agent's true goal from amongst a set of candidate goals.In the first stage of the approach, the agent behaves deceptively up until a to-be-determined "last deceptive state" (LDS); in the second stage, the agent follows the minimum-cost path from the LDS to the true goal. The deception metrics used to perform deception combine the notions of exaggeration and ambiguity from previous works on DPP ([Masters & Sardina, 2017], [Savas et al., 2022]) with the maximum entropy-type observer model developed in [Ziebart et al., "Planning-based prediction for pedestrians", 2009], [Ziebart et al., "Modeling interaction via the principle of maximum causal entropy", 2010] and previously used for DPP in [Savas et al., 2022], [Chen et al., "Deceptive planning for resource allocation", 2024], and [Suttle et al., "Value of information-based deceptive path planning under adversarial interventions", 2025]. The notion of LDS is very closely related to the notion of "last deceptive point" developed in [Masters & Sardina, 2017]. The term "Value-differences-based deceptive model", or VDM, refers to the overall two-stage approach combining the maximum entropy observer model (referred to as the "V-differences" approach in this work) with previous deception metrics. A key motivation for the two-stage nature of the VDM approach is to avoid the inefficient deceptive behavior observed in the later stages of DPP trajectories generated using the Ambiguity Model (AM) for deception of [Liu et al., 2021], when deception is no longer useful and minimum cost planning is most efficient. Experimental results on gridworld problems are provided that validate the performance of the proposed approach against AM and indicate robustness to the deception hyperparameter of the VDM method.

### Strengths
The idea of separating the DPP process into two phases -- a deceptive phase and a min-cost/shortest-path phase -- is natural and intuitively appealing. Developing a formalism that provides a principled way to divide and structure the two phases is thus well-motivated. The proposed use of LDS as the dividing point between the two phases and the specific VDM variants proposed provide reasonable instantiations of such a formalism. Furthermore, this paper bridges two distinct lines of work on deceptive planning -- (i) the control-/planning-based approaches of [Ornik & Topcu, 2018], [Savas et al., 2022], and their successors; (ii) the DPP work of [Masters & Sardina, 2017] and its successors -- that, to my knowledge, no previous work has made a serious attempt to combine. Bridging these two lines of research is a worthy objective and is definitely of interest to their respective deception communities.

### Weaknesses
The paper suffers from the following weaknesses:
1. The LDS definition is highly similar to the "last deceptive point" (LDP) notion of [Masters & Sardina, 2017]. The primary difference appears to be the use of the observer model of [Savas et al., 2022], [Ziebart et al., 2009] to enable practical belief computation. This undermines claim (2) from the introduction.
2. Since the MDPs are finite and dynamics are deterministic, the underlying problem can be viewed as a standard DPP problem, as considered in [Savas et al., 2022] and [Masters & Sardina, 2017]. The setting considered in this work is on the one hand a special case of the general setting considered in [Savas et al., 2022], and on the other it is unclear how the LDS notion defined in this work is truly distinct from the LDP notion of [Masters & Sardina, 2017]. This undermines claim (3) from the introduction.
3. The proofs of the theoretical results follow immediately from the definitions, and the main theoretical contribution is thus the set of definitions proposed in Section 4. The theoretical contribution is thus minor, as it consists primarily of defining LDS and related objects, from which the main results follow easily.
4. The solution methods of [Savas et al., 2022], which apply to finite MDPs with stochastic dynamics, apply to the deterministic MDP setting considered in this work as a special case. Furthermore, the trajectories generated by these solution methods do not exhibit the wasteful late-trajectory behavior seen in the AM method of [Liu et al., 2021]. The methods of [Savas et al., 2022] are thus important baselines to compare with, yet the current work only compares with AM. This makes it difficult to assess the effectiveness of the proposed approach against these important and highly relevant baselines, weakening the experimental contribution of the paper.

### Questions
1. How does the proposed approach move "beyond the geometric constraints of traditional path planning" as claimed around lines 22-23?
2. In what way is the deterministic MDP setting addressed in your work more general than the problem setting considered in [Masters & Sardina, 2017]?
3. Is the deterministic MDP setting you consider a special case of the stochastic MDP setting considered in [Savas et al., 2022]? If not, how?
4. Definition 1 appears to assume an infinite-horizon discounted MDP, while the value function definition of line 158 has finite horizon $T$. Which setting do you consider?
5. In what way does the observer model described in Section 3, which holds for stochastic problems in general (see [Ziebart et al., 2009-2010], [Savas et al., 2022]), exhibit "natural alignment with deterministic MDPs", as claimed on lines 167-168?
6. The motivation of LDS in Section 4 appears to rest on the *assumption* that a two-phase approach is best or at least preferable to other possible approaches; is this accurate, or is there an *a priori* reason why the two-phase approach is best/preferable?
7. Can you elaborate on the exact differences between the LDS of Definition 4 and the LDP of Definition 6 in [Masters & Sardina, 2017]?
8. How are Definitions 6 and 7 different?
9. On lines 314-315 it is claimed that you "establish that the optimal decoy goal state is $g^*$" -- where is this established?
10. Around line 317 it is stated that "$g^* $ possesses the strongest deceptive potential among all fake goal states at $d^*_c$" -- where is this shown?
11. Aside from the use of the maximum entropy observer model developed in [Ziebart et al., 2009-2010] and used for DPP in [Savas et al., 2022], what are the primary differences between the deception notions considered in Section 5 and those considered in [Masters & Sardina, 2017] and [Liu et al., 2021]? How do the notions of exaggeration and ambiguity that you use relate to the corresponding notions in [Savas et al., 2022]?
12. Given that the observer model of Section 3 is a core component of your approach and is based on [Savas et al., 2022], it would be useful to compare your approach with the methods proposed in [Savas et al., 2022] -- are there any issues preventing such a comparison?

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
4

### Summary
The paper presents a framework for deceptive planning in Markov decision processes (MDPs) with deterministic transition functions. Specifically, instead of defining the deception over reward and Q functions, the authors define the likelihood of a goal state based on differences of the value functions (V-differences). Then, they develop a concept called the last deceptive state (LDS), a method to compute the optimal LDS, and deceptive models. The approach can compute trajectories with a lower cost compared to an existing method called the ambiguity model, while having a much lower likelihood of reaching the targeted goal state through the trajectory.

Overall, the paper develops a new deceptive planning concept based on LDS instead of having it as entropy-based, which improves efficiency in terms of the final cost, as it does not rely on acting randomly to increase entropy. 

I believe the paper develops an interesting framework for deceptive planning, and the approach is theoretically sound and improves over an existing deceptive model, AM, in practice. I believe the paper can be accepted with minor revisions.

### Strengths
Overall, the paper develops a new deceptive planning concept based on LDS instead of having it as entropy-based, which improves efficiency in terms of the final cost, as it does not rely on acting randomly to increase entropy. The metholodgy is sound and the practical examples show that the resulting trajectories look more reasonable while still being deceptive compared to the honest trajectories.

### Weaknesses
The framework requires computing the value functions for each state and action in the MDP. It also assumes perfectly rational observers, which may be restrictive in terms of applying the framework.

### Questions
What is the key intuition behind introducing LDS compared to existing concepts like the Last Deceptive Point (LDP), which the authors mentioned in the related work section? How does LDS improve upon or generalize the LDP conceptually and practically?

Could the authors explicitly describe how the LDS formulation directly addresses the inefficiency of AM? I believe it's because of not needing to act randomly to increase entropy, but the authors can further clarify it.

The authors assume that their formulation assumes a perfectly rational observer, i.e., \alpha=0, which makes the optimal policy deterministic, and the value function does not need to have a softmax operator over actions. How would the framework change if \alpha > 0, and is LDS guaranteed to exist with \alpha > 0?

The authors proved that an LDS always exists. Is the LDS unique, or can it change based on the agent's policy and trajectory?

What is the computational complexity of LDS in terms of the number of goal states and the states in the MDP? Finally, is it possible to approximately compute LDS in large state spaces, or does the current algorithm only support computing LDS with precomputed value functions?

Are there theoretical guarantees on deception strength or cost ratios for either variant, i.e., can we bound the increase in trajectory cost relative to the honest policy?

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
This paper introduces the V-differences-based Deceptive Models (VDMs), a novel framework for creating autonomous agents capable of strategic deception in Deterministic Markov Decision Processes (DMDPs). They addresses a critical limitation of previous approaches, such as the Ambiguity Model (AM), to provide a principled balance between the goals of deception and efficiency.

### Strengths
- The authors identify that AM's strategy of blindly maximizing entropy at every step leads to a large number of redundant and ineffective deceptive actions in the later stages of a task, resulting in significant efficiency loss once the agent's intention is effectively revealed. 
- They introduce the Last Deceptive State (LDS), which serves as an intermediate transitional state that clearly separates the agent’s strategy into two phases—deceptive and truthful. The work is theoretically grounded and explained
- They show results in path planning environments showing that VDMs outperform AM.

### Weaknesses
-  Although the work is effective in deterministic settings, the proposed models are not easily extendable to stochastic or partially observable environments. 
- I am not sure how applicable these findings are to the real-world where there is a lot of uncertainty and more complex environments
- The formalism assumes a a fully rational observer which also limits real word application, as humans often behave sub-optimally

### Questions
- Why did you choose value differences (V-differences) instead of Q-differences as the foundation for deception modeling?
- How does your definition of deception align with human's intuitive notion of deception?
- Are there real-world settings where your system would be applicable?

### Soundness
2

### Presentation
3

### Contribution
2
