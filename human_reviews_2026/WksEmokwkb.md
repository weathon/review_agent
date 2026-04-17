# SpoilDICE: Safe and Performant Offline Imitation Learning from Dual Demonstrations

- Decision: Reject
- Scores: 4, 8, 2, 2

## Abstract
Ensuring both high returns and strict safety guarantees remains a fundamental challenge in imitation learning (IL). Existing approaches often rely on manually specified reward or cost functions, which are difficult to design and rarely capture complex safety constraints in real-world settings. We tackle this issue by introducing the problem of imitation learning from safety–performance dual demonstrations, where training data naturally divides into (i) safe demonstrations that respect safety requirements but may be suboptimal, and (ii) performant demonstrations that achieve high returns but may violate safety. To address the problem in the offline setting, we propose SpoilDICE (Safe–Performant Offline Imitation Learning via stationary DIstribution Correction Estimation). SpoilDICE integrates DICE-based distribution matching with support constraints derived from safe demonstrations, enabling agents to exploit high-return behaviors while remaining within safety-compliant regions of the state–action space. We validate our approach in both tabular gridworld and continuous safety-critical domains from the DSRL dataset and Safety Gymnasium benchmark. Empirical results demonstrate that SpoilDICE consistently produces policies that achieve strong performance without sacrificing safety, substantially outperforming prior offline IL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an imitation learning method to learn a safe and performant policy from one safe dataset and one performant dataset. The method is built based on stationary distribution correction estimation (DICE) and it aims to minimize divergence with performant data while keeping the state-action distribution within the support of safe dataset. The authors compare their method with other imitation learning baseline on safety gymnasium environment.

### Strengths
- The paper is clearly written and easy to follow.
- The motivation of this paper is interesting, which aims to achieve two objectives from two types of datasets (safe and high reward). However, this paper mainly discusses the case that two datasets have an overlap. See question for more details.

### Weaknesses
- The contribution of this paper is incremental. The key formulation and optimization objective in sec. 4.2 are mainly based on previous DICE paper (e.g., [1]).
- In experiment, the task selection is questionable. In table 1, the pointcircle tasks and the "velocity" tasks are very simple: e.g., the safety constraint of "velocity" tasks is that the robot should not run into the region out of a straight lane. In other word, they only require the robot to run straightly and there is no significant reward-cost trade-off. I believe the authors shold consider other tasks such as Goal, Button in safety gymnasium.
- Meanwhile, even in the selected tasks, the proposed method does not show consistent advantage: in most circle and velocity tasks, the reward return is close to DemoDICE; in humanoid and ShadowHandOverSafeJoint, the propsed method also cannot learn a safe policy.

### Questions
- One underlying assumption of the proposed method is that there is a (substantial) overlap between the safe dataset and the performant dataset, as illustrated in fig.1. This means there are enough safe data with high performance. What if there is no overlap or the overlap is very small?
- The authors collect offline datasets by themselves. Why don't you use public dataset and benchmark such as [2]? It will also be easier to compare with more baselines.

[1] Optidice: Offline policy optimization via stationary distribution correction estimation

[2] Datasets and Benchmarks for Offline Safe Reinforcement Learning

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the problem of safety in offline imitation learning, where agents often end up being either safe or performant but rarely both. The authors introduce SpoilDICE, inspired by DICE-style distribution matching, to tackle this issue. They propose a new setting, offline IL-SP, where the available demonstrations are heterogeneous: a set of safe but potentially sub-optimal trajectories and a set of performant but possibly unsafe ones. The goal is to learn a policy whose stationary distribution matches the performant distribution while remaining within the support of the safe distribution. The method builds upon constrained MDP machinery and introduces distributional support constraints, employing two classifiers to estimate density ratios for the safe and performant datasets. Experiments in gridworld and continuous control environments (Safety Gymnasium) show that the proposed method achieves improved safety-performance tradeoffs compared to baselines.

### Strengths
* This is a timely and important contribution addressing safety in imitation learning without relying on external reward or cost functions.
* The formulation of the offline IL-SP setting is practically motivated and, to the best of my knowledge, novel.
* The derivations are clear. I particularly appreciate the explanation around Eq. 3 and the following (non-labeled) equation.
* The experiments cover a lot of ground and paint a convincing picture of the superiority of the approach when it comes to striking trade-offs between performance and safety. Being a novel setting (to the best of my knowledge), the baselines are limited, but the authors introduce good candidate baselines for salient comparisons.
* The availability of code (hosted anonymously) is highly appreciated.
* The use of unlabeled data to estimate density ratios with pre-learned classifiers, while grounded in prior works, in an ingenious and elegant way to evaluate the a priori complex objective derived throughout the paper.

### Weaknesses
* L77-78, “further leverages unlabeled data”. That should have been brought up earlier in the presentation, e.g. as you introduce the two other (labeled) dataset SpoilDICE employs. The use of three data sources (safe, performant, unlabeled) only becomes explicit in Sec. 4.2.
* L100-107, “offline IL” in related work section. I would be helpful to have a clearer and transparent picture of what prior art methods in safe offline IL assume access to. While SpoilDICE does not have any extra signal regarding performance or safeness, it has access to two datasets, each close to optimal along those axes (one near-optimal but occasionally possibly unsafe, the other safe but likely sub-optimal). It should be made clear precisely what the prior approaches assume, particularly when those need to discover what to imitate and what to avoid with different supervision. At present, the related work section does not clearly delineate just how strong of a signal is provided to the SpoilDICE agent through the two labeled datasets.
* About the code: in the “imitation” subfolder, I was only able to access the “spoildice.py” file. All the other files showed the message “The requested file is not found”.
* Fig 2b: it would be useful for the authors to explain how to read the plots.

### Questions
* In related work: considering the similarity in objective design between DICE/DemoDICE and SpoilDICE, it would be valuable to have a more detailed discussion on those simalirities, and where exactly SpoilDICE departs from the original DICE track as the authors derive the learning objective from the initial learning desideratum. Could the authors expand the discussion comparing SpoilDICE to DICE/DemoDICE, particularly focusing on the structural differences in the objective and constraint formulation?
* About subsampling: I would encourage the authors to clarify that “subsampling trajectories” mean, in the context of the paper, “taking out a subset of full trajectories from the dataset”. Is my understanding correct? Subsampling trajectories would most likely be interpreted as taking a subset of transitions from trajectories by many readers. It is particularly relevant to be clear about that point in a method building on the Bellman flow constraint like SpoilDICE does, since such methods fail when expert data are temporally subsampled or disconnected (cf. “Rethinking ValueDice: Does It Really Improve Performance?” by Li et al. 2022). It is easy to see how in the provided code: in the file “spoildice.py” L402-412, we see that the nu loss requires (state, action, next state) triplets, as opposed to state-action pairs.

Style, typos, suggestions:
* Right before going into the experiments, it would be valuable to recap what are the modules that are learned, in what order, and briefly how. At minima, there should be a link to an algorithm block presented in the appendix. What should be moved to the main text is how the actual reactive policy (the one we interact with at deployment time)  is derived from the estimated distribution and occupancy ratio. Albeit simple, the “weighted BC” objective inherited from the DICE family of methods is the center piece of the machinery developed throughout the paper and is not obvious for readers not familiar with the DICE line of work.
* [minor] End of preliminaries: it might be worth emphasizing that the cost “c” will never be visible to the agent; it is here for conceptual purposes.
* [minor] Last expression of Eq. 12: L(mu, lambda) instead of L(mu, c)
* [minor] L69-70: “it mitigates” -> “it can mitigate” (difficult to guarantee no OOD upon deployment)
* [minor] L275-277: verify that the reference for adversarial training from Goodfellow is the intended one (the date of the given reference is 2020)
* [minor] L319-320: it seems that the text refers to the jointly filtered BC and not filtered BC

### Soundness
4

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
4

### Summary
The paper “SpoilDICE: Safe and Performance-Aware Decision Making under Model Uncertainty” introduces SpoilDICE, an offline imitation learning framework that learns from both safe and risky demonstrations to achieve a balance between safety and performance. Extending the DICE framework, SpoilDICE incorporates a spoiling regularizer to handle model uncertainty and prevent unsafe decisions. Evaluated on HalfCheetah, Hopper, and Walker2d tasks with dual demonstration datasets, it outperforms baselines such as Behavioral Cloning (BC), Filtered BC (FBC), Jointly Filtered BC (JFBC), and DemoDICE, demonstrating superior robustness and safer high-performance learning.

### Strengths
- The paper addresses an important and underexplored setting: learning a policy that balances safety and performance from dual demonstration datasets (safe and risky). This setting is novel, practical, and worth exploring.  
- SpoilDICE extends the DICE framework with a well-justified regularization mechanism. The regularizer has a clear and intuitive role: it penalizes overconfident or unsafe policy updates, enabling a controlled safety–performance trade-off.  
- The paper evaluates SpoilDICE on three standard continuous-control tasks (HalfCheetah, Hopper, Walker2d) and consistently shows better safety–performance trade-offs than strong baselines such as BC, FBC, JFBC, and DemoDICE.

### Weaknesses
- I believe the original learning objective is problematic. In the formulation in (2), the constraints only require that the support of the learning policy is a subset of the support of d_S, where the support is defined as the set of (s, a) pairs with non-zero visitation probability. This formulation ignores the actual values of d_S(s, a) and only considers whether they are zero or not, which cannot guarantee a truly safe policy. For example, even if all d_S(s, a) > 0 and the constraint supp(d_π) ⊆ supp(d_S) is satisfied, the learned policy could still assign higher probabilities to unsafe actions, leading to unsafe behaviors. Therefore, I find the main learning objective unconvincing.  

- All the derivations in Section 4.2 are rather straightforward and follow directly from standard results in the DICE framework, offering little novelty.  

- Section 4.3 describes a method to estimate the indicator function 1_{d_S}, which I find unclear and unconvincing. From a high-level perspective, such values are difficult to estimate meaningfully. For instance, when d_S(s, a) takes very small values, it is ambiguous whether they should be treated as zero or non-zero. In practice, it is possible that all d_S(s, a) values are non-zero (some being extremely small). In such cases, the use of the indicator function and the proposed estimation method becomes questionable and potentially meaningless.  

- The evaluation focuses only on three MuJoCo continuous-control tasks (HalfCheetah, Hopper, Walker2d), which is quite limited. More diverse tasks or environments should be considered to strengthen the empirical evidence.  

- I also suggest including comparisons with more direct and relevant baselines such as SafeDICE, DWBC, or COPTDICE (where costs can be manually assigned as 0 for unsafe and 1 for safe) to better contextualize the contribution.

### Questions
1. How does the proposed learning objective in Equation (2) ensure safety beyond the simple support inclusion constraint (supp(d_π) ⊆ supp(d_S))?  
2. What is the practical justification for using the indicator function 1_{d_S}, and how can it be estimated meaningfully when d_S(s, a) values are very small but non-zero?  

Please all address the concerns I raised in the Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers an imitation learning setting with dual demonstrations, where data contain safe but possibly suboptimal demos and performant but potentially unsafe demos. It proposes SpoilDICE, which augments DICE-style stationary distribution matching with support constraints from safe demos, so the learned policy could exploit high-return behaviors while remaining in safety-compliant regions; experiments on gridworld and Safety Gym show performance improvement without sacrificing safety, outperforming other IL baselines.

### Strengths
- The paper is well written and straightforward to follow.
- Sound, simple method. SpoilDICE = DICE with a safe-support constraint, yielding a clean Lagrangian, tractable optimization, and a closed-form density-ratio update.
- Consistent results. Experiments (diagnostic + Safety Gym) show strong returns without sacrificing safety, with helpful ablations and data-efficiency analyses.

### Weaknesses
- Unrealistic data assumption. The IL-SP setting assumes the existence of unsafe yet high-performing demonstrations; in practice, such data typically do not exist (and are not collectable by design) in safety-critical pipelines, making the premise largely hypothetical.
- Limited novelty. The method reads as an incremental extension of DICE—adding a safe-support constraint and PU estimation.
- Compared to the baseline DemoDICE-U, improvements are minimal.

### Questions
In addition to weaknesses:
- Could you provide concrete real-world scenarios or datasets where unsafe yet high-performing demonstrations actually exist for offline imitation, and evidence that these demos indeed yield higher return than the safe ones?

### Soundness
2

### Presentation
2

### Contribution
2
