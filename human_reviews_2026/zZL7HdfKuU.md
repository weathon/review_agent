# RAD: Retrieval High-quality Demonstrations to Enhance Decision-making

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Offline reinforcement learning (RL) learns policies from fixed datasets, thereby avoiding costly or unsafe environment interactions. However, its reliance on finite static datasets inherently restricts the ability to generalize beyond the training distribution.
Prior solutions based on synthetic data augmentation often fail to generalize to unseen scenarios in the (augmented) dataset.
To address these challenges, we propose Retrieval High-quAlity Demonstrations (RAD) for decision-making, which innovatively introduces a retrieval mechanism into offline RL. Specifically, RAD retrieves high-return and reachable states from the offline dataset as target states, and leverages a generative model to generate sub-trajectories conditioned on these targets for planning. Since the targets are high-return states, once the agent reaches such a target, it can continue to obtain high returns by following the associated high-return actions, thereby improving policy generalization. Extensive experiments confirm that RAD achieves competitive or superior performance compared to baselines across diverse benchmarks, validating its effectiveness. Our code is available at https://anonymous.4open.science/r/RAD_0925_1-690E.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RAD, a retrieval-augmented framework for offline RL that enhances policy generalisation. RAD dynamically retrieves high-return and reachable states from the offline dataset as intermediate targets and employs a diffusion-based generative model to plan sub-trajectories towards them. This retrieval-guided planning allows agents to escape low-return or out-of-distribution regions and achieve higher rewards. Experiments show that RAD consistently matches or outperforms strong baselines.

### Strengths
1. The manuscript is well-structured and clearly written.

2. The motivation is strong and effectively highlights the rationale of the method.

### Weaknesses
The novelty of this method appears limited. To my understanding, many prior works in this area employ model-based approaches to augment offline datasets, such as various trajectory-stitching methods. The key contribution of this paper seems to lie in combining diffusion models with generalisation beyond the offline dataset. However, diffusion models have already been used for dataset augmentation, and other approaches have separately explored generalisation beyond offline data. As such, this paper mainly integrates these two existing directions, and its conceptual novelty may be insufficient for an ICLR paper.

### Questions
1. The authors should cite the related work properly.

2. There are several typos that should be corrected: line 159, line 201 (capital letter), Eq. 13 (subscript), line 294 (definition of $V(s_t)$), and line 406 (an extra line break).

3. I believe that the TS, SE, and PL modules are executed during evaluation, functioning similarly to real-time planning. Therefore, I am concerned about the computational cost during evaluation. Could the authors provide details on the inference time or latency required to produce a single action?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a method for improving offline RL. Their method (RAD) focuses on three components, (1) a target picker that from a state finds a reachable state with high return, (2) a step estimator to estimate how far the target position is from the current position and (3) a planner that creates a trajectory to get to the goal states. The high level idea is to train these components and use them to act in the environment. They perform experiments on offline RL tasks to compare to existing baselines and ablate the components of their method. They find that their method is able to perform as well as or better as baselines on their experiments and that their components all contribute to their final results. They also perform experiments on the generalization of their method and find it is able to reasonably generalize.

### Strengths
This work has reasonable novelty. This work isn't directly proposing a diffusion model but instead builds upon other works models and adds in a method for guiding these models to create the training data.

The experimental evidence is reasonable. They compare to many baselines on a reasonable set of experiments. I would really like to see 95% confidence intervals here (like in table 1) as without them it is harder to distinguish great results from good ones.

The actual results on the tasks perform quite well. In general this method is doing much better than the baselines.

### Weaknesses
In general I feel like the writing could be more clear and I think we are missing some information that I feel is critical. I will put my questions in the question section but I don't feel confident I understand how your method actually runs. Paragraph near 155 - This paragraph is pretty hard to read/understand, what does "transit" mean here? Do you mean a trajectory from s_t to s_t^g?.

The limitations of this method are not properly addressed. I'll put specific questions in the question part again but some limitations that I can think of. Speed - running diffusion models and querying a large dataset seems slow. Long horizon - there is a limit to your planner right so does increasing this limit cause issues with attempting to plan from your proposed state?

In general I come away from the paper with a lot of questions. I'll put them below but if you can answer them and update the paper then the weaknesses will be minimal and I will gladly raise my score to an accept since this is an interesting paper but I just don't fully understand the method and weaknesses.

### Questions
How do you actually run your policy? Do you run one step of your method, then take the action or do you create a trajectory with your planner and run it open loop? How long does this take? Is it a feasible real-time method or not?

You say that you can just follow high return actions once you get to the in distribution states but how are these actions computed? Is it still from your method? Or are we following the exact actions in the dataset? Is the method used to simply get out of OOD situations and then we run a normal policy?

Diffusion models are slow, how long does your method take with the extra search on top? If you have to query your entire state space when you run your method how long does that take? How would this deal with a large amount of offline data?

212 - So the idea is you want to find the number of steps to a new state but doesn't the feature vector contain the position in each trajectory the state is at? Is there a problem with assuming we are in the "same" location as long as the difference between states is small enough?

Small thing but make sure to check for typos as well there are a handful.

### Soundness
3

### Presentation
2

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
The paper proposes RAD (Retrieval High-quality Demonstrations), a retrieval-augmented trajectory stitching method for offline RL, which is composed of three modules as Target Selection (TS), Step Estimation(ES), and Planning (PL). Given a current state, RAD (i) retrieves similar, high-return, and purportedly reachable target states from an offline dataset; (ii) estimates the temporal distance (step span) to the target; and (iii) conditions a diffusion planner (Diffuser or DiffuserLite) to generate a sub-trajectory that steers the agent toward the target, after which the policy continues along high-return actions in the demonstration. Experiments on D4RL tasks report broadly competitive results, plus ablations on each module and a generalization test on Random datasets.

### Strengths
The proposed RAD method combines target state retrieval with diffusion-based planning in offline RL. While trajectory stitching and diffusion planners exist, RAD’s idea of adaptive target retrieval at inference time (instead of static, offline augmentation) is a meaningful design point, appealing in sparse-reward or long-horizon domains (e.g., AntMaze), where “latching onto” good sub-goals helps escape low-value regions.

The “target-then-plan” decomposition (TS → ES → PL) is conceptually clear, and may be applied to many sequence-modeling or model-free offline baselines. The presentation of three modules and the illustrations make the paper easy to follow.

The experiment done in this paper is quite comprehensive, including a solid suite of baselines, visualization, ablations, and a small distribution-shift study.

### Weaknesses
The idea is straightforward, but lacks theoretical support. Firstly, “reachability” is asserted, not guaranteed. TS currently filters by cosine similarity and high return, then picks the candidate with the longest remaining length; there is no principled guarantee that the target is reachable without collisions/obstacles under the learned dynamics—especially salient in mazes or any environment with an inconsistent transition model (e.g., a wall separating the two near states exists and has not been explored much in the dataset). The author trained an extra binary classifier to predict the connectability in Appendix D for D-RAD, but did not show that in the main text, nor did extra comparison on that. Besides, the pipeline seems deterministic once top-k are retrieved (tie-breaks aside). This can induce policy shift brittleness: if the top candidate is slightly wrong or not reachable, the planner may commit to a poor target with no alternative selection. For potential improvement, maybe introduce stochastic target sampling. Finally, the way to select high returns also seems problematic. The cumulative discounted reward from step t is defined as $v_t = \sum_{i \ge t } \gamma^i r_i$ instead of $v_t = \sum_{i \ge t } \gamma^{i-t}r_i$, which means that earlier states will always have a higher return in an environment with non-negative rewards.

Experiment Issues: Only 3 seeds and no standard deviations/confidence intervals. The table highlights entries within 0.95×MAX, which is an unusual criterion and can obscure variance. Besides, it will be clearer if an average score is presented.

The proposed RAD seems closely related to trajectory stitching-based methods like DiffStitch, but there is no further comparison and discussion in the related works. 

Minor issues:

- Typos (e.g., Line 159 “makinga”; Line 157 “TS then estimates the step” should be ES; Inconsistent capitalization for “Diffuserlite”; “Diverde” in Table 1).

- Notation Inconsistency (‘G’ is used both as the goal state in Introduction and the return in 4.1. The latter could be replaced as $R(\tau)$ denoted in equation (2); $f_e$ refers to different things in 4.2 and Appendix D)

- 4.3 presents two sub-trajectory types (noisy state-action pair vs. noisy state), but no context for which one is used.

### Questions
1. Please clarify or correct the definitions of v_t in 3.2 and anywhere else they propagate. 

2. Is ES trained with cross-entropy over H–1 classes as suggested in 4.2 or with the MSE in Eq. (14)? If the latter, how do you backprop through argmax?

3. Can you show ablation results on how much the classifier mentioned in Appendix D helps to prevent TS from selecting an unreachable target state? 

4. Why does DL-RAD underperform in Maze2D compared with other baselines?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RAD (Retrieval High-quality Demonstrations), a retrieval-augmented offline reinforcement learning framework. Instead of relying on static data augmentation, RAD dynamically retrieves high-return and reachable states from an offline dataset as intermediate targets. It then employs a diffusion-based generative model (built upon Diffuser or DiffuserLite) to generate sub-trajectories toward these targets, improving generalization beyond the dataset distribution. Experiments on D4RL benchmarks show consistent or superior performance over baselines across locomotion, navigation, and manipulation tasks.

### Strengths
1. Integrating state retrieval into offline RL is conceptually appealing, and the paper provides a clear motivation for using retrieved high-return states as adaptive guidance for policy improvement.
2. The experiments cover a wide range of D4RL tasks (MuJoCo, AntMaze, Kitchen, Maze2D) with solid baselines including model-free, model-based, and diffusion-based methods. RAD demonstrates competitive or superior performance on most datasets.

### Weaknesses
1. Several sections contain minor grammatical errors and redundant phrasing (e.g., “novelly integrates,” “makinga decision”). Figures could be improved for clarity and caption detail.
2. The distribution-shift test (training on Medium-Replay, testing with Random starts) is limited to three environments; more systematic OOD tests would strengthen claims.
3. Although the retrieval mechanism is new, the overall architecture largely reuses existing components from Diffuser/DiffuserLite, and the retrieval is applied at the state level without deep theoretical justification for its optimality or stability.

### Questions
1. How sensitive is RAD to the accuracy of the value-based ranking in target retrieval? Would errors in return estimation significantly degrade performance?
2. Can RAD handle multi-modal retrieval results (e.g., when several high-return trajectories exist but lead to different goals)?

### Soundness
3

### Presentation
3

### Contribution
3
