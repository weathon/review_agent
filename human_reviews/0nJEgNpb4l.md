# PEAR: Primitive Enabled Adaptive Relabeling for Boosting Hierarchical Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 5, 8, 6, 6

## Abstract
Hierarchical reinforcement learning (HRL) has the potential to solve complex long horizon tasks using temporal abstraction and increased exploration. However, hierarchical agents are difficult to train due to inherent non-stationarity. We present primitive enabled adaptive relabeling (PEAR), a two-phase approach where we first perform adaptive relabeling on a few expert demonstrations to generate efficient subgoal supervision, and then jointly optimize HRL agents by employing reinforcement learning (RL) and imitation learning (IL). We perform theoretical analysis to bound the sub-optimality of our approach and derive a joint optimization framework using RL and IL. Since PEAR utilizes only a few expert demonstrations and considers minimal limiting assumptions on the task structure, it can be easily integrated with typical off-policy \RL algorithms to produce a practical HRL approach. We perform extensive experiments on challenging environments and show that PEAR is able to outperform various hierarchical and non-hierarchical baselines and achieve upto 80% success rates in complex sparse robotic control tasks where other baselines typically fail to show significant progress. We also perform ablations to thoroughly analyze the importance of our various design choices. Finally, we perform real world robotic experiments on complex tasks and demonstrate that PEAR consistently outperforms the baselines.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method to improve HRL, utilizing a few expert demonstrations. The key insight is that subgoals selected from the dataset can effectively guide exploration for the policy. An adaptive relabeling method is proposed to select the proper subsequent subgoal based on the Q value of the low-level policy. The relabeled data provides an imitation-based regularization for the high-level policy, encouraging it to output reachable, high-quality subgoals for the low-level policy. Experiments in diverse simulation robotic tasks demonstrate the effectiveness of the method.

### Strengths
1. Using a few expert demonstrations to improve goal-based HRL is promising. The proposed adaptive relabeling method for IL regularization is straightforward, well-motivated, and yields good results.
2. The paper includes some theoretical analysis.
3. The experiments cover a variety of robotic tasks, including real-world test.

### Weaknesses
1. Inconsistent definition: Section 3 states that the expert data contains only states. But in Section 4.2, the low-level regularization term uses actions from the expert data.
2. Addressing the non-stationarity issue in HRL is a main claim of the paper. However, the proposed method does not resolve this issue. The high-level policy still faces non-stationarity, as the transitions in its replay buffer, which are determined by the low-level policy, continue to change throughout the training process.

### Questions
1. Many existing works [1,2,3] also use IL loss from data to regularize high-level policy learning. What differentiates the proposed regularization method from these approaches?
2. Why can we set the threshold of Q to 0 (Section 4.1) in all experiments? I believe this hyperparameter should vary, depending on the reward function specific to each task.

[1] Pertsch, et al. "Accelerating reinforcement learning with learned skill priors." Conference on robot learning. PMLR, 2021.
[2] Shi, et al. "Skill-based model-based reinforcement learning." arXiv preprint arXiv:2207.07560 (2022).
[3] Yuan, et al. "Pre-training goal-based models for sample-efficient reinforcement learning." The Twelfth International Conference on Learning Representations. 2024.

### Soundness
3

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
PEAR combines its key feature, adaptive goal relabeling, with IL regularization (either MSE or IRL), and other tricks (e.g. margin classification objective) to beat several prior HRL methods on a standard array of tasks using expert demonstrations.

### Strengths
The authors carefully outline their construction of the PEAR algorithm, justifying the usage of each component (goal relabeling, the general algorithm, the joint optimization framework, etc) thoroughly and clearly. The sub optimality analysis provides further credence to their method. The method is novel, and notably outperforms previous HRL works (while, in some cases, removing the need for e.g. hand-made action primitives).

I agree with the author view that the significance of the work should be gauged less by its immediate improvement over other LfD methods, and more by its conceptual groundwork. In this regard, this paper is well-written, the findings are well-presented, and the extensive ablations provide further insight into key aspects of the method.

### Weaknesses
As authors note, the method is currently reliant on expert demonstrations. However, many benchmarks exist which include other kinds of demonstrations, including human teleop. While the method may not perform well on these demonstrations just yet (as it is listed as an aim of future work), providing results on suboptimal demonstrations would help demonstrate concretely the strong and weak points of authors' method, and potentially provide insights on why it fails in these settings.

Furthermore, it seems inaccurate to state that PEAR uses only “a handful” of demonstrations, when Fig. 13 shows that generally 50-70+ demonstrations are needed to solve the provided tasks (with the exception of Franka Kitchen, which provides fewer demos).

### Questions
Have you tested on suboptimal demonstrations? Are there any interesting findings/results which may point to future avenues for research or failings of the method which should be investigated/improved upon?

Were you able to find notable reasons for why the MSE-regularized learning objective would occasionally outperform the IRL-regularized version? Is there any relationship to task difficulty, data diversity, etc?

In “Algorithm 2: PEAR,” line 8, shouldn’t the lower-level policy’s IL regularization be done with $D_g$? Since we are providing state $s^f$ from the goal dataset $D_g$ and subgoal supervision $s^e_g$ to the goal-conditioned low-level policy, then the policy predicts action $a$, and we regularize this to be close to the dataset action $a^f$ (either with MSE or the IRL objective)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce an algorithm, Primitive Enabled Adaptive Relabeling (PEAR), to address the issue of non-stationary transition and reward functions when implementing HRL.  Like Relay Policy Learning (RPL) (Gupta 2019), PEAR encourages the high level policy to output feasible subgoals using imitation learning.  The main difference from RPL is that instead of imitating the actions from a dataset that occur at fixed intervals, PEAR uses a heuristic to select the latest subgoal that the low level policy can achieve.  The authors show in a variety of experiments that PEAR can outperform several baselines.

Gupta et al. Relay Policy Learning. 2019

### Strengths
- The writing was clear and easy to understand.
- The authors included several ablation experiments that provided some important insights.
- The paper included some real world robotic experiments.

### Weaknesses
My main concern with this approach is that the contribution seems to be marginal as I do not see a compelling reason why PEAR would consistently perform better than (a) HAC (Levy 2019) with replay buffer augmented with demonstration data and (b) hierarchical behavior cloning, in which pure imitation learning is applied to both a high and low level policies.  

HAC already addresses the problem of nonstationarity through relabeling and subgoal testing.  The problem with HAC is that it does not have a built-in mechanism to explore but this can be remedied with the demonstration data that is provided to PEAR.  An advantage of HAC + demonstration data over PEAR, which would use a pure RL objective, is that if the demonstration data is suboptimal, it would not have an imitation learning regularization term forcing the agent to output suboptimal subgoals.  HAC has also demonstrated it can learn more than two levels of hierarchy.  The results of HER+BC, which I understood to be HER with a replay buffer augmented with demonstration data, was often ultimately able to match the performance of PEAR (see Figure 14), making it more likely that HAC, which is close to a hierarchical version of HER, should be able to match PEAR.  

In addition, it seems that a pure hierarchical imitation learning approach, in which both levels are trained with supervised learning should also work, at least potentially with more data.  The baseline BC may not have worked well because the tasks were too long horizon, but the addition of a high level policy trained with imitation learning should help.

Levy et al. Hierarchical Actor-Crtiic. 2017

### Questions
1.  Why is the Q_{Thresh} set to 0?  If the low-level reward is -1 for not achieving the goal and 0 for achieving the goal, the Q value should be negative for any correct action that puts the agent on a path to achieve the goal.  
2.  Is the high level policy only trained with a dataset containing expert trajectories?  Or is it also trained on its own interaction data?  If it is trained on its own interaction data, is there any relabeling done on that?
3.  I did not understand the “margin” component of the objective.  Can you provide another explanation of this component?
4.  Why does the paper characterize the number of demonstrations as a “handful” of demonstrations, when it uses 100 for most tasks?
5.  Have you experimented on any domains with image observations?

I am willing to raise my score if the authors can (i) provide some principled reasons why PEAR should outperform HAC+demonstrations and hierarchical behavior cloning and (ii) provide some answers to the above questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel approach called Primitive Enabled Adaptive Relabeling (PEAR) aimed at enhancing Hierarchical Reinforcement Learning (HRL) for complex long-horizon tasks. The authors propose a two-phase methodology where the first phase involves adaptive relabeling of expert demonstrations to generate subgoals, followed by joint optimization of HRL agents through Reinforcement Learning (RL) and Imitation Learning (IL). The results indicate that PEAR outperforms various baselines in both synthetic and real-world robotic tasks, achieving up to 80% success rates in challenging environments.

### Strengths
1. **Innovative Approach:** The use of adaptive relabeling to generate subgoals tailored to the capabilities of the lower primitive is a significant contribution that addresses the non-stationarity issue in HRL.
2. **Theoretical Justification:** The authors provide theoretical analysis that bounds the sub-optimality of their approach, lending credibility to their claims.
3. **Comprehensive Experiments:** Extensive experiments across multiple challenging tasks demonstrate the practical efficacy of PEAR, showing improved performance and sample efficiency over existing methods.
4. **Real-World Application:** The validation of PEAR in real-world tasks enhances the relevance and applicability of the research.

### Weaknesses
1. **Expert Demonstrations Requirement:** The first phase of PEAR relies on expert demonstrations to generate subgoals, which raises concerns about the fairness of comparison with other HRL methods that do not require such demonstrations. This could affect the generalizability of the findings.
2. **Lack of Recent Comparisons:** The paper does not include comparisons with several hierarchical reinforcement learning methods published in the last three years. This omission limits the contextual relevance of the results and could misrepresent the state of the art, such as [1], [2].

[1] Kim J, Seo Y, Shin J. Landmark-guided subgoal generation in hierarchical reinforcement learning[J]. Advances in neural information processing systems, 2021, 34: 28336-28349.

[2] Wang, Vivienne Huiling, et al. "State-conditioned adversarial subgoal generation." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 8. 2023.

3. **Citation Issues:** Many citations are sourced from CoRR instead of their formal conference versions. This oversight should be corrected to ensure academic integrity and accurate referencing. For example, the last two references should be cited as follows (Note that there are many other citation errors beyond these two):

Wulfmeier, Markus, et al. "Data-efficient hindsight off-policy option learning." International Conference on Machine Learning. PMLR, 2021.

Zhang, Tianren, et al. "Generating adjacency-constrained subgoals in hierarchical reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 21579-21590.

### Questions
- How do the authors justify the reliance on expert demonstrations in the first phase of PEAR compared to HRL methods that function without such requirements?
- Can the authors provide additional comparisons with recent HRL approaches to strengthen the positioning of PEAR within the current landscape?

### Soundness
3

### Presentation
2

### Contribution
2
