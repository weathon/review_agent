# Hierarchical Preference Optimization: Learning to achieve goals via feasible subgoals prediction

- Avg Score: 5.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 8, 5

## Abstract
This work introduces Hierarchical Preference Optimization (HPO), a novel approach to hierarchical reinforcement learning (HRL) that addresses non-stationarity and infeasible subgoal generation issues when solving complex robotic control tasks. HPO leverages maximum entropy reinforcement learning combined with token-level Direct Preference Optimization (DPO), eliminating the need for pre-trained reference policies that are typically unavailable in challenging robotic scenarios. Mathematically, we formulate HRL as a bi-level optimization problem and transform it into a primitive-regularized DPO formulation, ensuring feasible subgoal generation and avoiding degenerate solutions. Extensive experiments on challenging robotic navigation and manipulation tasks demonstrate HPO’s impressive performance, where HPO shows an improvement of up to 35% over the baselines. Furthermore, ablation studies validate our design choices, and quantitative analyses confirm HPO’s ability to mitigate non-stationarity and infeasible subgoal generation issues in HRL.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposed a Hierarchical Preference Optimization (HPO) algorithm for hierarchical reinforcement learning. The algorithm aims to generate feasible subgoals and mitigate the non-stationary in HRL. HPO leveraged the low-level value functions to condition higher-level policy for subgoal generation and utilized the direct preference optimization (DPO) to optimize the higher-level policy.

### Strengths
The idea to introduce low-level value to regularize high-level policy optimization, and leveraging DPO to optimize a traditional RL problem is novel.

### Weaknesses
1. The proposed HPO algorithm is built on the goal-conditioned HRL concept. However, in the problem formulation, the definition of the high-level reward deviates from the standard goal-conditioned HRL framework, making it not a robust problem definition. Additionally, some derivations need further clarification or analysis. See details in Questions.

2. The HPO is an HRL approach, but the paper doesn't compare with SOTA HRL works. I encourage the author to involve at least one recently representative HRL algorithm as baseline to further demonstrate HPO's advantages (reference [1][2][3]).

[1] Gürtler, Nico, Dieter Büchler, and Georg Martius. "Hierarchical reinforcement learning with timed subgoals." Advances in Neural Information Processing Systems. (2021).

[2] Kim, Junsu, Younggyo Seo, and Jinwoo Shin. "Landmark-guided subgoal generation in hierarchical reinforcement learning." Advances in neural information processing systems. (2021).

[3] Zhang T, Guo S, Tan T, Hu X, Chen F. Generating adjacency-constrained subgoals in hierarchical reinforcement learning. Advances in Neural Information Processing Systems. (2020).

### Questions
**Question 1: Problem Formulation**.

The problem formulation for goal-conditioned HRL is not entirely accurate. Specifically, in the paragraph starting from Line 157: "the lower-level policy is driven by a sparse reward signal, ...., indicating that the subgoal is reached." This is correct, as the low-level policy aims to achieve the subgoal set by the high-level policy. However, the high-level reward function is defined as $r^H = \sum_{sub-trajectory}{r^L}$, where $r^L$ is the low-level reward. This doesn't seem correct to me, as the high-level aims to generate sub-goals guide the low-level to **achieve the final task objective**, i.e., the high-level reward is usually defined based on the environmental reward signal from the problem MDP. (Check the goal-conditioned HRL framework definition in reference [4]). With the definition given in the paper, the high-level reward appears to be evaluating "how many steps in total of the low-level policy is staying near my generated subgoal." In this problem formulation, the original environmental reward signal is completely omitted, so how can HPO ensure that it is optimizing the original task rewards?

This definition also leads to an extreme case where the high-level policy simply generates the current state as the next subgoal, making the low-level policy do nothing and still "achieve" the subgoal. In this scenario, both the low-level policy and high-level policy would receive the highest reward, fully satisfying their optimization objectives. However, this would cause the agent's overall policy to just be idle. (My main concern is the problem formulation doesn't involve the MDP reward function).

Given this, I'm unclear how HPO is supposed to work.

[4] Kulkarni, Tejas D., et al. "Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation." Advances in neural information processing systems (2016).

**Question 2**. Following up on Question 1, around Line 485, it is mentioned that "HPO consistently generates low average distance values, which implies that HPO mitigates non-stationarity." The subgoals generated by HPO are often "close" to the current state. Could this be due to the aforementioned definition of the high-level reward function, which evaluates whether the low-level policy has achieved the subgoal? If so, would generating only near subgoals prevent the agent from progressing toward the overall task objective?

**Question 3**. At around Line 347, could you further prove why the advantage equals the entropy of the policy ($A(s_t,g^*,g_t) = \beta log(\pi^H (g_t | s_t, g^*))$)? and how is the $\beta$ defined? The advantage directly equates to the entropy of the policy is not intuitive to me. Ziebart's paper studies a special case based on some assumptions, it may not be generally applicable to all RL problems.

I would like to increase the score if these concerns are addressed.

### Soundness
1

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
This paper formulate HRL as a bi-level optimization problem and transform it into a primitive-regularized DPO formulation. The proposed method HPO incoporates token-level DPO into Max-Ent RL for mitigating non-stationary issue and infeasible subgoal generation issue.

### Strengths
1. This paper proposes a primitive-regularized preference optimization approach for HRL, which is a novel try. 
2. The derived DPO formulation has theoretical groundings.

### Weaknesses
HPO is sensitive to the two introduced hyperparameters, $\lambda$ and $\beta$, according to Figure 5 and Figure 6 in the Appendix. Further, it is not clear the values of $\lambda$ and $\beta$  used in each task of HPO in Figure 2-4.

### Questions
1. Based on the experimental settings detailed in the Appendix, the pick-and-place task appears simple enough for single-level RL methods to solve, as seen in environments like panda-gym [1], meta-world [2], and td-mpc [3]. Therefore, it is unclear what makes your experimental setup unique, given that none of the baselines aside from HPO achieve a satisfactory success rate.

   [1] Gallouédec, Quentin, et al. "panda-gym: Open-source goal-conditioned environments for robotic learning." arXiv preprint arXiv:2106.13687 (2021).

   [2] Yu, Tianhe, et al. "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning." Conference on Robot Learning, PMLR, 2020.

   [3] Hansen, Nicklas, Xiaolong Wang, and Hao Su. "Temporal difference learning for model predictive control." arXiv preprint arXiv:2203.04955 (2022).

2. Could you clarify why HPO’s performance is relatively low on Maze navigation tasks?

3. To better illustrate HPO's ability to address non-stationarity and infeasible subgoal generation (Figure 3 and Figure 4), a comparison with the HAC baseline would be better, as HAC also addresses non-stationarity and is a recent work. I think this comparison should not pose too much burden on the authors, as HAC is already included as a baseline for success rate comparison in Figure 2.

4. There is a typo in line 761:  "in Figure 6."

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present HPO, a hierarchical RL method that directly optimizes environment reward and preferences coming from sparse sub-goal reaching reward by a lower level policies. The specific objective is derived from DPO, and helps mitigate non-stationarity common to most HRL methods.

### Strengths
**Motivation:** Non-stationarity in HRL is a big issue and this paper presents a well-motivated solution for it.

**Comparisons:** The # of and relevancy of baselines is solid, this is a convincing set of comparisons.

**Experiments:** The experiments are performed on tasks well-suited for HRL, and the analysis on goal distance prediction against HIER and HAC demonstrates that HPO’s objective encourages sampling reachable goals for the lower-level policy.

**Clarity:** THe paper is overall quite clear and the walkthrough of how to obtain the objective was both interesting and easy to read.

### Weaknesses
**Clarity:** Overall clarity is good, but the reason *why* non-stationarity is solved should be explicated better, earlier in the paper. Non-stationarity occurs because a high level policy outputting a certain subgoal can result in a different reward later in training. The reason why this is solved is because the ***reward** for the high-level policy automatically adapts with the low level policies changin*g as it is based on the value function. The part I italicized isn’t that clearly presented in the paper. 

- For example, when looking at Figure 1, it just looks like the Value function being given to DPO is the reason why non-stationarity is solved. The caption states “Since this preference-based learning approach does not depend on lower primitive, this mitigates non-stationarity. Note that since the current estimation of value function is used to regularize the higher policy, it does not cause non-stationarity.”
    - Instead, this can be simplified to some form of the italicized statement above; the current statement does not directly explain why.
- Similar comment for the introduction and after giving the full objective in Eq.14.

**Experiments:** Why not compare HAC and HIER on the same graphs in Figs 3/4? It’s a little strange to pick each one individually for a separate comparison when they can be compared on the same things. 

**Minor Issues:**

- A high level policy discount factor is missing from Equations 4, 9, 10 and so on. Maybe it’s not necessary as the authors are considering the one-step DPO objective, but perhaps that could be mentioned?
- Figure 2 text size and line widths are too small

### Questions
From Eq. 6 to Eq. 7, the constraint that $V_{\pi_L} > V_{\pi_L^*}$ is dropped for $V_{\pi_L} > \delta$ due to the justification that for sparse-reward goal-reaching, $V_{\pi_L^*} > 0$ must be true. But this no longer optimizes the same objective, right? We still don’t know the ground truth value that $V_{\pi^*_L}$ should be; the writing seems to ignore this issue. A simple footnote or extra sentence of discussion stating this problem would make this part clearer.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper is about (goal-conditioned) hierarchical reinforcement learning. The authors describe two key challenges in hierarchical reinforcement learning: training instability due to non-stationary of off-policy learning for the higher-level policy and generation of infeasible sub-goals by the higher-level policy. It proposes a hierarchical approach in which the higher-level policy is optimized with a token-level direct preference optimization method and the lower-level policy is optimized with reinforcement learning. The goal of this approach is to make the learning of the higher-level policy independent from the lower-level policy (i.e. its current sub-optimal form) to avoid issues arising from non-stationarity. To this end, the paper re-formulates the hierarchical reinforcement learning problem as a bi-level optimization problem which is solved by first posing an equivalent constrained optimization problem. The proposed method is evaluated in a set of experiments and compared to a set of baselines.

### Strengths
The paper provides good background on reinforcement learning from human feedback and direct preference optimization. The paper also clearly describes the limitations it aims to address. The authors introduce a bi-level formulation of the hierarchical reinforcement learning problem to provide formalized arguments for the issues that they want to address. The overall issue that is raised in this paper, i.e. the complications arising for the interplay between the high-level and low-level policies is highly relevant for hierarchical reinforcement learning and satisfying solutions for this problem are in demand.

### Weaknesses
In parts, this paper is hard to follow. For example, the part where the notation and the sub-goals are introduced is confusing as to the nature and purpose of the sub-goals. More clarity as to the definition of the hierarchical MDP would be good. Another reason is the level to which the paper is self-contained. For example, in line 206, the authors use an equation for the optimal policy with reference to a tutorial, but it is unclear what the equation means and why it is used.

### Questions
In Fig. 3, the authors are presenting a form of evaluation for their claim regarding non-stationarity. This evaluation is indirect and is relying on the task (i.e. what distances mean). Can the authors present a task independent evaluation that supports their claim? E.g. sometime closer to the formalization?

### Soundness
2

### Presentation
2

### Contribution
2
