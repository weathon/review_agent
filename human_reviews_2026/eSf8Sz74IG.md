# Reinforcement Learning for Durable Algorithmic Recourse

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Algorithmic recourse seeks to provide individuals with actionable recommendations that increase their chances of receiving favorable outcomes from automated decision systems (e.g., loan approvals). While prior research has emphasized robustness to model updates, considerably less attention has been given to the temporal dynamics of recourse---particularly in competitive, resource-constrained settings where recommendations shape future applicant pools. In this work, we present a novel time-aware framework for algorithmic recourse, explicitly modeling how candidate populations adapt in response to recommendations. Additionally, we introduce a reinforcement learning (RL)-based recourse algorithm that captures the evolving dynamics of the environment and generates recommendations that are both feasible and valid.
We design our recommendations to be durable, supporting validity over a predefined time horizon $T$. This durability allows individuals to confidently reapply after taking time to implement the suggested changes. Through extensive experiments in complex simulation environments, we show that our approach substantially outperforms existing baselines, offering a superior balance between feasibility and long-term validity. 
Together, these results underscore the importance of incorporating temporal and behavioral dynamics into the design of practical recourse systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new framework of algorithmic recourse to address the challenge of providing durable and effective recourse recommendations. 
The authors focus on competitive resource-constrained environments (e.g., admissions, loan approvals), where providing recourse recommendations to rejected applicants can alter the applicant pool and shift the decision boundary.
The authors propose a concrete solution based on reinforcement learning that accounts for the dynamics of the environment to generate feasible and valid recourse recommendations. 
By experiments on synthetic data, the proposed method was evaluated against existing standard algorithmic recourse methods.

### Strengths
1. This paper is well-written, well-organized, and easy to follow. 
1. The authors tackle an underexplored problem in the field of algorithmic recourse. I agree with the importance of ensuring long-term validity in dynamic environments. The authors also propose a first concrete solution to this problem. I think it is interesting to leverage reinforcement learning to model dynamics and optimize recourse recommendations that are both valid and feasible.

### Weaknesses
1. One of my main concerns is the lack of theoretical analyses. For example, there is no theoretical guarantee of the cumulative reward achieved by the proposed algorithm. While I acknowledge that this paper focuses on establishing a foundation for the proposed problem, I think providing at least a preliminary theoretical discussion or bounds, even under simplified assumptions, would significantly strengthen the paper. Such an analysis could help clarify the conditions under which the proposed method is expected to perform well.
1. Another concern is the simplified experimental setting. For example, all features are continuous and drawn from independent distributions. In addition, the authors employed a logistic regression model as a prediction model $M$. However, real-world scenarios involve correlated or categorical features, as well as more complex decision boundaries produced by major models, such as neural networks or tree ensembles. Thus, the current experimental setting seems too simple to validate the efficacy of the proposed method. 
1. I am also concerned that the proposed method appears to require a substantial amount of interaction to learn an effective policy. In Appendix, the authors note that training involves tens of thousands of episodes in simulation. In many high-stakes decision-making tasks, such a large number of interactions may be costly and impractical. Thus, I am concerned that the proposed method may not work well in more practical situations.

### Questions
1. Some previous studies, such as (Wu et al. 2024) and (Kanamori et al. 2025), also consider algorithmic recourse in the online setting and conducted experiments on real datasets. As done in these studies, could the authors also conduct simulation experiments on real datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the problem of ensuring robust recourse suggestions in dynamic and resource-constrained environments. The authors first characterize a novel simulation environment, and then provide a reinforcement learning solution to provide population recourse suggestions that stay valid for a longer period of time. Lastly, the authors validate their approach in synthetic experiments comparing their approach with the state of the art.

### Strengths
The paper’s topic is interesting and original. It departs from the classical literature and it considers a more realistic environment where users also compete for resources. The simulation environment is interesting, and it provides additional options to simulate complex scenarios for recourse. If extended and/or tailored with some causal flavor (e.g., [1]), I believe it could become a nice synthetic benchmark for recourse algorithms and/or in general social decision making.

[1] Cinquini, Martina, et al. "A Practical Approach to Causal Inference over Time." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 14. 2025.

### Weaknesses
I believe Section 4 is not very clear in its exposition. The proposed setup is somewhat non-standard for recourse, and some of the assumptions they make are not very detailed (e.g., access to the POMDP, stochasticity of the bank’s score $g$).

- The recourse policy $\phi$ and the predictor policy $\mu$ are trained with two different environments (as specified by 4.1 and 4.2). It is not clear the reason for this choice. Indeed, having access to historical traces, or directly the POMDP environment, is already a very strong assumption, so I would have expected to have also $\phi$ trained with it.  
- The paper does not compare with the “naive” baseline, which is robustifying recourse with a simple $\epsilon$-ball, as is done in [2]. Indeed, the proposed approach requires building a very expensive world model (Section 4), while simple robustification solutions could be more viable in practice. As shown by [3], robustifying recourse with an $\epsilon$-ball can still provide some robustness to time. Although both [2,3] consider a causal setting, I believe a comparison with [2] could have been contemplated, or at least a discussion about why an RL approach is better.
- The simulation environment (Section 3.1) feels very abstract. Although it extends previous results, the authors do not describe potential real-world phenomena or population trends they could simulate with it. For example, they could have considered more challenging or interesting scenarios (e.g., different groups with different urgency/self-confidence that compete) to show how recourse changes the system dynamics.
- The notion of “equity” in the reward function (Equation 1) should be better contextualized. Indeed, the classifier score could be an imperfect proxy of the difficulty of a task. Thus, we might end up providing similar actions to diverse participants that produce the same score, but which require a very different effort per user.

[2] Dominguez-Olmedo, Ricardo, Amir H. Karimi, and Bernhard Schölkopf. "On the adversarial robustness of causal algorithmic recourse." International Conference on Machine Learning. PMLR, 2022.

[3] De Toni, Giovanni, et al. "Time can invalidate algorithmic recourse." Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency. 2025.

### Questions
Here, I leave some questions for the authors. The “mandatory” ones (I apologize for the lack of a better term) are those that could entail an improvement of the score if answered satisfactorily. 

**[Mandatory Questions]**
- How does your approach compare with a robustified recourse with an $\epsilon$-ball as in [2] in your scenario? Alternatively, what are the advantages of your approach with respect to [2]? 

**[Optional Questions]**
- Could you comment on the notion of “equity” you describe in your paper (Equation 1)? 
- Can you comment on potentially more complex scenarios you could simulate with your environment (Section 3.1)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a reinforcement-learning (RL) framework for *durable algorithmic recourse*—recommendations that remain valid over time in competitive, resource-limited settings. The authors model the recourse process as a partially observable Markov decision process (POMDP), capturing how applicants’ responses influence future environments. A hierarchical RL design is introduced with two policies: a recourse recommender φ that generates counterfactual feature vectors for rejected individuals, and a predictor μ that selects target scores to balance equity, feasibility, and validity over a horizon T. The recommender adaptively learns feature-specific modification difficulties from observed success rates, while the predictor anticipates population dynamics to maintain reliability under competition. Experiments on synthetic settings show improved Pareto trade-offs between Recourse Reliability (RR_t^T) and Feasibility (RF_t^T) compared to threshold-based and optimization-based baselines such as Ustun, Wachter, and DiCE.

### Strengths
The work formalizes the new notion of *durable recourse*, explicitly modelling time-dependent validity in dynamic, competitive environments. It is, to my knowledge, the first to cast recourse generation as a hierarchical RL problem, addressing endogenous feedback effects. The reward structure is well designed, combining the Gini index for equity with validity and feasibility objectives. The two-level decomposition (recommender and predictor) is a thoughtful design choice that improves computational stability and interpretability. The presentation is clear, with detailed formalism, well-motivated objectives, and effective figures illustrating Pareto frontiers and convergence behaviours. The paper addresses an important gap in the literature by moving beyond static or exogenous-robust recourse, and it provides a principled direction for future work connecting RL, fairness, and causal inference.

### Weaknesses
The RL training procedure is computationally heavy: convergence requires around 23,000 episodes for the recommender and 7,000 for the predictor, corresponding to unrealistically long simulated cycles. Although offline training is mentioned, there is no empirical demonstration or discussion of the data requirements for such a setup. The trained policies remain static and cannot adapt to changing models or applicant distributions, which undermines the claim of robustness to environmental shifts. All experiments rely on synthetic datasets with small populations (N=20, d=10) and simplified normal feature distributions; real-world evaluation on credit or recidivism data would strengthen the paper’s impact. The use of a fixed T-step history as a proxy for partial observability is a heuristic; there is no comparison against recurrent or belief-state approaches, nor analysis of the sufficiency of T. Finally, the ethical framing, though present, could better connect the notion of durability with user trust and systemic accountability.

### Questions
1. The paper mentions offline RL as a possible direction. What data would be required to pre-train the policies before deployment, and how could such data be obtained without an existing recourse system?  
2. How should practitioners select the time horizon T in real applications, where reapplication intervals vary among individuals?  
3. Could the approach be extended with continual or meta-RL to adapt to distributional shifts or updated decision models?  
4. Would results generalize to real-world datasets (e.g., COMPAS, FICO, or credit scoring)? Why or why not?
5. What are the computational and sample requirements for training both policies jointly, and how do they scale with feature dimensionality?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a RL approach to address algorithmic recourse in a competitive environment, where the action of each agent influence the chances of achieving recourse for the others.

### Strengths
Algorithmic recourse in competitive environments is a very sensible setting for real world applications, and very few works have started addressing it.

The RL method is carefully designed to maximise the realism of the setting.

### Weaknesses
Changes wrt baselines are not dramatic. Given that experiments are run in simulated settings, I am wondering whether there are conditions under which the proposed solution has larger advantages.

The novelty with respect to the works by Fonseca et al., 2023 and Bell et al., 2024 is not entirely clear. The authors do list assumptions being removed (end of section 3.1), but how much this affects the complexity of the task and the requirement for novel solutions is unclear.

### Questions
Can you better clarify the differences wrt existing solutions for competitive recourse? and their impact in the design of the method? isn't it possible to have experimental comparisons with their approach?

### Soundness
4

### Presentation
3

### Contribution
2
