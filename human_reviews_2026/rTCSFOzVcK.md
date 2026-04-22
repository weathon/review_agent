# Hierarchical Contrastive Reinforcement Learning: learn representation more suitable for RL environments

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 6

## Abstract
Goal-conditioned reinforcement learning holds significant importance for real-world environment, but its inherent sparse reward structure brings challenges. In recent years, some researchers have attempted to learn better representations to solve RL tasks, among which the contrastive reinforcement learning that is independent of rewards demonstrating strong performance. However, how to make contrastive learning better applicable to reinforcement learning settings is still a problem. In this paper, instead of learning representations related to goals in environments, we present hierarchical contrastive reinforcement learning (HCRL) method that reduces the difficulty of learn goal representations by introducing intermediate representations related to states. HCRL expresses such an idea that the agent should first understand the environment and then understand the tasks in the environment. Our method fully utilizes the information in the GCRL setting and provides a better representation learning method, which has better performance in various complex and sparse environments without adding additional assumptions or constraints. The results of comparison experiments show that HCRL has faster convergence speed and higher success rate than prior work on a range of goal-conditional RL tasks. We further conduct ablation studies and additional evaluations to validate our method. Anonymous code: https://anonymous.4open.science/r/HCRL-6E88.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents an incremental method for learning goal-conditioned reinforcement learning (GCRL) policies. The authors build on contrastive reinforcement learning and propose learning intermediate sensorimotor action-state representations using contrastive objectives. The experiments show improved results across several environments compared to a set of baselines.

The method appears is novel, albeit incremental over CRL. However, the paper suffers from grammatical issues, missing words, and a general lack of clarity. The motivation behind the proposed approach could be more clearly framed. In addition, the experiments do not demonstrate the benefits of the approach in complex state and goal spaces.

Writing and Presentation Issues

- Line 48: The meaning of “a representation that implies a probability” is unclear.
- Line 77: It is unclear what is meant.
- Line 83: Sentence appears incomplete.
- Line 134: The subtitle is awkwardly phrased.
- Abstract: The sentence “Our work fully utilizes the information in the GCRL setting” is vague. What specific information is being referred to? Moreover, the abbreviation GCRL is not introduced before use.
- Line 232: It seems the authors intended to refer to s′ (next state).
- Figures 1–3: These figures are never cited or discussed in the text.
- Lines 256–259: The intuition behind the method is unclear and should be elaborated.
- Line 285: A parenthesis is missing.
- Line 316: The subsection title “Apply Representation to RL” is unclear—consider rephrasing (e.g., “Applying Learned Representations to RL”).
- Line 377: The statement “The recommended target representation dimension is 64” requires either a citation or an experimental justification.
- Section 5.3: The qualitative results are not analyzed or discussed.
- More detail should be provided about the state and action spaces used in experiments.

Experimental Concerns

- It is unclear why the experiments in Figure 9 are not presented alongside those in Figure 4. The authors explain that these environments are “too simple to converge quickly,” which is confusing: simpler environments are typically expected to converge more easily.

- A central objective of the method is to learn intermediate representations. Therefore, experiments in more challenging settings (e.g., with image-based state and goal inputs) would strengthen the paper.

- The paper does not clearly identify which factors contribute to the method’s performance. In the ablation study, the authors should analyze success rates with and without each of the two loss functions. Without the first loss function (Equation 11), the method closely resembles CRL, differing mainly in neural network architecture. It is therefore important to clarify whether the performance improvements stem from architectural changes or from the first contrastive loss. The loss functions in Equation 11 should have distinct names for clarity.

### Strengths
- Interesting topics and relevant benchmark
- Preliminary experiments demonstrate significant improvements over the baselines

### Weaknesses
- General lack of clarity
- Incremental method, relative to CRL
- Critical experiments are missing to validate the method

### Questions
Please, see above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for learning decoupled state-to-state and state-to-goal representations for goal-conditioned reinforcement learning. While the idea is interesting and potentially valuable, the paper suffers from very poor presentation, missing key baseline comparisons, unclear notation, and numerous writing errors. Several claims are unsupported, and the experimental results lack statistical analysis. Overall, the work needs substantial rewriting and stronger empirical validation before it can be considered for acceptance.

### Strengths
The paper presents an interesting idea for learning decoupled state-to-state and state-to-goal representations for goal-conditioned reinforcement learning (RL). This is a promising perspective and is shown to perform better than the baselines.

### Weaknesses
**Presentation**
- The paper contains numerous typos, incomplete sentences, and inconsistent or incorrect notation.
- In addition, several symbols are used before being introduced (or never introduced at all). For example, $p_g$ used in equation (2) and $B$ used in equation (6) are never defined, $\phi$ used in equation (3) is introduced only later. Overall, these issues make the paper extremely confusing to read.

**Missing baselines**
- Several important baselines are missing from the comparisons, such as CURL: Contrastive Unsupervised Representations for Reinforcement Learning, HIQL: Offline Goal-Conditioned RL with Latent States as Actions, Contrastive Learning as Goal-Conditioned Reinforcement Learning, and Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression, among others. 
- The paper ignores these baselines entirely and only compares against the simplest one, CRL. Even if the proposed approach does not outperform these stronger baselines, their inclusion would still be valuable, at least in the appendix.

**Unsupported claims**
- The paper also includes unsupported statements such as “This setting is beneficial to enhance the generalization of RL and is meaningful to sim2real,” and “In some papers, this sparse reward function is equal to…”. Such claims should be substantiated with references or evidence.
- Finally, the authors do not report the statistical significance of their results. There are no standard deviation plots, min–max regions, or p-value analyses, which raises concerns about the robustness of the reported performance.

### Questions
- Why did the authors choose the baselines they did, and why not include more relevant ones?

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
The authors propose HCRL, which additionally learns intermediate representations to simplify target representation learning and leverages them for policy improvement in reinforcement learning (RL). They claim this addresses the limitation of existing contrastive learning-based representation methods, which struggle to effectively capture state representations relative to goals.

### Strengths
Deriving intermediate representations through a hierarchical structure and applying contrastive learning across both stages appears to distinguish this work from prior studies.

### Weaknesses
Overall, the explanations supporting the authors' claims feel inadequate. Inferring from the paper's descriptions, the authors seem to treat goals and states as defined in separate spaces. If accurate, this constitutes a crucial assumption. Many studies, including those on CRL, presuppose that the state space and goal space are identical; thus, adopting a different setup would necessitate explicit discussion. Yet, the paper lacks any such clarification or details on how goals are defined in the experimental settings. Moreover, if baselines receive only goals with limited information while HCRL is provided with intermediate states containing additional details, this could undermine the fairness of comparisons.

### Questions
1. As noted in the Weakness section, is it correct that the goal space and state space are not identical?
2. In line 73, the authors assert that the encoder vector for goals is non-discriminative, whereas it is discriminative for states. Supporting evidence for this claim should be added, such as references to prior work addressing the issue or experimental demonstrations.
3. Line 287 mentions “q and k are inputs.” What exactly do q and k refer to?
4. I am curious about how goals are defined in the experimental environments employed.
5. The paper states that the experimental graphs represent averages over five random seeds. Including standard deviations across these seeds in the graphs would be advisable.
6. In line 837, further clarification is needed on how sampling is performed for states, actions, future_states, and achieved_goals.

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
4

### Summary
The paper presents an interesting adaptation of contrastive reinforcement learning (CRL) for the specific GCRL setting where the goal representation differs from the full state representation. This is a practical and interesting setting affecting many real-world use cases of GCRL. The idea is to use the state trajectory to learn a useful intermediate representation of the state, and then use this representation to learn effective goal distances. The paper provides evidence of the success of this methodology on a suite of GCRL environments, demonstrating superior performance compared to CRL and other state-of-the-art goal-conditioned RL algorithms.

### Strengths
- The paper is supported by a comprehensive set of experiments across a diverse suite of challenging GCRL environments. The results convincingly demonstrate that HCRL significantly outperforms strong baselines, including CRL and HER-augmented agents, in terms of both sample efficiency and final success rate.

- The paper is well-written and clearly structured and the methodology is presented in a logical flow that is easy to follow.

- The paper addresses a practical and important problem in GCRL, the common scenario where the goal is a sub-component of the full state.

### Weaknesses
- The contribution doesn't seem to be much novel compared to the original CRL. Moreover the problem doesn't seem to be very well motivated, I think the characteristic of the goal space where this intermediate representation would help have to be improved and is not easy for me to understand this limitation and when this extra representation could hurt instead of helping learning. 

- While CRL is the most direct baseline, the comparison would be more compelling with the inclusion of other representation-learning or contrastive-based GCRL methods, such as QRL (Wang et. al. Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning).

### Questions
- See weaknesses points.

- Why repeating equations 8 and 9 again in equation 10 and 11?

- What would happen if the goal represent a set of acceptable terminal states, could the precision learned by the intermediate representation directly interfere with the necessary goal-set invariance?

### Soundness
2

### Presentation
3

### Contribution
3
