# Generalizing Subgoals from Single Instances using Hypothesis-Preserving Ensembles

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
A reinforcement learning agent trained on a single source subgoal has no way to determine during training which features will be relevant for future instances of that same subgoal. This creates ambiguity: multiple plausible models of a subgoal can fit the training data but not all will successfully transfer. Humans address this ambiguity by maintaining alternative hypotheses until new information reveals the most effective one. Drawing inspiration from this, we introduce a hypothesis-preserving ensemble in which each member is a distinct, plausible subgoal classifier trained on the same source task. The agent then tests these alternative hypotheses in a new task, learning policies for the corresponding subtasks and uses task reward to select the most effective classifier. Experiments on Montezuma's Revenge and MiniGrid DoorMultiKey show that our method recovers subgoals learned in the source task, successfully adapting them to visually different tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses a challenge in reinforcement learning where an agent trained on one instance of a subgoal cannot tell which features will remain relevant in new situations. To solve this, the authors propose a hypothesis-preserving ensemble, using a set of alternative subgoal classifiers representing different plausible interpretations of the subgoal. When faced with a new task, the agent tests these hypotheses, learns the corresponding policies, and uses task rewards to identify the most effective classifier. Experiments across various environments demonstrate that this method successfully transfers subgoal knowledge to visually distinct tasks.

### Strengths
1. The problem is well described and formalized. The paper presents a well-defined and rigorously formalized problem statement. The authors clearly articulate the challenges inherent in transferring subgoal knowledge across tasks using lemmas and equations.

2. The proposed approach is methodologically sound and conceptually well-motivated. The experimental results provide strong empirical support for the method’s effectiveness, demonstrating consistent improvements over non-ensemble baselines across multiple benchmark environments.

### Weaknesses
1. The proposed method appears to depend heavily on the training of multiple subgoal classifiers. Although the authors provide an adequate discussion of data efficiency, the paper lacks an analysis of the potential computational overhead introduced by this process. Given that the ensemble approach may require substantial additional computation, it could pose practical limitations on scalability and real-world applicability. A more thorough examination of the method’s computational complexity, perhaps through empirical runtime comparisons, would strengthen the evaluation and clarify its trade-offs relative to more lightweight alternatives.

2. The authors use task reward as an indirect signal. While the conceptual explanation of this mechanism is clear, the paper would benefit from a more formal theoretical analysis or empirical validation demonstrating the reliability of this strategy over others.  This could better support the claim "task reward reliably selects the subgoal hypothesis that best matches the demands of the current task".

### Questions
1. Could the authors provide additional evidence to address my concerns listed under “Weaknesses”?

2. Out of curiosity, I would like to ask whether the proposed method—currently relying on trained classifiers as subgoal selectors—could be extended to incorporate generative approaches capable of producing potential subgoals without requiring classifier training on labelled data. Such an extension might offer greater flexibility and reduce dependence on supervised subgoal annotation.

### Soundness
3

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
The authors address the challenge that, when subgoals are pre-defined and labeled within a dataset corresponding to a single task, the resulting subgoal classifier may be poorly learned and fail to accurately identify subgoals when the environment or task shifts. To mitigate this, the paper proposes a method whereby the agent trains an ensemble of subgoal classifiers and, when faced with a novel task, evaluates these candidates to select the classifier with the best generalization capability.

### Strengths
Raising the issue that a subgoal classifier defined for one task may not function reliably on others introduces a new challenge that has not been discussed in previous research.

### Weaknesses
- The authors point out that subgoal classifiers trained via supervised learning on human-labeled data may be based on features unrelated to the actual reason for their selection as subgoals. This is reminiscent of the causal confusion issue raised in [1], which highlighted difficulties in discerning which features of a state lead to particular actions in imitation learning scenarios. The relationship between state-action in [1] and state-subgoal in this work appears analogous. If so, it is unclear how the DivDis algorithm, employed in Section 5, benefits the subgoal classification process; rather than revealing the causal relationship between states and subgoals, it might merely encourage the classifier to label previously unlabeled states as subgoals.
- Furthermore, requiring manual subgoal definitions could limit the applicability of this framework to real-world or more complex environments.

[1] De Haan, Pim, Dinesh Jayaraman, and Sergey Levine. "Causal confusion in imitation learning." *Advances in neural information processing systems* 32 (2019).

### Questions
1. The paper notes that only data from a single training task is used. In this context, coverage of the state space in the dataset is likely critical for generalizability across diverse tasks. Was this considered during data collection, and can the approach handle out-of-distribution (OOD) states not present in the dataset?
2. Including quantitative evaluations of ensemble members would be beneficial. This would help reveal how diverse the classifiers are and demonstrate the importance of reward-guided selection.
3. It would be instructive to provide qualitative evaluations of the diversity of classifiers generated through the DivDis algorithm. While Figure 6 visualizes only the most and least frequently selected classifiers, visualizing classifiers individually could better illustrate their diversity.
4. There is a typo in Equation 4.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tackles subgoal generalization when all labels come from a single training task, a setting that is naturally under-specified. The authors propose keeping an ensemble of subgoal classifiers, each representing a plausible hypothesis of where the subgoal is. Options are trained for each hypothesis, and a high-level policy uses task reward to choose which option to execute at test time. Experiments on pixel-based Montezuma’s Revenge (recognition + options) and MiniGrid DoorMultiKey (full hierarchy) show that preserving multiple hypotheses (via ensembles) improves subgoal recognition and downstream performance over a single-model baseline.

### Strengths
- Using ensembles for subgoal classification rather than a single model is a meaningful concept.
- Results follow intuitively: on Montezuma, ensembles recognize subgoals better and options terminate closer to targets; on MiniGrid with distractors, the hierarchical agent closes much of the gap to an oracle and beats flat baselines and a single-classifier option agent.

### Weaknesses
- Scope is limited to predefined subgoals; it is unclear how this interacts with subgoal discovery.
- The approach relies on reward to pick among hypotheses, so it's not clear what is the benefit of generalization of predefined subgoals as an overall approach.
- It was not clear why and how extra unlabeled data is used from other seeds.
- The baselines of CNN v/s ensemble of CNNs are trivial, e.g., representation learning and domain randomization are valid methods for low-diversity data.
- Not sure what is the RL benefit of this approach as one still needs to learn different policies for each subgoal, and then expects these policies to generalize on the target task which also assumes access to rewards.
- Using task reward for subgoal / option selection is greedy but not long-term optimal because we make a policy decision based on intermediate reward w.r.t reaching the subgoal and not the cumulative reward in the task including the trajectory part beyond the subgoal. For instance, it's possible that a seemingly suboptimal subgoal is actually optimal w.r.t. full task solution as it allows for a shortcut later.

### Questions
- What is the motivation behind the problem studied in this work? Why is a niche problem like single-subgoal generalization important to study and what are the implications and applications of the findings of this paper?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes to capture uncertainty in the prediction of subgoal by using ensemble in hierarchical RL. More specifically, the method gathers supervised subgoal data in one environment, and transfers the learned ensemble of subgoal networks to new environments. It shows that, by maintaining the uncertainty of subgoals, the agent achieves better results on new environments. Through policy learning, the agent automatically identifies and relies on the more accurate subgoal classifiers.

### Strengths
The paper provides controlled experiments in section 6.1 and 6.2.

### Weaknesses
Because the method needs supervised subgoal data, the whole setup feels a bit artificial. I am not sure if the method can be used in real-world scenarios where subgoal cannot be supervised.

The paper has limited novelty - using ensemble to capture uncertainty is an established practice.

### Questions
1. In eq 3, could the authors clarify the notation? It seems to me that $y_i$ and $y_j$ are not the groundtruth subgoal label, but the prediction from the subgoal classifier $\hat{y}_i$ and $\hat{y}_j$.

2. In section 6.1, the experiment extends the dataset size by adding data from new rooms. What about adding more data in the same room? Is the size of the data sufficiently for accurate prediction for a single room?

3. The paper mentioned using unlabelled dataset for DivDis, and for Minigrid DoorMultiKey environment, the unlabelled dataset is gathered in unseen environments. Does DivDis experiment on Montezuma’s Revenge use the same setup? Is it important to be unseen environment?

4. In the current setup, only the subgoal classifiers are transferred to new environments. What if we learn subgoal policy $\pi_{o_i}$ as well and transfer the policy instead? What's the performance and would that be more data efficient? 

5. Learning a policy for each subgoal hypothesis is quite expensive. Do the authors think it's possible to learn a single policy that leverages the ensemble's uncertainty estimate? If so, how that performs compared to the proposed method?

### Soundness
2

### Presentation
3

### Contribution
1
