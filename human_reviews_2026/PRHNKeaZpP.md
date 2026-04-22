# Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Multi-objective reinforcement learning (MORL) seeks policies that effectively balance conflicting objectives. However, presenting many diverse policies without accounting for the decision maker’s (DM’s) preferences can overwhelm the decision-making process. On the other hand, accurately specifying preferences in advance is often unrealistic. To address these challenges, we introduce a human-in-the-loop MORL framework that interactively discovers preferred policies during optimization. Our approach proactively learns the DM’s implicit preferences in real time, requiring no a priori knowledge. Importantly, we integrate this preference learning directly into a parallel optimization framework, balancing exploration and exploitation to identify high-quality policies aligned with the DM's preferences. Evaluations on a complex quadrupedal robot simulation environment demonstrate that, with only 
 interactions, our proposed method can identify policies aligned with human preferences, e.g., running like a dog. Further experiments on seven MuJoCo tasks and a multi-microgrid system design task against eight state-of-the-art MORAL algorithms fully demonstrate the effectiveness of our proposed framework. Demonstrations and full experiments are in https://sites.google.com/view/pbmorl/home.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates an interesting problem — exploring the decision maker’s preferred regions on the Pareto front in MORL. The authors propose a human-in-the-loop framework that interactively infers implicit preferences and guides policy optimization. However, the experimental analysis is relatively weak and requires further clarification and justification.

### Strengths
1. The paper addresses a timely topic, integrating human feedback into MORL with clear motivation.
2. The proposed pipeline is clearly presented and supported by detailed engineering implementation.
3. Experiments on the Unitree robot are technically solid and visually demonstrate distinct preference-aligned behaviors.
4. The overall idea of focusing learning on human-desired regions of the Pareto front is interesting and potentially useful.

### Weaknesses
1. Unfair baseline setup. Under different DM preference profiles, PBMORL explores specific regions on the Pareto front, while baselines still aim to cover the entire front.
2. Questionable evaluation metric. The use of a hand-crafted “golden policy”  as reference is under-explained and may not align with the DM preferences used during training, thus potentially biasing results toward extreme objectives.
3. Scalability and practicality of human feedback. The approach assumes frequent pairwise preference queries; it is unclear how feasible this remains when the number of objectives grows or in realistic human-in-the-loop settings.

### Questions
1. The paper states that the DM provides pairwise preference feedback, but it is not specified how this feedback is generated in simulation. Could the authors provide the exact rule or scoring function used by the oracle to decide which policy is preferred in each preference profile?
2. In the benchmark experiments, you evaluate using “golden policies.” Do these golden policies correspond to (or approximate) the same utility models that the DM uses during training to choose between policies? If not, how should we interpret the reported alignment scores?
3. Can you provide baselines that are constrained to the preference-aligned region of the Pareto front as PBMORL, rather than baselines that attempt to cover the entire Pareto front uniformly? For example, if you are considering the evolutionary methods, such as PGMORL, maybe you can conduct a rough selection of policies for specific regions? If this is impractical, please provide a detailed explanation.
4. How are training steps allocated to PBMORL and the baselines? If PBMORL is allowed to concentrate its entire budget in a small region of interest while the baselines are required to cover the full front under the same total budget, this seems structurally unfair. Please clarify.
5. In the UNITREE experiments, how many distinct policies are actually trained per preference profile for the scalarized PPO baseline? The paper notes that PBMORL returns multiple candidate policies per preference profile — what is the corresponding number allowed for scalarized PPO under each profile?
6. In the UNITREE experiments, how are the scalarization weights for the PPO baseline chosen? For example, why are extreme weightings such as [1, 0] and [0, 1] not included, and how sensitive are the baseline results to this choice?
7. The “approximation accuracy” metric equation is unclear. The text suggests an L2 distance of policies, but is that distance computed in policy parameter space, or in objective space? Please clarify the intended interpretation.
8. The paper claims that most MORL work considers only 2–3 objectives, and that work with more than 3 objectives exists only in toy settings. This is incorrect. There are recent methods that explicitly tackle higher-dimensional objective spaces (4+ objectives, and up to 9) in realistic domains such as robotics, transport planning, and scheduling [1-3]. (Since the paper primarily emphasises the human-in-the-loop framework, I don’t expect extensive many-objective experiments. However, the authors should correct the misleading statement.)

Overall, I find the problem important and the design interesting, but the experimental setup leaves several open questions. I would be happy to raise my score if the authors can clarify these issues.

[1] Huang, Bo-Kai. Q-pensieve: Boosting sample efficiency of multi-objective RL through memory sharing of q-snapshots. MS thesis. National Yang Ming Chiao Tung University, 2022. (their arXiv version includes experiments up to 5 objectives)

[2] Liu, Ruohong, et al. "Efficient Discovery of Pareto Front for Multi-Objective Reinforcement Learning." The Thirteenth International Conference on Learning Representations.

[3] Michailidis, Dimitris, et al. "Scalable multi-objective reinforcement learning with fairness guarantees using lorenz dominance." arXiv preprint arXiv:2411.18195 (2024).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a human-in-the-loop multi-objective reinforcement learning (MORL) framework that learns preferences from human feedback. The preferences are modeled using a Gaussian process, and the method is evaluated across various robotic environments.

### Strengths
1. The proposed method is conceptually straightforward and easy to follow.
2. The evaluation on realistic Unitree robotic environments adds practical relevance and credibility to the study.
3. The work is well-presented, with clear figures and informative videos that effectively illustrate the results.

### Weaknesses
1. Limited technical novelty:
While the proposed framework is well-structured and clearly presented, it lacks substantial technical innovation. Each component builds on existing techniques, and the overall contribution may be more suitable for a strong course project rather than meeting the bar for a top-tier venue like ICLR.

2. Potential limitations of the preference model:
The method models preferences via a latent utility function over objective values. However, it is unclear whether this formulation can always capture nuanced human preferences. For instance, two policies with similar objective values might still differ significantly in terms of perceived desirability by the DM, which this model may fail to distinguish.

3. Assumption of stationary preferences:
The experiments assume that the human’s preferences remain fixed throughout the optimization process. In realistic human-in-the-loop scenarios, preferences often evolve as users observe different policies or as external conditions change. The proposed framework may therefore struggle to handle non-stationary or dynamically shifting preferences.

### Questions
Are there other ways to design the high level MDP

### Soundness
2

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
4

### Summary
In multi-objective reinforcement learning (MORL), the utility function of the decision maker is often unknown. In this context, MORL algorithms typically learn the whole coverage set of optimal trade-offs, so that the decision maker can select their preferred solution a posteriori. This can be impractical, as the coverage set scales exponentially with the number of objectives, and requires thorough exploration of the state-space to learn many different policies. This work proposes PBMORL, an interactive MORL approach that learns the utility function over time to reduce the search space. PBMORL keeps a population of policies optimized for different utility functions, and uses a Gaussian Process (GP) to estimate the utility function. It refines the GP using pairwise preference elicitation, and then updates the population of policies with a 50/50 split between the top performing policies according to the current estimate of the utility function, and policies optimizing on new utility functions (that are biased towards the utility functions of the top performing policies). This process is then repeated for a fixed number of iterations.

### Strengths
Overall, the authors make a laudable effort towards interactively learning the optimal policy wrt the decision maker's preferences, a setting that has been mostly investigated in multi-armed bandits [1-3] and not RL. This is an important topic, which could be realistic to put in practice, as they show that only 10 to 40 interactions with the user are enough to accurately model their preferences. I find the approach interesting, as it could drastically improve sample efficiency compared to learning the Pareto front. However, I found it very hard to assess the results of the diverse experiments, as they use different metrics, different baselines, with many of the details and comparisons spread across the appendix.

### Weaknesses
My main concerns are as follows:

a) Many of the design choices of the algorithm seem ad-hoc, and would benefit from some clarifications, e.g.:
  1. PBMORL uses a population of independent PPO agents, each optimized on their own scalarization weights. This is quite inefficient, as many of the experiences could be shared across policies. Existing methods, such as [4,5], instead condition the policies on scalarization weights, such that the same network can be used regarless of the utility function. Is there a reason for not doing the same here? Also, even though the authors mention a multi-objective variant of PPO (Algo3), it is unclear what is different from running independent PPO instances.
  2. The setting focuses on a weighted sum over objectives as scalarization function. In that case, there is no need to model the utility function as a GP, which would handle non-linear utility functions. I am curious as to why methods from prior work in the interactive multi-objective bandits literature were not used, such as Bayesian logistic regression [1] or particle filtering [3]. Incorporating prior knowledge (i.e., the fact the the utility function is linear) into the belief should reduce the number of queries required to accurately predict the decision maker's utility function.
  3. During Step 4 of preference translation (line 235), weights are sampled evenly then biased towards promising weights. Since the GP naturally encompasses uncertainty, why not sample weights from the belief posterior? This would seem like a more principled approach to weight selection.

b) I found it hard to assess the results of the diverse experiments:
  1. The authors propose an "approximation accuracy" metric $\epsilon^\star$  and "average accuracy" metric $\bar{\epsilon}$. I don't understand the insights that can be gained from $\bar{\epsilon}$, as the baselines approximate the whole Pareto front, which of course will result in worse $\bar{\epsilon}$ values than PBMORL, that focuses on one region of interest. Moreover, the optimal policy used to compute both $\epsilon^*$ and $\bar{\epsilon}$ is called the "golden policy" (Appendix C5). But the values of the golden policy are defined by arbitraty, extreme values that are not guaranteed to be attainable in practice.
  2. The scalarized PPO baseline (Section 4.1.1) is run with 5 arbritrary weight combinations, who may not correlate with the optimal policy. As such, it us unsurprising that the learned policies may underperform compared to PBMORL, which actually learn the weights of the decision maker.
  3. Appendix D5 shows fuzzy preferences, and shows coverage sets of solution where "$f_1$ is weakly preferred". This seems to indicate that the GP does a good job at learning weights, but it does not show how close the learned weights are to the true weights, nor does it show how close the corresponding policy is to the optimal policy (the policy trained on the ground-truth weights).
  4. From my understanding, except D5 and Section 4.1.1, all the other experiments focus on optimizing one of the objectives (eg, "$f_1$ is preferred"). And so, the vast majority of the experiments involving multi-objective baselines focus on learning extreme policies that maximize a single objective, not trade-offs. I believe this simplifies the core MORL problem of balancing conflicting objectives.
  5. Section 4.1.1 contains extensive reward engineering, trying to fit basic and complex reward functions to optimize the objectives. I would appreciate comments on the reason why this has been done, since the advantage of using MORL is that you can optimize multiple, interpretable objectives, and should not have to perform reward engineering.
  
I believe a more principled way to evaluate PBMORL would be to:

a) sample random weights that are considered the (a priori unknwon) decision maker's utility function

b)
  1. train PBMORL, where the queries to the decision maker are resolved by saying that $\pi_1$ is preferred over $\pi_2$ if the utility of $\pi_1$ is higher than the one of $\pi_2$ (using the ground-truth weights)
  2. train an oracle (e.g., PPO) directly using the ground-truth weights that serves as an upper bound
  3. Similar to the spirit of Appendix D3, use the whole query budget a priori (e.g., on random returns sampled from the objective space), then train an agent (e.g., PPO) on the weight from the belief
  4. train preference-based baselines and MORL baselines (this is already done in the paper)

c) compare the utility of all the trained policies (using the ground-truth weights), and compare the expected utility loss [luiza] wrt the optimal policy (i.e., the oracle).

I apologize for the long review, and hope the authors are not discouraged by all the comments, as overall I like the proposed approach, and will gladly change my score depending on the discussion with the authors and the clarifications they can provide.

[1] Roijers, D. M., Zintgraf, L. M., & Nowé, A. (2017). Interactive thompson sampling for multi-objective multi-armed bandits. In International conference on algorithmic decision theory (pp. 18-34). Cham: Springer International Publishing.

[2] Roijers, D. M., Zintgraf, L. M., Libin, P., Reymond, M., Bargiacchi, E., & Nowé, A. (2020). Interactive multi-objective reinforcement learning in multi-armed bandits with gaussian process utility models. In Joint European conference on machine learning and knowledge discovery in databases (pp. 463-478). Cham: Springer International Publishing.

[3] Reymond, M., Bargiacchi, E., Roijers, D. M., & Nowé, A. (2024). Interactively Learning the User's Utility for Best-Arm Identification in Multi-Objective Multi-Armed Bandits. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (pp. 1611-1620).

[4] Abels, A., Roijers, D., Lenaerts, T., Nowé, A., & Steckelmacher, D. (2019). Dynamic weights in multi-objective deep reinforcement learning. In International conference on machine learning (pp. 11-20). PMLR.

[5] Yang, R., Sun, X., & Narasimhan, K. (2019). A generalized algorithm for multi-objective reinforcement learning and policy adaptation. Advances in neural information processing systems, 32.

### Questions
please see the comments above

### Soundness
2

### Presentation
2

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
This paper proposes a framework for MORL that interactively incorporates DM's preferences during policy optimization. The paper consists of 3 stages: 1. The Seeding stage initializes diverse policies; 2. The Preference Elicitation stage queries DM and translates preferences into weight vectors 3. The Policy Optimization stage refines policies in parallel. Evaluation shows it achieves superior alignment performance compared to baselines.

### Strengths
1. The idea of incorporating querying the DM's preferences into the MORL learning pipeline is novel. The Preference Elicitation stage actively queries the DM via pairwise policy comparisons and translates these responses into weight vectors. The approach avoids the need to design scalarization. 

2. Extensive empirical evaluation covers a wide range, showing that it outperforms the baselines.

### Weaknesses
1. A key concern is that, despite presenting itself as a multi-objective RL framework, the method effectively reduces the problem to a form of preference-weighted single-objective optimization after the elicitation stage. While the Preference Elicitation stage is novel, the final policy optimization is conducted in a scalarized utility space. Although the algorithm remains multi-policy in structure (e.g., using multiple weighted tasks in MOPPO), the search is guided toward a narrow vector space, ignoring global trade-offs. This raises questions about the "multi-objective" claim, especially in settings where a full Pareto frontier might be desired.
2. While the framework is claimed as "human-in-the-loop", all user preferences are simulated with predefined vectors. An empirical study with actual humans would strengthen the paper.  
3. The writing tends to be verbose and redundant. A tighter structure and more concise explanations would improve readability.

### Questions
1. The proposed method narrows the optimization to the learned preference region, whereas baselines (e.g., scalarized PPO) attempt to cover the full objective space. This may make the comparison unbalanced, especially since the method benefits from more targeted exploration. Could the authors clarify whether the baselines were also given any access to preference vectors or if they were evaluated using the same simulated DM? How do you ensure a fair evaluation?
2. How does PBMORL perform when queried with test-time preferences that differ from the training-time preference region? For example, if a DM changes their mind or reveals a new preference vector after training, can the method generalize to this unseen region? Or is it limited to the narrow region learned during interaction?

### Soundness
3

### Presentation
2

### Contribution
3
