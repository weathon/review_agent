# CURATE: Automatic Curriculum Learning for Reinforcement Learning Agents through Competence-Based Curriculum Policy Search

- Decision: Reject
- Scores: 2, 2, 2, 2, 6

## Abstract
Due to fundamental exploration challenges without informed priors or specialized algorithms, agents may be unable to consistently receive informative rewards, leading to inefficient learning. To address these challenges, we introduce CURATE, an automatic curriculum learning algorithm for reinforcement learning agents designed for difficult target task distributions. Through "exploration by exploitation," CURATE dynamically scales the task difficulty to match the agent's current competence. By exploiting its current capabilities that were learned in easier tasks, the agent improves its exploration in more difficult tasks. Our key insight is that the performance increase in tasks that are close to those used for training is inversely proportional to their difficulty, and an agent that chooses a nearby distribution of the easiest unsolved tasks at any given time can automatically induce an easiest-to-hardest curriculum. To achieve this, CURATE conducts policy search in the task space to learn the best task distribution for training the agent. As the agent's mastery grows, the learned curriculum adapts in an approximately easiest-to-hardest and task-directed fashion, efficiently culminating in an agent that can solve the target tasks. Our experiments across three domains of varying task parameterization and dimensionality demonstrate that CURATE learns highly effective curricula, matching or exceeding prior curriculum methods in target task performance. Moreover, CURATE curricula are effective beyond solving the difficult target tasks, yielding broadly capable agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CURATE, an automated curriculum generation method for sparse RL domains. Using relative entropy policy search, CURATE searches for a policy in the task space that prioritizes the easiest unsolved tasks as the policy improves. This induces an approximately easiest-to-hardest curriculum, where difficulty is inversely proportional to the discounted return.  Experiments on MiniGrid MultiRoom and Procgen Leaper demonstrate that CURATE improves sample efficiency in comparison to domain randomization and some UED methods. The paper also introduces a benchmark for systematic evaluation of curriculum learning methods.

### Strengths
- The paper has a detailed related work covering a wide range of curriculum learning works. 
- The proposed method seems novel in its framing of exploration by exploitation for curriculum learning for RL.
- The domains of interest in the experiments seem difficult to tackle due to their sparse-reward nature.

### Weaknesses
- The paper is difficult to follow, the method section has a circular structure, and is confusing.
- The notion of difficulty is limited to a niche group of sparse reward domains where the number of steps to reach the goal is directly associated with the difficulty. Although mentioned in the limitations section, this is a clear obstacle against the practicality of the proposed approach.
- Despite claiming that the authors scale existing ideas to multidimensional settings, they only do it up to two-dimensional task spaces.
- There is no ablation for the impact of different reward components for training the curriculum policy.

### Questions
- What ablations would demonstrate the use of having different rewards for different stages?
- Is there a reason why ALP-GMM cannot handle multi-dimensional task spaces? 
- Could you please give me an example domain where the assumption on the difficulty ordering doesn't hold, and explain how CURATE would perform there?

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
4

### Summary
The paper introduces CURATE, an automatic curriculum learning method for reinforcement learning tasks. The approach learns a curriculum policy and updates it using REPS to shift the task distribution toward the easiest unsolved tasks for the agent, resulting in an emergent easiest-to-hardest training progression. However, the approach requires a known and ordered task space to function. Experiments on MiniGrid MultiRoom and Procgen Leaper show that CURATE improves sample efficiency and outperforms several baselines.

### Strengths
- The paper tackles an important  challenge in reinforcement learning, how to automate curriculum generation to improve training efficiency in sparse-reward settings.
- The paper adapts the Procgen environment for curriculum-based evaluation. While only three environments are used (and mainly one discussed in the main text), these setups could still serve as useful benchmarks for the community and support more consistent evaluation of curriculum learning methods.

### Weaknesses
W1: The paper is difficult to follow. The description of the proposed method in the main text is mostly narrative, intermixed with mathematical formulas that are not well connected to the algorithmic intuition. The pseudocode in the appendix is poorly formatted and hard to read. Furthermore, the paper lacks any theoretical justification or analysis of why the proposed approach should converge or be optimal.

W2: Although the authors acknowledge this limitation, the requirement of having a known and ordered task space is a major assumption. CURATE depends on explicit task parameters that can be continuously ordered by difficulty, an assumption that many other baseline methods do not exploit. This significantly narrows the generality of the approach.

W3: The evaluation procedure appears to exclude the cost of repeated task evaluations from the total training budget. This underestimates the true computational cost of CURATE compared to baselines. Moreover, for this kind of problem setup, evaluation alone can sometimes be sufficient to train an agent, for example, in Evolution Strategies (ES), where learning is driven entirely by repeated evaluations of policy performance without a separate training phase. This makes the comparison potentially unfair, as CURATE benefits from frequent evaluations but seems not include them in its reported training budget.

W4: The evaluation of the proposed approach requires the definition of a solved threshold, which in most cases means that we already need to know how to solve the problem in order to define it. This requirement limits the range of domains where the approach can be applied.

W5: The testbed consists of only two environments: MiniGrid MultiRoom and the modified-for-curriculum Procgen Leaper. This is a very small set of tasks, and the task spaces themselves are quite limited (e.g., 4 possible rooms in MiniGrid and a two-dimensional task space with 3 road lanes and 3 water lanes for Leaper). Overall, the testbed is too restricted to demonstrate scalability. Moreover, the paper ignores several benchmarks from curriculum learning research, such as those used in the ACCEL paper.

W6: It seems that the approach is not applicable to hierarchical curriculum setups (e.g., MineCraft or Crafter), where preventing catastrophic forgetting is crucial. These environments typically require mechanisms for skill retention or replay, as presented, for example, in PLR. CURATE does not appear to include such mechanisms.

### Questions
Q1: How were the baselines tuned or adapted to the tasks being solved?
For example, in PLR replay there are several possible prioritization methods (e.g., policy_entropy, least_confidence, min_margin, gae, value_l1, etc.). Which specific configuration was used in the experiments? Additionally, what does the “PLR prioritization - rank” entry in Table 3 of the appendix refer to?

Q2: Does the evaluation budget include the cost of all curriculum evaluations used? If not, can the authors provide an estimate of how many additional environment steps these evaluations require?

Q3: Can CURATE be adapted to environments with continuous task parameters, where task difficulty changes smoothly (e.g., obstacle density or agent's speed)? How would the Gaussian curriculum policy and REPS update handle such cases?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CURATE, an automatic curriculum learning algorithm for reinforcement learning agents. CURATE frames curriculum generation as a policy search problem in task space, where a Gaussian curriculum policy selects tasks based on the agent’s current competence. By rewarding performance on “unsolved but easiest” tasks and updating the task distribution using Relative Entropy Policy Search (REPS), the method aims to produce an approximately easiest-to-hardest progression. Experiments on MiniGrid MultiRoom and Procgen Leaper compare CURATE with domain randomization, handcrafted incremental curricula, and automatic approaches such as Robust PLR and ACCEL. Results show that CURATE can learn task-directed curricula and achieve better sample efficiency than the unsupervised baselines.

### Strengths
1. Proposes a clear formulation of curriculum policy search based on competence, with explicit loss terms and an implementable algorithm.

2. Includes quantitative comparisons with both non-learning and learning-based curriculum methods.

3. Uses two distinct RL domains (grid-based and image-based) to demonstrate the approach.

4. Provides full pseudocode and implementation details, which supports reproducibility.

### Weaknesses
1. Unclear novelty and positioning - The abstract and introduction do not clearly articulate what distinguishes CURATE from prior teacher-student or learning-progress methods (e.g., ALP-GMM, bandit-based curricula).

2. Lack of theoretical grounding - The main insight that “performance on nearby tasks increases inversely with difficulty” is presented empirically, without discussion of the assumptions or limits under which it holds.

3. Potential local minima - Because CURATE samples from tasks the agent already performs well on, it is unclear how it avoids getting stuck in competence plateaus or repetitive easy tasks.

4. Assumption of monotonic difficulty - The claim that “easier tasks yield higher returns” may not generalize.

5. Reward threshold - R_S is critical to defining competence but is only described in the appendix. Its determination, sensitivity, and generality are not discussed in the main text.

6. Key objective terms deferred to appendix – Loss components L_diff and L_dist are central to how CURATE functions but are explained only in Appendix B.1, with minimal intuition in the main body.

7. Limited experimental scope - Only two domains are evaluated, both with low-dimensional, discrete tasks, making it difficult to claim broader applicability.

8. Missing ablations and trajectory comparisons – There is no ablation of loss terms or analysis of the actual task trajectories compared to other methods.

9. Ambiguous statement about “best initial tasks” – The paper claims CURATE finds the “best” initial task set (line 365) without defining what constitutes “best” or whether it is benchmarked against any ground truth.

### Questions
1. How is R_S chosen in practice, and how sensitive is performance to this value?

2. What prevents CURATE from remaining in a local optimum of easy tasks that yield consistently high returns?

3. Under what formal assumptions does the “nearby task improvement” insight hold true?

4. Why were the key curriculum objective terms (L_diff, L_dist) placed in the appendix, and how do they affect learning if removed or modified?

5. Can you provide quantitative comparisons of the task trajectories (sequence or spread of sampled tasks) between CURATE and baselines such as ACCEL or PLR⊥?

6. Would the method still perform effectively in task spaces that are not strictly monotonic in difficulty or where difficulty is non-separable across dimensions?

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
3

### Summary
The paper proposes a curriculum learning approach for RL tasks where the curriculum is evaluated based on task difficulty.

### Strengths
* The proposed method is intuitive.

### Weaknesses
* The framework has a strong assumption that the environments are fully parameterized within space $\Theta$. This raises practical concerns about whether such an assumption holds for generic decision-making tasks. More non-game examples and also clarifications on the limitations of the method would help address the concern. 
* Critical values such as initial $\mu_\theta, \Sigma_\theta$ are treated as hyperparameters to be tuned. Either ablations on showing robustness of these hyperparameters or discussions on how to choose them would make the framework more practical for general tasks. 
* Results are validated on simulated tasks either with low-dimensional task space, or with human-engineered procedural environment distributions, with oracal access to the procedural parameters to generate the environments being benchmarked. These simulated domains are much simpler than real-world or general decision-making tasks.

### Questions
* The environment distribution seems to be a Gaussian with mean $\mu_\theta$ and covariance $\Sigma_\theta$, although I didn't find it clearly defined in the manuscript (but it's possible I missed it).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CURATE, an automatic curriculum learning algorithm for reinforcement learning. CURATE conducts policy search in task space, dynamically adapting the curriculum based on the agent’s competence to progress from easier to harder tasks. Evaluated on MiniGrid MultiRoom and the Procgen Leaper environments, CURATE achieves higher sample efficiency than other baselines.

### Strengths
1. CURATE is conceptually well-motivated, formulating curriculum design as a policy search over task distributions guided by agent competence.
2. The paper provides a careful qualitative and quantitative examination of the learned curricula, including how the agent starts with easier tasks, maintains narrow and focused task distributions, and progresses in a generally easiest-to-hardest trajectory. The visualizations  effectively illustrate the evolution and path of the curriculum in the task space.
3. The paper introduces the Procgen Curriculum Suite, a new environment designed to evaluate curriculum learning. Given the lack of effective tools for rapid assessment of curriculum methods on parameterized tasks, this contribution appears promising.

### Weaknesses
1. A key limitation of CURATE is its reliance on a fully defined task space with clear difficulty gradients, limiting its use in many RL environments that lack this or need significant effort to specify. This limitation is noted but not fully explored (see Questions 1, 2).
2. The baseline selection is not entirely fair. ACCEL and PLR are designed for generalization across many tasks (retaining and revisiting previously learned tasks throughout training) and lack prior knowledge of task difficulty until attempting to learn each task. These constraints inherently limit their sample efficiency compared to CURATE, which assumes difficulty is known a priori. A more equitable comparison would be against TSCL, which evaluates performance across all tasks and is designed for environments with a small number of tasks. This would balance CURATE's advantage of having prior difficulty information against TSCL's ability to evaluate all tasks .
3. CURATE is only evaluated on two relatively simple environments with few parameters, which were specifically selected for this method. It would be more informative to test CURATE on widely used benchmarks for curriculum learning, such as BipedalWalker, which includes multiple configurable parameters and has been used to assess POET and ACCEL. Broader evaluation would better demonstrate CURATE’s generalizability and robustness. 
4. The description of the method lacks clarity and is difficult to understand on the first reading. As a result, readers often need to revisit the pseudocode for better understanding. Including a visual diagram of the method would make the section much clearer and improve the overall presentation.

References
- [ACCEL] Parker-Holder, J., Jiang, M., Dennis, M., Samvelyan, M., Foerster, J., Grefenstette, E. and Rocktäschel, T., 2022, June. Evolving curricula with regret-based environment design. In International Conference on Machine Learning (pp. 17473-17498). PMLR.
- [PLR] Jiang, M., Grefenstette, E. and Rocktäschel, T., 2021, July. Prioritized level replay. In International Conference on Machine Learning (pp. 4940-4950). PMLR.
- [ TSCL ] Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher-Student Curriculum Learning. IEEE Transactions on Neural Networks and Learning Systems, 31(9):3732–3740, 2019.
- [ POET ] Wang, R., Lehman, J., Clune, J. and Stanley, K.O., 2019, July. Poet: open-ended coevolution of environments and their optimized solutions. In _Proceedings of the genetic and evolutionary computation conference_ (pp. 142-151).

### Questions
Questions on task parametrization
1. How could the constraint linking task complexity to the parameter value be addressed?
2. If the principal axes are not perfectly disentangled, how robust is CURATE? Does it degrade gracefully or collapse?

Questions on environments

3. What criteria were used to select the parameters for task configuration in Procgen?
4. Have you experimented with the other curriculum-based version of Procgen (C-Procgen)? In what key aspects does  Procgen Curriculum Suite differ from theirs?
5. Why were the experiments not performed in a standard environment where other curricula were evaluated? Is this decision connected to the method’s main limitation concerning task parametrization?

Questions on curriculum performance

6. If the environment has a large task space, does the model forget simpler tasks over time since the method lacks a built-in mechanism to revisit them?

References
- [C-Procgen] Tan, Z., Wang, K., Wang, X., 2023. C-Procgen: Empowering Procgen with Controllable Contexts. [https://doi.org/10.48550/arXiv.2311.07312](https://doi.org/10.48550/arXiv.2311.07312)

### Soundness
3

### Presentation
3

### Contribution
3
