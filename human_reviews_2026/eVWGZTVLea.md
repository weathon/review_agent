# Learning from Less: Guiding Deep Reinforcement Learning with Differentiable Symbolic Planning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
When tackling complex problems, humans naturally break them down into smaller, manageable subtasks and adjust their initial plans based on observations. For instance, if you want to make coffee at a friend’s place, you might initially plan to grab coffee beans and go to the coffee machine. Upon noticing that the machine is full, you would skip the initial steps and proceed directly to brewing. In stark contrast, state-of-the-art reinforcement learners, such as Proximal Policy Optimization (PPO), lack such prior knowledge and therefore require significantly more training steps to exhibit comparable adaptive behavior. Thus, a central research question arises: \textit{How can we enable reinforcement learning (RL) agents to have similar ``human'' priors, allowing the agent to learn with fewer training interactions?} To address this challenge, we propose \textbf{d}ifferentiable s\textbf{y}mbolic p\textbf{lan}ner (Dylan), a novel framework that integrates symbolic planning into Reinforcement Learning. Dylan serves as a reward model that dynamically shapes rewards by leveraging human priors, guiding agents through intermediate subtasks, thus enabling more efficient exploration. Beyond reward shaping, Dylan can work as a high-level planner that composes (logic) options to generate new behaviors while avoiding common symbolic planner pitfalls such as infinite execution loops. Our experimental evaluations demonstrate that Dylan significantly improves RL agents' performance and facilitates generalization to unseen tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
DYLAN is a differentiable symbolic planner that integrates human prior knowledge into reinforcement learning through structured logical rules. It functions as both a reward model providing interpretable guidance for sparse-reward environments and an adaptive planner that composes logical options for zero-shot generalization. Experiments demonstrate improvements in sample efficiency and task performance over standard RL methods.

### Strengths
Originality
- The combination of RL and a differentiable symbolic planner is original and novel. The concepts of differentiable symbolic planners are elaborated and verified, demonstrating the potential of neural-symbolic approaches in RL.
- The results show improvements over other baselines.

Clarity
- The paper is clear and well-written.  It provides clear illustrations to depict the procedure.

### Weaknesses
While DYLAN presents a novel differentiable symbolic planning approach for reinforcement learning, several methodological concerns limit its contribution.

Baseline

- Although differentiable symbolic planners represent a novel contribution, the paper inadequately addresses existing hybrid RL-symbolic integration methods, and the baseline only includes PPO and A2C A comprehensive comparison with established approaches is essential to properly contextualize DYLAN's contributions and demonstrate advancement over existing techniques.

- The rules involve human supervision, where the improvement might come from carefully hand-crafted rules rather than from your framework design, which mitigates the contribution. In addition, there are some works that already focus on automatic rule induction to eliminate human intervention. Therefore, beyond PPO/A2C comparisons, evaluation should include methods incorporating hand-crafted rules and automatic rule induction systems, which are more relevant benchmarks for this domain.

Evaluation

- The evaluation is constrained to relatively simple grid environments, failing to demonstrate scalability to complex, high-dimensional domains. This limitation raises questions about real-world applicability and the method's practical utility beyond toy problems.

Scalibiity

- The framework presents a critical inconsistency: while differentiable symbolic reasoning is positioned as a scalability solution, the concurrent requirement for human supervision to refine rules creates a bottleneck that contradicts scalability claims. This dependency becomes prohibitive as domain complexity increases, undermining the approach's practical viability.

Overall, DYLAN presents an interesting conceptual framework, but requires more comprehensive experimental validation and methodological refinement to establish robustness and competence. The current evaluation framework is insufficient to support the claimed contributions without addressing these fundamental limitations.

### Questions
See above.

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
This paper presents DYLAN, a differentiable symbolic planner that helps reinforcement learning agents learn faster by providing them with structured, interpretable rewards. DYLAN combines symbolic planning with deep RL to provide subgoal-based rewards. It can also operate as a planner on its own, learning when to behave more like DFS or BFS, composing sub-policies for new tasks, and inferring goals from demonstrations. 
The experiments in MiniGrid show improvements in sample efficiency, robustness when some knowledge is missing, and good generalization to unseen combinations of tasks.

### Strengths
This is a well-motivated paper that connects symbolic reasoning with RL in a meaningful way. 
The motivation is clear: RL struggles with sparse rewards, and symbolic priors can help address that.
The method is technically consistent and reasonably well explained.
Results on MiniGrid show faster learning and better stability compared to PPO and A2C.
The planner’s ability to adapt search strategies (DFS/BFS), compose sub-tasks, and infer goals is impressive and shows flexibility.

### Weaknesses
- Missing a formal definition of the setting (e.g., how is a state and goal condition defined?)
- The practical impact is limited by the narrow scope of the experiment
- The experiments are limited to a small, discrete MiniGrid task; it’s unclear how well DYLAN would scale to continuous control or vision-based domains.
- Important baselines are missing, especially Reward Machines or intrinsic motivation approaches (RND, ICM), which deal with similar problems.
-There’s no discussion or measurement of computational cost. Since the adaptive version reasons every step, so runtime overhead should be reported.
-Sensitivity to the shaping parameters (λ, ω) and reasoning depth isn’t analyzed.
- The description of how DYLAN builds and prunes its candidate plan set is too brief.


Minor comments: 
- For the experiments, report the runtime and compute overhead of both DYLAN variants compared to PPO/A2C. 
- Add ablations for λ, ω, and reasoning depth.

### Questions
Q1- The environment provides symbolic states directly, and the symbolic rules are generated by GPT-4o and manually verified. This makes DYLAN hard to apply in realistic settings. How can this be mitigated?

Q2 - What is the scope of problems and the definition of the kind of problems this framework aims to address. What is the input to the method?

Q3 - Why use DPF and BFS and not search in relaxed versions of the problem or perform a heuristic search ?

Q4-  Does the Dylan component learn in a supervised learning method where the GT is created by running the BFS/DFS algorithm to search for the next subgoal? If so, then if the assumption is that the space is small enough to be searchable in a naive manner, then why do we need the Dylan component at all? In that case the policy from which we shape the reward can be figured out using the resulting path (of the DFS/BFS) from current state to the next subgoal.

### Soundness
2

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
2

### Summary
This paper proposes DYLAN (differentiable symbolic planner), a framework that integrates symbolic planning with RL. DYLAN serves as a high-level planner and reward shaping method that improves sample efficiency of baseline RL algorithms on MiniGrid tasks. The authors claim that DYLAN facilitates generalization in multi-task settings, is robust to partial observability, and can learn from demonstrations.

The paper is well organized and easy to follow. The results show that DYLAN improves sample efficiency over baseline methods. However, there are some limitations with respect to scalability and evidence for generalizability. Overall, I recommend against acceptance due to these limitations.

### Strengths
The authors present convincing evidence that DYLAN can improve sample efficiency when used as an auxiliary task on top of existing RL methods. Both the standard and adaptive versions of DYLAN demonstrate a substantial performance boost over the baselines in the tasks shown (Figure 3, Figure 4, Table 1).

The method appears to be robust to partial information and adaptive to different strategies, which are valuable properties for practical deployment.

The plans generated by DYLAN can improve interpretability of agent behavior, which is important if the method were to be used in real-world or safety-critical tasks.

### Weaknesses
Though the results indicate that DYLAN can infer the underlying goal from expert demonstrations (Q6), I'm not sure if this necessarily supports the claim that DYLAN has a "unique ability to generalize from demonstrations, a property beyond the scope of existing RL or hierarchical approaches." The ability to infer a goal does not necessarily correspond to the ability to stitch trajectories together. Additionally, existing RL approaches do demonstrate stitching and generalization of demonstrations, as shown in recent work by Li et al. (https://proceedings.mlr.press/v235/li24bf.html), Myers et al. (https://arxiv.org/pdf/2509.20478), and Ki et al. (https://dl.acm.org/doi/full/10.1145/3747227.3747233).

As the authors mention in Appendix A, the need to use GPT-4o and human verification to generate and transform candidate game rules is a limitation for the practicality and scalability of the method. It is unclear whether the method can scale to environments where policy primitives are harder to generate and verify.

### Questions
**Questions for clarification:**

1. Why does Figure 5 show loss rather than success rate of the chosen strategy over training?
2. Can you clarify the setup of the experiments described in Section 4.3 (DYLAN's compositionality paragraph)? The tasks were Key Retrieval, Red Door Reaching, Goal Reaching, and Safe Goal Reaching. For the baseline agents trained to navigate to the goal, how was the goal defined for each task? Were the policy primitives generated by GPT-4o as well?
3. (Line 448) "DYLAN successfully composes logic options to generalize across diverse tasks, whereas the PPO, A2C, hDQN and Galois agents, which are trained solely for goal navigation, fail to generalize to tasks requiring more complex behavior." While DYLAN does show better success rates for the multitask settings (Table 1), could you elaborate on how these logic options demonstrate novel behaviors and the ability to generalize across diverse tasks? Some example logic options could also be helpful here.

---

**Minor comments:**

- The connection to human priors and the claim that DYLAN "aligns with human intent" (lines 70-71) is a bit tenuous since LLM priors do not necessarily correspond to human priors.
- The best-performing model mark on Table 1 is in the wrong position for the first column.

---

**Suggestions for improvement (not factored into score):**

1. To provide evidence for the claim that DYLAN can generalize from demonstrations, the authors could conduct an experiment which tests whether DYLAN can stitch together components of plans from expert demonstrations to solve a new task.

### Soundness
2

### Presentation
3

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
This paper proposes Dylan to integrate symbolic planning into Reinforcement Learning. Dylan serves as a reward model and a high-level planner, enabling efficient exploration and generating new behaviors. Experimental evaluations demonstrate that Dylan improves the performance and facilitates generalization to unseen tasks in minigrid environments.

### Strengths
The core idea of integrating symbolic planning into RL through a differentiable framework is novel and promising.

Experiments show that the proposed method is able to generalize to unseen tasks and compose new behaviors without retraining, which is an advantage over traditional RL approaches.

### Weaknesses
- [Theoretical analysis] Regarding Dylan as a reward model, it is necessary to provide a theoretical analysis to show that the new reward function doesn't change the optimization objective compared with pure PPO/A2C.

- [Single test domain] The experiments are currently limited to the MiniGrid environment suite. Expanding the evaluation to include more diverse and complex environments, especially those with continuous action spaces (for example, robotics manipulation tasks), would demonstrate the method's scalability and robustness.

- [Insufficient comparison baselines] This paper should compare to more recent related works. Results over 3 random seeds are insufficient to validate the results.

- [Imperfect or Noisy knowledge] The paper only considers a partial knowledge setting, discussing how the method handles incomplete or noisy knowledge would enhance its practical applicability.

- [Clarity] The paper's clarity needs to be improved. For example, subsection 4.2's text is inconsistent with Figure 4. The label 'PPO_reasoner'  is nowhere described. Instead, static/adaptive reward is used. Also, the notation in Table 1 is incorrectly used after the algorithm.

### Questions
Would the role of Dylan as a reward model and a planner be combined? What's the performance then?

Why does the experiment section contain results of different baselines in each subsection? For example, why are there no training curves for all methods in Table 1? Why doesn't section 4.2 show the results of the A2C-based static/adaptive reward?

### Soundness
2

### Presentation
3

### Contribution
2
