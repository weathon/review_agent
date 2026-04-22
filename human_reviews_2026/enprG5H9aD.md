# SLAP: Shortcut Learning for Abstract Planning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 8

## Abstract
Long-horizon decision-making with sparse rewards and continuous states and actions remains a fundamental challenge in AI and robotics. Task and motion planning (TAMP) is a model-based framework that addresses this challenge by planning hierarchically with abstract actions (options). These options are manually defined, limiting the agent to behaviors that we as human engineers know how to program (pick, place, move). In this work, we propose Shortcut Learning for Abstract Planning (SLAP), a method that leverages existing TAMP options to automatically discover new ones. Our key idea is to use model-free reinforcement learning (RL) to learn *shortcuts* in the abstract planning graph induced by the existing options in TAMP. Without any additional assumptions or inputs, shortcut learning leads to shorter solutions than pure planning, and higher task success rates than flat and hierarchical RL. Qualitatively, SLAP discovers dynamic physical improvisations (e.g., slap, wiggle, wipe) that differ significantly from the manually-defined ones. In experiments in four simulated robotic environments, we show that SLAP solves and generalizes to a wide range of tasks, reducing overall plan lengths by over 50\% and consistently outperforming planning and RL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Shortcut Learning for Abstract Planning (SLAP), a novel framework that enhances robotic problem-solving by synergizing traditional Task and Motion Planning (TAMP) with Reinforcement Learning (RL). Addressing the limitation that TAMP relies on pre-programmed and often inefficient skills, SLAP autonomously discovers more effective behaviors by analyzing the high-level structure of an abstract plan to identify potential "shortcuts." It then uses RL to train new, dynamic, low-level skills (such as "slapping" or "wiping" obstacles) to bridge these gaps in the plan. These learned shortcut policies are integrated back into the planner, enabling it to find significantly shorter and more efficient solutions. The key contributions are a dramatic reduction in plan length by over 50%, a high success rate on complex, long-horizon tasks where pure RL methods fail, and strong generalization to new tasks and objects not seen during training.

### Strengths
- Figure 1 is very well made and clear
- The explanation of how the work fits into the existing literature is very good, even for someone unfamiliar with the relevant literature
- The results seem pretty amazing in terms of improvement over baselines

### Weaknesses
- Figure 2 and 3 are pretty confusing in my opinion and it is not clear just from looking at the figure what the grey and black balls are supposed to represesent -- this should be made clearer so that it is understandable just from looking at the figure
- Some analysis of sample/computational efficiency should be provided in the main text and results (how many extra samples, how much extra computation, how much extra training time is required to learn these shortcuts). There may be an appendix related to this but I believe at least a summary should be included in the main text so that the reader can know if potentially this improvement in performance is just from giving extra data to the learning algorithm through the shortcut learning policies 
- Some discussion of how this algorithm might be used when there is not access to a simulator (or if it just not possible in that case) should also be included, or what additional pieces would be needed in the case of no simulator
- I am unsure of the novelty of the approach. It seems like SLAP is just a way to choose which option policies to learn during training? The relationship/difference to previous work on hierarchical RL should be made clearer in the Related Work or Introduction section

### Questions
Questions/suggestions are included in the "Weaknesses" section. I am very open to raising my score if my concerns are addressed

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
This paper proposes SLAP (Shortcut Learning for Abstract Planning), a method that integrates model-free reinforcement learning (RL) with task and motion planning (TAMP) to automatically discover shortcut policies between abstract states. The main idea is to augment a symbolic planning graph with additional edges learned via RL. Each shortcut policy is trained to transition directly between two abstract states which effectively bypasses multiple intermediate actions and reduces plan length.

### Strengths
- The intuition of the paper (discovering shortcut connections in an abstract planning graph) is conceptually sound and easy to grasp. It is a simple yet effective idea that directly addresses the inefficiency of long hierarchical plans in TAMP frameworks.

- The approach can produce genuinely new high-level actions that are not part of the manually defined option set (e.g., slap). This demonstrates the potential of the method to extend the agent’s action set beyond what is explicitly encoded by human designers.

- The paper is clearly written and well-structured, making the overall framework and experiments easy to follow.

- Long horizon is one of the reasons why RL does not work effectively. Solving small RL problems with short horizons makes sense.

### Weaknesses
A primary weakness of this work lies in its conceptual alignment with the TAMP paradigm and the choice of experimental baselines.

1. Conceptual Dissonance with TAMP: The core philosophy of TAMP is to find plans that are valid with respect to a given symbolic model, ensuring that high-level action sequences are grounded and physically feasible according to predefined rules. The proposed method, SLAP, learns "shortcut" policies (e.g., "slapping" a tower) that achieve a goal state by operating outside of this symbolic structure. While this can yield more efficient plans, it fundamentally reframes the problem from one of constrained, symbolic planning to one of unconstrained trajectory optimization. This raises the question of whether SLAP is improving upon a TAMP solution or solving a different, less constrained problem altogether. The potential for emergent, "destructive," or unpredictable behaviors is at odds with the safety and predictability that motivates the use of TAMP in the first place.

2. Inadequate Baselines: The comparison to model-free RL baselines is not particularly insightful. It is well-established that vanilla RL algorithms struggle with the long-horizon, sparse-reward problems that TAMP is designed to solve. Since SLAP heavily leverages the strong structural priors from the TAMP framework (the abstract graph and initial options), outperforming these baselines is an expected outcome and does not sufficiently isolate the contribution of the shortcut-learning mechanism itself.

In summary, the paper positions itself as a hybrid TAMP-RL method, yet it diverges from the core principles of TAMP and compares against RL methods on a task setup where they are known to be inefficient. A clearer articulation of its contribution, perhaps as a method for discovering new symbolic operators rather than just bypassing them, and a comparison against more relevant baselines would significantly strengthen the paper.

Below are some further remarks:
- The methodology depends on finding two abstract states where there is at least a success rate of K_{rollout}/N_{rollout} transitioning from one to another state by taking random actions. I find this very restrictive and dependent on the abstract planning graph generation.

- The paper lacks a clear and fair baseline. It does not make sense to use pure RL algorithms as baselines for such complex, long-horizon tasks that fundamentally require high-level task planning. Even in simpler motion-planning problems, RL methods typically demand more training steps than the 500k used here to achieve meaningful performance. Since SLAP is built on a TAMP framework with strong structural priors, comparisons to model-free RL is not ideal. That’s why it cannot be claimed that the proposed method increases the performance. The paper already includes a comparison with the no-shortcut case. But this is more of an ablation study rather than a baseline. Finally, reporting success rates over only 10 trials per task is insufficient for statistical reliability; larger-scale evaluations or confidence intervals are needed to support the claimed improvements.

- The core idea of using RL to learn shortcut policies between abstract states is not a significant conceptual contribution. The method essentially applies standard model-free RL within a predefined TAMP structure to learn transitions that skip multiple existing options. While this integration is practically useful, it does not introduce new algorithmic insights or theoretical advances in either reinforcement learning or task and motion planning. The contribution is therefore more of an engineering combination of known components than a novel methodological development.

- The notation in Section 4.4 is not intuitive. add(â) and del(â) denote sets of atoms, whereas rel(â) denotes a set of objects involved in those atoms. The subsequent use of expressions such as add(â_train)[σ(rel(â_train))] ⊆ add(â_eval) is confusing (the square-bracket substitution notation is unconventional and not clearly defined). I can understand what is meant but a more rigorous and transparent mathematical formulation (or pseudocode) of the object substitution process would significantly improve clarity.

- The paper claims that learned shortcut policies can be reused on new objects through an object-substitution mapping σ, which aligns the add/del atom sets after substitution. However, the method for finding σ is not clearly described. When multiple objects are involved, determining consistent multi-object mappings becomes nontrivial, yet the paper provides no explanation of how σ is computed, whether it must be one-to-one, or how ambiguities are resolved. This lack of detail makes it difficult to evaluate the reliability and scalability of the object substitution mechanism proposed in Section 4.4.

- Figure 4 is hard to read. I would create two figures: one with graphs, one showing a learned shortcut (and improve this image).

- The analysis for Q5 (“Which RL design decisions are important for learning shortcuts?”) is limited in scope. The section “Shortcut Policy Learning Analysis” examines only one factor (whether shortcut policies are trained independently or as a shared universal policy). Other key RL design aspects (e.g., algorithm choice, reward shaping, exploration strategy) are not explored. As a result, the section provides a partial answer to Q5 and does not fully justify the general phrasing of the question.

- The claimed generalization ability of SLAP is limited. Section 4.2 describes “generalization over objects,” but the mechanism is a simple object-substitution procedure based on symbolic equivalence of add/del atoms. This allows policy reuse only when new tasks are structurally isomorphic to training ones. The method does not learn to generalize over varying object geometries, dynamics, or unseen relational structures (its transfer is purely symbolic). Consequently, the generalization claim overstates the scope of what the approach can handle.

- The abstract claims that SLAP “consistently outperforms planning and RL baselines”. As Table 1 demonstrates, SLAP matches the Pure Planning baseline in success rate (100%) and only improves efficiency by reducing plan length. The improvement therefore lies in shorter trajectories, not higher task success.

### Questions
In addition to the remarks made for the Weaknesses section, here are some additional questions:

- In the pruning phase, a shortcut is only retained for RL training if random rollouts reach the terminal abstract state s_{\text{term}} in at least K_{\text{rollout}} / N_{\text{rollout}} (which is \approx 5% with the mentioned parameters) of attempts. Given that this requires non-negligible random success (especially for a 3-D environment) between abstract states, how are such reachable abstract-state pairs obtained? Is there any mechanism during the construction of the abstract planning graphs or the definition of abstract states that biases the graph toward physically close or feasible state pairs, making this 5 % success rate attainable? Because, even for the simplest 3-D tasks, having more than a 5% success rate by taking random actions is highly unlikely. 

- My understanding is that Figure 4 shows how, as training progresses, more shortcut policies become reliable and therefore less likely to be excluded from use during inference. In other words, the number of shortcuts that remain usable in the evaluation graph increases as RL training improves their success rates. Is this interpretation correct? The current wording (“increasing the number of training steps leads to more shortcuts being successfully learned and incorporated as graph edges”) is somewhat unclear because the number of learned shortcuts does not change with training.

- “To create an initial state distribution, we do not assume that we can sample directly from sinit; instead, we sample from the states encountered in the abstract planning graphs for the training tasks.” I suppose the graph creation process is deterministic. So, doesn’t this limit the diversity of the initial state distribution? How robust is SLAP to out-of-distribution physical configurations at test time, and could additional randomization during graph construction improve generalization?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an approach to facilitate Task and Motion Planning efficiency (in term of plan length) by learning additional options to get across abstract states using RL. Compared to simply using a set of limited options (e.g., move the block tower one by one), the proposed method enriches the set of options and opens the possibility of reaching goal states with few steps (e.g., push down the tower entirely).

### Strengths
The paper is well written and easy to read, especially Fig 1 provides a good illustration of desired shortcuts and Fig 3 gives a clear overview of the method.

Technically, the proposed method to generalize over objects / object numbers, to the best of my knowledge, provides an interesting and novel way for downstream adaptation.

Additionally, the experiment results strongly support the proposed method, where it outperforms vanilla TAMP and hierachical RL methods.

### Weaknesses
My major question is around the novelty of the proposed method. There are many skill learning methods where people learn skills by specifying a goal state (and optionally a start state) with RL, and then apply such learned skills to downstream planning or HRL. If I understand correctly, the proposed method is very similar to those methods, except that the start and goal states come from the planning graph. Is that right? Any novelty I missed? 

Meanwhile, I wonder what if the learned shortcut has non-zero failure rates, will SLAP still adopts such options? If so, what if the RL learned options fail will lead to an unrecoverable state, will the planning take such possibility into account?

### Questions
When conducting random rollout pruning, what is the action space, atom action or provided options?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles long horizon decision making in robotics where classic task and motion planning relies on hand specified skills and often yields long satisficing plans. The authors propose Shortcut Learning for Abstract Planning, or SLAP, which automatically learns new low level options that act as shortcuts between abstract states in a planning graph induced by the existing skill library. The system builds a two level abstract planning graph, searches for shortest executions at the low level, then augments the graph with learned options trained through model free RL in self contained shortcut MDPs with step penalties and goal termination. A simple but effective pruning strategy screens shortcut candidates using random rollouts before launching PPO training. At test time, SLAP plans with both given skills and learned options, prunes failing edges online, and selects shorter plans via Dijkstra on the ground graph. To generalize beyond the training object set, SLAP projects observations onto relevant objects determined from add and delete atom sets and applies object substitution to reuse learned policies when object identities differ. This preserves the relational inductive bias of TAMP while enabling dynamic behaviors beyond fingertip grasp and place such as slap, wiggle, and wipe.

### Strengths
Framing and practical value
- Clear problem statement that existing TAMP systems rely on hand designed skills which limits efficiency and expressivity. SLAP focuses squarely on reducing execution time without discarding the benefits of abstraction and search.
- Elegant algorithmic design that keeps planning and learning modular. The abstract planning graph yields a well defined search problem, and shortcuts are learned in parallel MDPs with simple goal conditions and dense step penalties.
- Relational generalization via relevant object projection and object substitution is well motivated and effective, letting the same shortcut policy apply across object sets and counts.

Empirical evidence
- Consistent improvements in plan length across four diverse domains with long horizons and sparse rewards. Reductions are large in magnitude, for example about 69 percent in Obstacle Tower and over 50 percent in the PyBullet tasks.
- Clear wins over strong baselines representing three regimes: pure planning, pure RL with PPO and SAC plus HER, and hierarchical RL with access to the same predefined skills.
- Training steps analysis and shortcut discovery counts show monotonic gains as more shortcuts are learned, connecting learning progress to planning improvements.
- Generalization experiments demonstrate robustness to different numbers of obstacles and to additional distractor objects. The learned dynamic behaviors like slap and wipe act on multiple objects and keep plan lengths stable.

### Weaknesses
Assumptions and scope
- Relies on a known transition function and fully observable deterministic settings for graph construction, although variants relax these assumptions. Real systems often face sensing delays, latency, and controller noise which may require tighter integration of failure recovery and uncertainty aware planning.
- Assumes the provided option set enables task completion. When this is not true, completeness can be lost. Appendix results discuss such cases but a stronger treatment of failure detection and fallback would help.

Scalability and compute
- The number of candidate shortcuts scales with the square of the number of abstract states. The pruning heuristic is simple and effective but a learned prioritizer or search over the shortcut space could further reduce training cost. Wall clock budgets and GPU hours per environment would make the compute footprint transparent.
- Planning time modestly increases compared to pure planning in two domains before being outweighed by shorter execution. A more thorough analysis of search heuristics and graph size versus latency would help practitioners choose configurations.

### Questions
- How many shortcut policies were ultimately trained and kept per environment during the runs reported in Table 1, and what was the total wall clock training time per environment including pruning and PPO updates?
- What termination condition is used for each learned option at test time beyond the abstract state check and step limit. Do you implement safety guards such as maximum end effector velocity or minimum clearance during dynamic actions like a slap or wipe?
- You evaluate separate PPO policies per shortcut pair. How do results compare to a single goal conditioned policy trained across shortcut goals when both are tuned equally, and how does data efficiency change as the number of abstract states grows?
- The object substitution criterion checks add and delete inclusions. How is the mapping computed in practice when multiple candidates exist, and what is its computational cost during planning on the larger PyBullet graphs?
- For hierarchical RL, did you try curriculum learning, intrinsic motivation, or hindsight relabeling at the high level to mitigate sparse rewards before concluding failure on the three PyBullet domains?

### Soundness
3

### Presentation
3

### Contribution
3
