# GraphPlan: Graph-enhanced Planning via Thinking LLMs for Embodied Agents

- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Embodied agents that follow instructions to complete complex tasks in visual environments have attracted increasing attention. Large Language Models (LLMs) based planners, notwithstanding the progress achieved, still suffer from three main limitations: (i) a lack of physical grounding, often resulting in hallucinatory plans; (ii) poor generalization to unseen long-horizon tasks; and (iii) an absence of environmental awareness in the open-loop planning process. To address these issues, we propose GraphPlan, a novel framework that integrates a task graph to provide structured knowledge for robust planning and a scene graph to maintain environmental memory for event-driven replanning. Specifically, the task graph guides the LLM's reasoning through contextual prompting and iterative refinement, effectively mitigating planning hallucinations. Furthermore, within the GRPO framework, the task graph offers delicate reward design to train LLMs' reasoning, enhancing long-horizon planning capabilities and improving generalization. Finally, the memory constructed by a dynamic scene graph empowers an event-driven replanning module, enabling the agent to foster environment awareness and correct instruction misalignment within a closed-loop planning process. On the standard benchmark ALFRED, GraphPlan achieves state-of-the-art performance on the official leaderboard. Moreover, its high-level planner outperforms a series of leading API-based LLMs on both the validation set and unseen long-horizon tasks. Additional experiments reveal the promising potential of our graph-enhanced framework in few-shot or zero-shot learning scenarios, and its generalization to novel tasks beyond the benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GraphPlan, a graph-enhanced framework for embodied agents that integrates structured knowledge into LLM-based planning. It builds a task graph to constrain and verify language-generated plans, a scene graph to represent updating object relations, and a memory-aware low-level policy with replanning capability. It combines these components with an event-driven replanning mechanism and a reinforcement learning objective that rewards graph-consistent actions. The method is validated in the ALFRED benchmark and outperforms previous state-of-the-art methods by notable margins.

### Strengths
- The paper tackles challenging and important issues for long-horizon planning.
- The proposed method outperforms previous state-of-the-art methods with large margins.
- Incorporating subtask structure in graph forms is straightforward sensible.

### Weaknesses
- For task graph construction, the authors use four types of meta classes, but their motivation is not well described. Why should be they? In addition, the authors use subtasks from a specific benchmark. Can they generalize to novel tasks?
- For task graph construction and its verification, the task graph is constructed from expert trajectories. How sensitive is the quality of the task graph to the number and diversity of the training samples? Quantitative analyses should be conducted. In addition, can the proposed method be used in case of tasks without training demos?
- For memory-aware low-level action policy, logging waypoints and object masks has already been explored (e.g., Kim et al, 2023). What are the core differences from the prior work? And, why are they significant?
- The proposed event-driven replanning is highly motivated by several failure modes in a specific benchmark, raising a concern of their generalizability.
- Replanning has been actively explored [1,2], but little to no discussion about them is provided. What are the main difference with them?
  - Huang et al., "Inner monologue: Embodied reasoning through planning with language models," CoRL, 2022.
  - Kim et al, "Pre-emptive Action Revision by Environmental Feedback for Embodied Instruction Following Agents," CoRL, 2024.
- It is unclear if the comparison with prior work in Table 1 is fair. For example, how many training samples are used compared to prior work? Some prior work (Song et al., 2023, Kim et al, 2025) uses only a small portion of the training samples (~100 samples). In addition, is the comparison done under the same LLMs?
- The evaluation is conducted in a single benchmark, raising a generalizability concern. Can this approach be applied to other task setups?

### Questions
- In Sec. 3.3.2, how does the agent know if it fails at a task? Is it predicted by LLMs or measured by some predefined rules?
- Due to the discrete nature of graphs, it might be hard to apply the proposed method to tasks with continuous state and action spaces. Can the proposed method be used for such tasks?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed GraphPlan, a framework that incorporates a Task Graph and a Scene Graph for embodied agent task completion in the ALFRED environment. The task graph is a detailed representation of task-specific related information, including objects, subtask relationships, attributes, affordance, etc. It aims to help high level planning, generate reward function for action executions, and provide task verification for reasoning. The method also includes a memory for task relevant object states and locations through waypoints and segmentations. The Scene graph is generated on the fly to store key objects (extracted by LLMs), view points of the objects, and the relationship among objects. It aims to help replanning when a low level action error or subtask planning error is encountered. The paper benchmarked the performance in comparison against multiple methods and demonstrated improvement on task success rate and goal condition success rate. The paper also conducted thorough ablation studies to showcase the importance of each components in the proposed framework. Further more, the paper collected a new datasets of 1396 samples to evaluate long-horizon high level planning capabilities in comparison against 3 closed-source models and varied prompting strategies. The proposed method demonstrated outstanding performance especially among long horizon tasks.

### Strengths
- The paper is well structured and well written.
- The paper proposed a method for embodied agent planning and action prediction, leveraging a proposed task graph and a scene graph
- The paper incorporated a replanning stage to improve long horizon task performance in case of errors
- The paper conducted thorough experiments among baselines, evaluating the proposed method through task success rate, goal condition success rate, and numerous ablation studies. 
- The paper also collected a dataset specifically aiming at evaluating long horizon tasks, and conducted experiments against closed-source LLMs for high level planning

### Weaknesses
1. The paper is careful to highlight the that "the subtask nodes can be expended as the low-level policy evolves" and that "the proposed task graph-based approach can generalize to planning tasks in other domains". While they are indeed possible, the main question is really 'How useful is the method outside the ALFRED/simulated environment? In cases such as general home robot, takeout delivery robot, or disaster rescue robot, how feasible is it to manually generate the task-specific graphs, annotate all the key subtasks, and exhaust all possible/potential states/objects/attributes/conditions/relationships/subtasks?'
2. Section 3.2: it was a bit unclear how exactly low-level action policies are designed & trained when incorporating a memory of object states a locations?  
3. While evaluating the performance, the main metrics are Task Success Rate (SR) and Goal-Conditioned Success Rate (GC). Is task execution efficiency (e.g. #actions/#steps it took to reach a goal) a relevant metrics to consider perhaps?

### Questions
a. It is great that the scene graphs can be generated automatically on the fly. How truthful are the generated scene graphs? 
b. Figure 4: is there any reason why most of the models perform better in the valid unseen dataset compared to the valid seen dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces GraphPlan, a graph-enhanced planning framework for embodied agents that improves long-horizon reasoning and task execution in complex environments. It addresses key weaknesses of LLM planners such as hallucinated actions and poor generalization by integrating graph representations for reasoning. Through graph-guided prompting, verification, and RL, the system maintains grounded, feasible plans. Evaluated on the ALFRED benchmark, GraphPlan achieves SOTA performance, outperforming prior methods by about 6% in unseen success rate, and demonstrates strong generalization to long-horizon tasks.

### Strengths
- Symbolic structures for guiding LLM-based planning is novel and creative. The method is motivated intuitively. 
- The dual graph method (task and scene decoupled) for provides a principled way to make LLMs operate for planning. 
- The frameworks is well validated through comprehensive experiments, including ablations that isolate the contribution of each module.

### Weaknesses
- The construction of task graphs appears handcrafted/domain-dependent, raising concerns about how easily GraphPlan generalizes to new environments.
- In larger or more cluttered environments, these graphs could contain hundreds of nodes and relations. The paper assumes the LLM can interpret and reason over these graph descriptions accurately, but provides no analysis of performance degradation or prompt efficiency as graph complexity grows.
- Do the authors empirically demonstrate how GraphPlan’s performance scales with increasing task horizon length? While the benchmark shows overall results, there’s no detailed analysis of degradation trends of whether other methods fail progressively with more subtasks (longer horizons) while GraphPlan remains stable.

### Questions
- Could the authors provide additional evaluation of "reasoning fidelity". For example, how closely intermediate subgoals follow the intended logic of the instruction, or how often the model avoids invalid or redundant subtasks?
- How robust is GraphPlan when the scene graph contains missing or incorrect relations? Could the LLM detect and repair such inconsistencies through reasoning?
- While the long-horizon benchmark shows strong overall results, the paper doesn’t characterize computational or reasoning costs. How does inference time and #LLM calls scale with task length?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers use of LLM’s for embodied agent planning, and proposes GraphPlan that uses a task graph and scene graph.  The task graph is developed a priori and characterizes the possible sequences of actions (sub-tasks).  This can be used in an LLM planning prompt to ensure the plan adheres to the graph.  A scene graph is constructed that incorporates the entire environment, with all relevant objects and relationships discretely encoded. The scene graph will be updated when changes occur.  The method is used on the ALFRED benchmark and some comparisons are made.

### Strengths
The methodology discretizes both the action space and the scene in graphs, and this provides a framework for LLM-based plan generation, replanning, and adhering to feasible plans. The planning method might be useful for highly characterized environments with a fixed set of simple actions and set of objects with relations encoded.  

GRPO is used to generate a policy and rewards are linked to the graph structures. This guides the GRPO within the constraints specified for the tasks and scene.

The method is easy to understand and intuitively clear.

### Weaknesses
The method discretizes both action and scene, including objects and relationships.  This greatly simplifies the overall processing, which is logical but also highly constrained.  Given the task graph a priori, it isn't obvious that an LLM based planner is even needed, and why a graph searching type method can't be applied directly. 

Scalability is unlikely.  The scene graph will grow with the scene size and will be a serious bottleneck, even in a highly controlled simple environment.  Adding new task and objects apparently requires a new policy training phase. 

The overall novelty isn't obvious.  Learning scene graphs with objects has already been developed.  The verification of a finite LLM-plan has also been considered, e.g., by mapping to a finite automata type model. 

The underlying task graph assumes each node can be reached (e.g., grasping), and this doesn't account for perception errors, incomplete sub-tasks, or other forms of real life conditions.

### Questions
How does the task graph approach compare with other LLM-based encoders that choose among a set of actions?

What is the novelty with respect to the scene graph?

### Soundness
2

### Presentation
2

### Contribution
2
