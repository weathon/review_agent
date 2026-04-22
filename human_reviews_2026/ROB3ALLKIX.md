# APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability.
We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations.
We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces APEX, a framework that enhances LLMs with physics-based reasoning capabilities for real-time task planning in dynamic environments. APEX constructs structured interaction graphs from environmental snapshots, uses a Graphormer-based attention mechanism to identify salient dynamic interactions, and performs forward simulations via a MuJoCo to predict action outcomes. The LLM then selects optimal actions based on these simulations. The authors evaluate APEX on three benchmarks: a Physics Reasoning Benchmark, Tetris for long-horizon planning, and Dynamic Obstacle Avoidance. Results show that APEX significantly outperforms baseline LLMs and VLM-based models, demonstrating improved physical reasoning, foresight, and real-time adaptability.

### Strengths
1. Introduces a novel graph-simulation-loop architecture to help LLMs in physical reasoning.

2. Evaluation across three diverse benchmarks with strong baselines and ablations.

3. Well-organized and accessible, with clear explanations of both high-level ideas and implementation details.

### Weaknesses
1. The framework's applicability is currently limited to motion-related problems, particularly collision prediction. This narrow scope and its reliance on a specific simulator make it brittle; adapting it to new tasks or physical phenomena likely requires significant re-implementation.
2. To demonstrate generalizability, future work should include experiments in more complex settings, such as those with cluttered scenes or deformable objects.
3. Although APEX's modularity is a strength, the paper provides limited analysis of the design choices involved, such as the trade-offs between different graph encoders or simulators. Several architectural questions remain: for instance, are the graph and simulation modules strictly decoupled, or could they be jointly fine-tuned with the LLM? Furthermore, have the authors considered integrating learned world models (e.g., neural physics engines) as potential replacements for MuJoCo?
4. The real-world experiments demonstrate the system's capability under controlled conditions. However, it is unclear how sensitive APEX is to practical challenges like perception noise or delays in the dynamic graph construction process.

### Questions
See weaknesses. I am willing to discuss and improve the paper.

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
This paper proposes APEX, a framework intended to augment LLMs with explicit physics-based foresight for task planning. It introduces a Perception–Graph–Language–Physics–Action pipeline that integrates a graph attention module and a physics simulator to provide physical rollouts as feedback to an LLM during decision making. The authors evaluate APEX on three benchmark domains—synthetic physics QA, Tetris planning, and dynamic obstacle avoidance—claiming substantial improvements over vanilla GPT-4o and VLM baselines.

### Strengths
1. Interesting high-level motivation. Bridging symbolic reasoning in LLMs with physically grounded modeling is an important and timely goal.
2. Attempt to unify physics reasoning and LLM planning. The modular architecture (graph → simulator → LLM → action) provides a readable system outline.

### Weaknesses
1. Conceptual novelty is limited. The core idea—using a physics engine to simulate candidate actions and feeding the results back to an LLM—is conceptually straightforward and has appeared in prior “simulation-in-the-loop” or “world-model prompting” works (e.g., Mind’s Eye, PiLoT, PhysVLM). The proposed Perception–Graph–Language–Physics–Action paradigm is mostly a re-labeling of existing perception-simulation-planning loops in robotics; there is no theoretical or algorithmic advance beyond modular composition.
2. Questionable experimental design and fairness. Benchmarks are non-standard. The “Physics Reasoning Benchmark,” “Tetris,” and “Dynamic Obstacle Avoidance” are all custom setups with unclear data availability or reproducibility.
3. Paper is not well written. The main text could not fully present the results and the analysis. I suggest putting some result figures from appendix to the main text. Table format should be unified and the figures need further improvement for clarity.

### Questions
See above.

### Soundness
1

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
This paper introduces APEX, a framework for enhancing LLM’s physical reasoning capabilities for real-time task planning by integrating with simulations of physical engines. Given some prompt, the system is able to generate qualitative and quantitative predictions of the scene in text. Specifically, given two frames, APEX first creates a scene graph that captures object relations and interactions, then passes the current relational state to a physics engine to simulate possible futures under different actions. The resulting information is incorporated into a unified prompt for the LLM, which then selects the optimal action.

The paper introduces three benchmarks and evaluates APEX-enhanced LLMs with vanilla LLMs, and shows that APEX-enhanced LLMs outperforms vanilla LLMs on physics reasoning, Tetris, and obstacle avoidance.

### Strengths
- The paper attempts to address the physical reasoning limitations of LLMs by making use of physics engines.
- The paper provides some concrete examples of tasks and model outputs in the appendix.

### Weaknesses
- Overall, the introduced pipeline is very limiting. The framework makes many assumptions, but none are explicitly described in the paper. For example, the scene graph formulation assumes the scene contains mostly distinctive and rigid objects. Since the predictions are simulated with a physical engine, the method also assumes those objects are compatible with the simulator. What constraints are imposed on the acceptable objects in the scene? Moreover, the decision-making procedure requires a finite, enumerable set of actions, as the algorithm simulates each action one by one. This is largely infeasible in real decision-making tasks.
- The paper lacks baselines. It motivates the work by describing limitations of latent world models, RL, etc., but only compares APEX-enhanced LLMs with vanilla LLMs. What about latent world models, RL, or other methods as baselines? If they are not compatible, why motivate the paper based on their limitations?
- There is no description of how the physical engine is designed or used. How is the simulated scene created based on the current relation state? The paper only mentions MuJoCo in Figure 2, but what are the assumptions on the scene to make it simulatable by MuJoCo?
- The proposed pipeline appears to bottleneck information through text. If the system already has access to input frames, created scene graphs, and a physics engine, why is text used as the medium for reasoning and decision-making? Wouldn’t this cause loss of critical information, and isn’t it the case that not all scenes can be described accurately through text?
- The paper’s writing could be improved. It presents irrelevant information while missing some critical details about the method. E.g. Some works in the related work section appear misplaced,  for instance, it is unclear why RL is discussed, or why several non-RL papers are included in Section 2.3. Similarly, it is not clear why R3M is categorized under world models in Section 2.2. 
- Some of the task figures should be moved to the main body. Currently, it is difficult to ground the evaluations, even though the paper provides some visuals in the appendix.

### Questions
- What are the assumptions on the scene, dynamics, and action space?
- The method only uses $G_t$ and $G_{t + \Delta t}$ as context. How does this capture second-order dynamics information such as acceleration? If such information must be provided through text, this imposes an additional requirement on the system that these values be known, yet information such as object velocity or acceleration is typically unavailable in real-world decision-making.
- Why use scene graphs? Although Appendix 6.7.2 provides some motivation in terms of interaction and temporal saliency filtering, these considerations mainly arise from converting everything into text. Why is this representation preferable to, for example, a latent world model, where such filtering would not be necessary?
- In the Tetris experiment, only APEX and vanilla LLMs are evaluated. What would the oracle performance be, e.g. from a classical planning algorithm or human players?
- While the paper provides some prompt and task examples in the appendix, what would a complete input–output pipeline look like for a specific task (e.g., Tetris)? What are the exact inputs and outputs of the Graphormer encoder and the physics engine? How are the entities represented and instantiated within the physics engine?

### Soundness
1

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
2

### Summary
The paper proposes APEX, a plug-and-play loop that (i) constructs a relational scene graph from two snapshots, (ii) applies a “difference-graph” attention module (a Graphormer variant) to identify salient interactions, (iii) enumerates candidate actions, (iv) performs forward simulation rollouts with a physics engine for each action, and (v) asks an LLM to select an action using textual rollout summaries. Experiments cover synthetic physics QA, a custom short-horizon Tetris setup, a dynamic obstacle-avoidance toy task, PHYRE, and a small “real-world” vignette.

### Strengths
- Pragmatic integration of a physics engine into an LLM loop; the overall system is easy to understand and replicate in spirit.

- Clear problem motivation: teaching LLM agents to rely on external physics tools rather than internalizing fragile physical heuristics is a sensible direction.

- Breadth of tasks (synthetic QA, games/toy control, PHYRE) demonstrates that the loop can be wired up across settings.

### Weaknesses
- limited methodological novelty
- single-step lookahead: the algorithmic core is a 1-step brute-force evaluation of enumerated actions; claims and tables about multi-step/higher-horizon complexity are not matched by controlled, implemented evidence.
- Tetris lacks standard hand-coded/Tetris-AI baselines (e.g., height/holes/bumpiness heuristics, MCTS with lookahead).
- Obstacle avoidance omits classic MPC/DWA/A*/RRT with the same simulator and budget.
- PHYRE relies on 10k uniformly random actions, effectively reducing to random search rather than demonstrating LLM-guided planning

### Questions
Can you precisely define (\Delta G) for heterogeneous edge types and features. How are ((G_t, G_{t+\Delta t})) fused in attention (encodings, positional terms, pairwise features)?

### Soundness
2

### Presentation
2

### Contribution
2
