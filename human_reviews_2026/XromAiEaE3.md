# GraphMind: LLMs as Dynamic Knowledge Builders for Sequential Decision-Making

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
While the reasoning capabilities of large language models (LLMs) have advanced considerably due to their extensive internal knowledge, efficiently internalizing and leveraging new information in dynamic environments remains challenging. This limitation is particularly pronounced in partially observable environments, which require agents to manage long-term memory and perform effective exploration under incomplete information. To address this, we propose an LLM agent architecture that integrates a knowledge graph as a graph-based memory module to facilitate high-level action planning. The agent incrementally constructs the knowledge graph through environmental interactions and retrieves relevant information to formulate efficient plans. We evaluate our approach in complex navigating environments specifically designed to present long-horizon and partially observable challenges. Experimental results demonstrate that employing a knowledge graph as an external memory significantly enhances the success rate and efficiency of the LLM’s planning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
"GraphMind: LLMs as Dynamic Knowledge Builders for Sequential Decision-Making" proposes an LLM agent that constructs and uses a dynamic knowledge graph (KG) as an external memory for planning in partially observable BabyAI environments. The agent alternates between graph construction (building nodes/edges from observations) and structured planning via a domain-specific language (DSL). Experiments on small BabyAI gridworlds (maps with 2×2 and 3×3 rooms) with tasks like OpenDoor and PutNextTo suggest that the graph-based memory improves task success rates over a “stacked memory” baseline and a no-memory setup.

### Strengths
* The framework is well-engineered, with detailed modular prompts and clear visualization of how the knowledge graph evolves.
* The combination of structured memory and DSL-based planning is conceptually clean and helps connect symbolic reasoning with embodied action.
 * The experiments, though small-scale, are thorough within the limited setting, including GED-based analysis of graph accuracy and qualitative visualizations.

### Weaknesses
* **Novelty**: The idea of dynamically constructing and exploiting a knowledge graph as an LLM memory in partially observable settings is not new. Prior work like AriGraph explored very similar designs: LLMs building and querying graph-structured memory for decision-making in POMDP. The paper’s core claim of being the “first” dynamic KG-based planner is overstated.
 * **Evaluation scope**: The empirical validation is extremely limited—only two BabyAI missions (OpenDoor, PutNextTo) in 2×2 and 3×3 grids, with about 20 layouts and three trials each. There is no testing on larger, more complex environments, distractors, or longer object-interaction chains. Claims of generality to “real-world” or “long-horizon” reasoning are unsubstantiated.
* **Baselines**: Comparisons are weak—only a simple “stacked memory” and a no-memory version. Missing comparisons with prior KG/RAG-based or agentic-memory frameworks makes it unclear how much progress is achieved.
* **Practicality and efficiency**: The paper omits any discussion of runtime, compute, or token costs, which are crucial for assessing the feasibility of maintaining dynamic KGs with LLMs.

### Questions
Please address the weaknesses mentioned in the previous section.

1. Could you clarify how your approach fundamentally differs from prior work such as AriGraph or other KG-based memory frameworks for LLM-driven agents?

2. Could you elaborate on why experiments are limited to small BabyAI environments (2×2, 3×3) and discuss how you expect the method to scale to larger or more complex tasks?

3. Please provide quantitative data on computational and token-level costs (e.g., inference time per step, prompt length, graph update overhead) to assess the scalability and practicality of maintaining dynamic KGs in your setup

### Soundness
2

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
The authors present an approach to us a knowledge graph as a memory for reasoning with an LLM in tasks that have longer range dependencies.

### Strengths
* I like the setup of the task, which is such that the agent must have a long-range memory in order to find a good solution. It remains a bit unclear how hard the actual tasks are because of the generation and filtering approach. 
* The overall results show that the method results in an overall better performance.
* The method is rather intuitive

### Weaknesses
* I think the opening premise of the abstract is a bit strange: "the reasoning capabilities of large language models (LLMs) have advanced considerably due to their extensive internal knowledge". This is actually not really a proven thing. The reasoning capabilities seem to stem from analogies with knowledge, but it is, despite anecdotal evidence, not formally proven.

* This work might suffer from a confirmation bias. It is not really possible to say that an LLM could in no way solve these problems; it might just be that we have not yet found the right way to prompt it. A theoretical proof of such a limitation would be a much stronger contribution. In this context it is important that the paper adds computational tools, specifically BFS to the LLMs capabilities. It would be great if one can proof that the LLM cannot perform a BFS of such depth with its context window; but I think it could. It would also be useful to get statistics on how often these tools are called, and how deep the BFS needs to go. As a sub-comment, I am not sure why the details on the tool calls are hidden in the appendix, while they are actually pretty essential.

### Questions
* Is it possible for your agent to not store information in the knowledge graph and instead use the context window? 
* Do you have a way to measure the accuracy of the content in the knowledge graph compared to what really is in the environment? 
* In some places, you call your environment dynamic, but I don't understand what is dynamic in your environment, can you elaborate?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GraphMind, an LLM-based agent architecture for sequential decision-making in partially observable environments. It integrates a knowledge graph (KG) as dynamic memory, incrementally built from object interactions to support long-horizon planning. The LLM retrieves relevant KG subgraphs to generate high-level actions, refined into low-level steps. Evaluations focus on custom grid-world navigation tasks (e.g., object collection with partial observability), claiming superior success rates and efficiency over baselines like ReAct and naive LLM planners, especially in exploration-heavy scenarios.

### Strengths
The work addresses a pertinent challenge in LLM agents: managing long-term memory, where naive prompting fails due to context limits. The KG as structured memory is a reasonable extension, enabling interpretable retrieval (e.g., via subgraph queries) and reducing hallucination risks. Experiments show intuitive visualizations and quantitative gains, with ablations on memory types providing some insight.

### Weaknesses
Despite its aims, GraphMind lacks substantial novelty, largely recombining existing ideas: LLM prompting for planning, KG for memory augmentation, and rule-based updates in grid worlds. KG construction is simplistic (e.g., object-centric heuristics), without learned mechanisms or handling of noisy perceptions, limiting generalization beyond toys. 

Experiments are severely constrained: custom, low-dimensional grid tasks ignore standard benchmarks (e.g., MiniGrid, BabyAI), and baselines are weak—missing SOTA like Voyager. 

Claims of "dynamic knowledge builders" are vague without ablation on LLM sensitivity (e.g., GPT-4 vs. open models) or scaling to larger graphs (potential explosion). 

Robustness tests are contrived (e.g., fixed obstacles), overlooking real shifts like dynamics changes or multi-agent interactions. Overall, the method feels incremental and underexplored, with no theoretical analysis of efficiency or failure modes.

While LLM+KG integration is promising for memory in agents, GraphMind offers no groundbreaking advances. The narrow, toy-like evaluations fail to demonstrate broad impact, and overstated claims undermine credibility.

### Questions
In the end of Section 3, authors mentioned "This cyclical structure ensures that planning remains adaptive, resilient to execution errors, and
robust under partial observability." Regarding the partial observability, are there any specific techniques leveraged to tackle the challenge, or just rely on LLMs to provide additional information?

### Soundness
3

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
3

### Summary
This work proposes GraphMind, a framework that builds knowledge graph for LLMs to provide memory and help sequential decision making especially in BabyAI tasks.

### Strengths
The paper is clearly written and well structured. The use of a graph-based memory and DSL makes the system interpretable. Ablation studies (no memory, stacked memory) are clean and support the main claim. Graph-edit distance as a proxy for memory accuracy is intuitive.

### Weaknesses
I’m concerned about the generalization and scalability of the proposed framework. The experiments are confined to the BabyAI environment with only a few predefined layouts, which are relatively simple and small-scale. There’s no evaluation in more challenging or diverse settings, such as larger or irregular maze environments, tasks with richer object interactions, or real-world scenarios. So it’s unclear whether the approach would maintain its effectiveness beyond this narrow domain.

### Questions
Is the knowledge graph truly necessary? Couldn’t we achieve similar results by simply giving the LLM access to a global map of the environment or by providing prior trajectories and observations in plain text? Given the current strength of modern LLMs, such structured graph representations may not be essential for this relatively simple task.

### Soundness
2

### Presentation
3

### Contribution
2
