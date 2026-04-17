# CoSTA*: Cost-Sensitive Toolpath Agent for Multi-turn Image Editing

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Text-to-image models like Stable Diffusion and DALLE-3 still struggle with complex multi-turn image editing. We study how to break down such a task into a sequence of subtasks and address them by an agentic workflow (path) of AI tool use with minimum costs. 
Conventional search algorithms require expensive exploration to find tool paths. While large language models (LLMs) possess prior knowledge of subtask planning, their estimation of the quality and cost of tools is usually inaccurate to determine which to apply in each subtask. $\textit{Can we combine the strengths of both LLMs and graph search to find cost-efficient tool paths?}$ We propose a three-stage approach``CoSTA*'' that leverages LLMs to create a subtask tree that prunes a graph of AI tools for the given task, and then conducts A* search on the small subgraph to find a tool path. To better balance the total cost and quality, CoSTA* combines both metrics of each tool on every subtask to guide the A* search. Each subtask's output is evaluated by a vision-language model (VLM), where a failure will trigger an update of the tool's cost and quality on that subtask. Hence, the A* search can recover from failures quickly to explore other paths. Moreover, CoSTA* can automatically switch between modalities across subtasks for a better cost-quality trade-off. We build a novel benchmark of challenging multi-turn image editing, on which CoSTA* outperforms state-of-the-art image-editing models or agents in both cost and quality, and performs versatile trade-offs upon user preference. Our dataset and a hosted demo can be found at https://storage.googleapis.com/costa-frontend/index.html.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a method called CoSTA* to improve multi-step image editing. The main idea is to break down an editing task into a sequence of smaller subtasks and find the most efficient path of AI tools to complete them. The paper notes that LLMs are good at planning subtasks but are not accurate at estimating the cost or quality of the tools needed for each step. CoSTA* combines the planning strength of LLMs with a search algorithm. First, an LLM creates a plan, which helps to narrow down the number of possible tools. Then, an A* search algorithm finds the best path through these tools, balancing both cost and quality. VLMs are used to check the output. If a tool fails, the system updates its knowledge about that tool's cost and quality, allowing the search to find an alternative path. To test their method, the authors created a new benchmark of image editing tasks and showed that CoSTA* performs better in both cost and quality.

### Strengths
1. The paper is easy to understand.
2. The motivation is clear.
3. The engineering efforts in designing all the editing tools are appreciated and may be helpful to the community.

### Weaknesses
My major concerns are two-fold, focusing on its motivation and technical novelty.

1. The primary motivation of optimizing for cost, defined as inference time, seems insufficiently justified for the domain of multi-turn image editing. This type of task is often performed asynchronously; users typically submit a job and retrieve the results later, making real-time performance a secondary concern. Furthermore, this focus on model inference time overlooks two key factors: 1) Editing tools are rapidly evolving, meaning any cost calculation based on current inference times is ephemeral. 2) Real-world applications often rely on APIs, which introduce network latency and queuing delays that are unrelated to model inference and can dominate the total wait time.

2. The paper presents limited technical novelty, as its core components are well-established concepts within the agent and planning communities. Decomposing tasks into trees or graphs of dependencies is a well-studied paradigm, seen in prior work such as VISPROG, ViperGPT, HuggingGPT, Tree of Thoughts, and Graph of Thoughts. Using VLMs to evaluate intermediate steps and guide the subsequent action is a common strategy in building autonomous agentic workflows. The necessity of an A* search is questionable. A simpler, rough plan executed with a ReAct-style pattern (e.g., HuggingGPT + ReAct) is likely sufficient. The success of any plan ultimately hinges on the quality of the underlying tools. If a tool cannot produce the desired result, no amount of sophisticated planning or searching can salvage the outcome. The true bottleneck is often tool capability, not the planning algorithm.

### Questions
While I am not an expert in designing image editing benchmarks, I have two comments:

1. A potential concern is that the benchmark may be implicitly tailored to the specific editing tools and patterns developed by the authors. This could create an unfair comparison against general-purpose methods that have not been specifically optimized for this task suite.

2. The diversity and sample number might not be sufficient to draw robust and generalizable conclusions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CoSTA*, a hierarchical agent framework for complex, multi-turn image editing tasks that combines LLM-based high-level planning with cost-sensitive A* search for low-level tool selection. The system addresses the challenge of efficiently orchestrating sequences of editing operations (toolpaths) while balancing quality and computational cost. CoSTA* operates in three stages: (1) an LLM decomposes the editing task into a subtask tree, (2) this tree prunes a Tool Dependency Graph (TDG) to create a focused subgraph, and (3) A* search finds the optimal toolpath using a cost function f(x) = g(x) + h(x) that combines actual execution cost with heuristic estimates, controlled by a tradeoff parameter α. The system incorporates real-time VLM-based quality checking with adaptive retry mechanisms. The authors construct a benchmark of 121 curated image-editing tasks with 1-8 subtasks (550 total edits) covering both image-only and text-in-image operations. Experiments demonstrate CoSTA* achieves 94% accuracy overall, substantially outperforming existing agents (GenArtist: 73%, VisProg: 62%) and closed-source models (Gemini 2.0: 81%, GPT-4o: 78%), while offering explicit control over cost-quality tradeoffs through the α parameter.

### Strengths
1、 Multi-turn image editing with cost control is a genuine challenge for current systems. The motivation for combining symbolic search with neural planning is compelling and addresses real limitations of pure LLM-based agents.

2、The hierarchical combination of LLM subtask decomposition with A* search on a pruned tool graph is innovative. This design elegantly leverages LLM strengths (high-level reasoning) while mitigating weaknesses (poor cost/quality estimation) through structured search.

3、The analysis showing LLM pruning reduces search space from 100,000+ nodes to 15-20 nodes demonstrates practical scalability of the approach.

### Weaknesses
1、While the integration is novel, the core techniques—LLM task decomposition, A* search, VLM evaluation—are standard. The A* formulation itself is relatively straightforward, and the heuristic h(x) is computed via simple recursive propagation from empirical benchmarks rather than learned

2、All experiments use a single A100 GPU. How does the approach scale to larger tool libraries (50+ tools), deeper task hierarchies (10+ subtasks), or real-time editing scenarios? The paper provides no runtime analysis or computational complexity discussion beyond node count reduction.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel approach, CoSTA∗, for multi-turn image editing, combining large language models (LLMs) and A∗ search to efficiently plan and execute tool paths. It targets improving the cost and quality trade-offs in complex image editing tasks, specifically addressing multi-step workflows. The authors introduce an agentic mechanism that decomposes a complex task into smaller subtasks, which are then optimized using a hierarchical planning system. CoSTA∗ outperforms existing state-of-the-art image editing models in terms of both cost efficiency and output quality, offering a dynamic trade-off mechanism that adjusts based on user-defined preferences.

### Strengths
1. The integration of LLMs for task decomposition with A∗ search for tool path optimization is a novel and promising approach, addressing key challenges in multi-turn image editing.

2. The dynamic tuning of cost and quality through the use of a tunable coefficient (α) provides users with flexibility and optimization, offering Pareto-optimal solutions for both performance metrics.

3. The paper introduces a new, challenging benchmark for multi-turn image editing tasks, which contributes significantly to the field by allowing more rigorous evaluation of editing agents.

### Weaknesses
1. The system relies on prior knowledge from benchmark datasets, which may not always be available or applicable for all types of tasks.

2. Although CoSTA∗ excels in multimodal tasks, it may still face challenges when adapting to new, previously unseen tools or modalities. The adaptability in real-world usage needs to be tested more thoroughly outside of controlled benchmarks.

3. The method shows strong performance in the experimental setting, but its scalability to larger, more complex workflows could be an area for further investigation, especially in real-world applications.

### Questions
see my weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces CoSTA*, a cost-sensitive toolpath agent designed for multi-turn image editing. The framework integrates a large language model (LLM) for high-level subtask planning with an A* search algorithm for low-level optimization. It constructs a Tool Dependency Graph (TDG) and a Model Description Table (MDT) to define tool relations and supported subtasks, and incorporates a cost-quality tradeoff via tunable coefficient α. The authors also propose a new benchmark for complex multi-turn editing and claim superior performance over recent systems such as GenArtist, CLOVA, MagicBrush, and InstructPix2Pix.

### Strengths
- The hierarchical design combining LLM-based planning and A*-based search is conceptually elegant and well-motivated. The inclusion of structured metadata (MDT and TDG) demonstrates thoughtful system engineering.
- The paper introduces a new benchmark for multi-turn multimodal editing, which may be valuable to the community if released publicly with standardized evaluation protocols.
- The idea of incorporating cost sensitivity into multi-turn image editing is interesting and practically relevant, especially for agentic workflows where computational cost varies significantly across tools.

### Weaknesses
- While the framework description is detailed, the technical novelty appears incremental. The “cost-sensitive A*” mainly combines standard heuristic search with ad-hoc cost-quality weighting (α). The mathematical formulation (Eq. 3–4) is not rigorously justified, and the approach reads more like system integration than algorithmic innovation.
- The proposed benchmark is not clearly standardized, and human evaluation protocols are vaguely defined. Metrics like “accuracy” and “quality” are subjective and loosely defined; CLIP-based evaluation is known to be unreliable but is still used in parts of the analysis.
- The experiments show marginal improvements over strong baselines on public datasets suggesting that gains primarily come from tool diversity rather than planning effectiveness. Ablations confirm that removing multimodality or tool dependency graphs causes large drops, implying the method’s advantage lies in data engineering rather than algorithmic design.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
