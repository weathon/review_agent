# NaviAgent: Bilevel Planning on Tool Navigation Graph for Large-Scale Orchestration

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Large language models (LLMs) have recently demonstrated the ability to act as function call agents by invoking external tools, enabling them to solve tasks beyond their static knowledge. However, existing agents typically call tools step by step at a time without a global view of task structure. As tools depend on each other, this leads to error accumulation and limited scalability, particularly when scaling to thousands of tools. To address these limitations, we propose NaviAgent, a novel bilevel architecture that decouples task planning from tool execution through graph-based modeling of the tool ecosystem. At the task-planning level, the LLM-based agent decides whether to respond directly, clarify user intent, invoke a toolchain, or execute tool outputs, ensuring broad coverage of interaction scenarios independent of inter-tool complexity. At the execution level, a continuously evolving Tool World Navigation Model (TWNM) encodes structural and behavioral relations among tools, guiding the agent to generate scalable and robust invocation sequences. By incorporating feedback from real tool interactions, NaviAgent supports closed‑loop optimization of planning and execution, moving beyond tool calling toward adaptive navigation of large‑scale tool ecosystems. Experiments show that NaviAgent achieves the best task success rates across models and tasks, and integrating TWMN further boosts performance by up to 17 points on complex tasks, underscoring its key role in toolchain orchestration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents NaviAgent, a bilevel framework for large-scale tool orchestration by LLMs. It decouples task planning (4D decision space: direct response, intent clarification, toolchain retrieval, execution) from tool execution (Tool World Navigation Model, TWNM). TWNM models tool structural/behavioral dependencies via a dynamic graph, enabling adaptive toolchain search. Closed-loop feedback optimizes planning/execution. Experiments on API-Bank/ToolBench show NaviAgent outperforms baselines in TSR, balancing efficiency/robustness .

### Strengths
1. This paper proposes a novel bilevel architecture that decouples task planning from tool execution, enabling NaviAgent to handle thousands of tools without being hindered by inter-tool complexity, thus addressing scalability issues of existing agents .
2. The Tool World Navigation Model (TWNM) dynamically encodes tool structural and behavioral dependencies, supports adaptive toolchain search/evolution, and significantly boosts performance on complex tasks by up to 17 points .
3. It integrates a closed-loop optimization mechanism using real tool interaction feedback to refine planning and execution, enhancing robustness and adaptability to dynamic API ecosystems .
4. For me, the strengths of this paper lie in the following aspects: it can dynamically adjust based on the difficulty level of various problems and the latest status when addressing them; it exhibits strong overall engineering feasibility; and in terms of innovation, the search strategies such as Alpha-Beta pruning can handle and prune some extreme cases, enabling rapid acquisition of effective toolchains.

### Weaknesses
1. My biggest concern is that this paper imposes overly strong constraints on input problems. For example, if we obtain the corresponding tool invocation path through the proposed graph, can we dynamically switch to an alternative path if a problem occurs in the middle of the current path? The paper seems to lack sufficient explanation regarding how to handle such errors.
2. When using Alpha-Beta pruning for search, the evaluation of Alpha and Beta values is crucial. For instance, if I choose a certain edge under a specific decision, how do you update the evaluation value of this decision in the global context? If only factors like tool invocation success rate and relevance are used, the accuracy of relevance evaluation needs to be very high. From this perspective, the paper’s evaluation of heuristic search seems relatively simplistic, and in some cases, it might eliminate effective tool invocation branches.
3. How do you evaluate the dependency between two tools? The paper mentions using weights for evaluation—are these weights based solely on the historical information of the tools observed so far? If the invocation relationship between two tools changes significantly, will the tools overly rely on historical data?
4. Does the proposed framework rely too heavily on the evaluation of tool invocation success rates? If a tool has a certain invocation success rate but delivers excellent results when it works, we might not should sacrifice its usage frequency. Alternatively, for your paper, is tool invocation speed more important compared to the overall reasoning performance?
5. Real-world requirements are highly diverse. For a new requirement, can this framework demonstrate better tool planning capabilities compared to traditional methods like ReACT and Tool-Planner?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents NaviAgent, a bilevel planning framework for tool-use agents. It separates high-level reasoning (deciding when to respond, clarify, retrieve, or execute) from low-level execution using a Tool World Navigation Model (TWNM), a dynamic graph that captures dependencies among tools. Their quantitative experiments show significant gains in task success and completion rates over baselines.

### Strengths
The research problem is timely and important, given the rise of agentic LLMs

The quantitative experiments show consistent improvements over the baselines.

### Weaknesses
1. The paper provides no qualitative or quantitative analysis of the learned graph structure; TWNM is evaluated only indirectly through overall task performance, making it difficult to see what the graph actually learns.

2. The high-level decision labels (Direct Response / Clarify / Retrieve / Execute) are derived from rule-based relabeling of ToolBench and API-Bank traces with additional synthetic augmentation, rather than real human data. This raises concerns about whether the learned planner generalizes to real-world cases, particularly in the Direct Response and Clarify categories.

### Questions
1. In Section 3.3 (L319–323), the paper mentions repeating recombination “until infeasibility,” but the stopping criterion is not defined. Could the authors clarify how termination is determined?

2. How sensitive is the planner to the 4 action taxonomy? Could a different or better set of decision type help?

3. The paper claims TWNM supports dynamic integration of new tools; do you have evidence for tool generalization unseen during training?

4. How expensive is maintaining and updating TWNM as tool set grows?

### Soundness
2

### Presentation
3

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
The paper proposes the NaviAgent bilevel framework, which decouples task planning and execution in LLM tool orchestration. Integrating TWNM (Tool World Navigation Model) to dynamically model tool dependencies and enable closed-loop optimization, NaviAgent outperforms baselines significantly in Task Success Rate (TSR) on the API-Bank and ToolBench datasets, achieving efficient navigation in large-scale tool ecosystems.

### Strengths
1.  Its architectural innovation decouples key components (planning and execution).
2.  The  TWNM design unifies the capture of both tool structural and behavioral dependencies.
3.  The paper have conducted comprehensive experiments cover multiple models and scenarios.
4. The paper is well-written, featuring a clear logical structure.

### Weaknesses
1.  There is a gap between tools in simulated and real environments. Real-world APIs are diverse and dynamic, with frequent error fixes, feature updates (e.g., new parameters added), or temporary outages. Although TWNM incorporates a dynamic graph evolution design, it relies on historical execution feedback to update the graph structure, leading to a time gap—for instance, if an API’s error is just fixed but its weight in the graph remains low, NaviAgent may still avoid using it; conversely, sudden API failures without timely pruning result in invalid calls.
2.  Real-world APIs are extensive, and numerous tools not included in the initial graph exist, causing TWNM to fail in generating optimal toolchains.

### Questions
please refer to Weaknesses

### Soundness
3

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
The paper addresses the challenges LLM-based agents face when orchestrating large-scale, dynamic tool ecosystems, specifically targeting issues arising from sequential, step-by-step tool invocation, such as error accumulation and inability to handle complex inter-tool dependencies. The authors propose NaviAgent, a bilevel framework that decouples high-level task planning from low-level tool execution.

### Strengths
- **Bilevel Decoupling:** The core architecture uniquely decouples high-level task reasoning (the four-dimensional decision space: respond, clarify, retrieve, execute) from low-level tool orchestration. This contrasts with standard ReAct-style agents that interleave reasoning and single-step execution, often leading to error accumulation in complex tasks.
- **Evolving Tool World Navigation Model (TWNM):** Moving beyond static tool graphs, the TWNM is highly original in its integration of "behavioral chains" derived from actual execution traces alongside standard "structural chains" (API schemas). Treating inter-tool dependency discovery as a link prediction problem using Heterogeneous Graph Transformers is a sophisticated approach to a typically heuristic-heavy domain.
- **Search Algorithms for Toolchains:** The adaptation of classic search algorithms—specifically Alpha-Beta pruning for backward search and a hybrid genetic/simulated annealing heuristic for forward search—to orchestrate entire toolchains is a creative departure from standard retrieval-augmented generation (RAG) or simple depth-first search methods.

### Weaknesses
- **Cold-Start**: The Tool World Navigation Model (TWNM) heavily relies on "behavioral chains, derived from historical usage data" and "statistical weight... reflecting empirical invocation patterns". This creates a significant cold-start problem. The framework might underperform significantly in new domains where these rich execution traces are unavailable, yet the paper does not quantify this degradation.
- **Justifications of Four-Dimensional Decision Space**: The high-level planner uses a fixed "four-dimensional decision space" (Direct Response, Intent Clarification, ToolChain Retrieval, Tool Execution). While functional, it is not clear if this specific taxonomy is optimal or necessary compared to a more flexible, LLM-driven dynamic planning approach. It risks being too rigid for edge cases that don't fit neatly into these four categories (e.g., partial execution with mid-stream replanning without full re-retrieval).
- **No Optimization for Arguments**: For tool function calling tasks, selecting the correct function is important. But most time, LLMs usually fail at using the correct arguments for the functions.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
