# Improving Language Agents through BREW: Bootstrapping expeRientially-learned Environmental knoWledge

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large Language Model (LLM)-based agents are increasingly applied to tasks requiring structured reasoning, tool use, and environmental adaptation, such as data manipulation, multistep planning, and computer-use automation. However, despite their versatility, current training paradigms for model weight optimization methods, like PPO and GRPO, remain relatively impractical with their high computational overhead for rollout convergence. In addition, the resulting agent policies are difficult to interpret, adapt, or incrementally improve. To address this, we investigate creating and refining structured memory of experiential learning of an agent from its environment as an alternative route to agent optimization. We introduce \textbf{BREW} (Bootstrapping expeRientially-learned Environmental knoWledge), a framework for agent optimization for downstream tasks via KB construction and refinement. In our formulation, we introduce an effective method for partitioning agent memory for more efficient retrieval and refinement. BREW uses task graders and behavior rubrics to learn insights while leveraging state-space search for ensuring robustness from the noise and non-specificity in natural language. Empirical results on real world, domain-grounded benchmarks---OSWorld and $\tau^2$Bench---show BREW achieves 10--20\% improvement in task precision, 10--15\% reduction in API/tool calls leading to faster execution time, all while maintaining computational efficiency on par with base models. Unlike prior work where memory is treated as static context, we establish the KB as a modular and controllable substrate for agent optimization---an explicit lever for shaping behavior in a transparent, interpretable, and extensible manner.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new method named BREW, it can update the insights based on MCTS. The author design Reflector Agent to construct (concept, insight) pairs and use Integrator Agent to exapnd the insight based on a given concept. It than use EG-MCTS to expand this insight in the tree and use it.

### Strengths
1. Consider how to construct and update the insight in the tree is an important topic.

2. This method can continuously update the insight in the tree, which is important in real-world applications.

### Weaknesses
1. It seems that the size of concepts can't be changed.

2. Lack of introducing the insight-based memory work, like Expel[1], MSI-Agent[2] and SelfGoal[3]. This will also harm the contribution 1 (line 111-114) in this paper.

3. The writing is not clear. The method is not easy to understand.

4. It only works in several sub-tasks, although I agree it true reduce the total steps and costs in all works.

[1] Expel: Llm agents are experiential learners

[2] MSI-Agent: Incorporating Multi-Scale Insight into Embodied Agents for Superior Planning and Decision-Making

[3] SelfGoal: Your Language Agents Already Know How to Achieve High-level Goals

### Questions
1. The si, sj, si+1 should be presented in the trees in Figure 2.

2. Will you use all insights in Dcurrent? It may cause irrelevant insight to be pushed into the context. (The concept that not related to the current task)

3.  Considering Figure 2, it seems that the sabling nodes generation is independent of the current node. Is this may cause some information loss?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To address the limitations of opaque training and the inability of language agents to learn from historical experience, this paper introduces the BREW framework. This approach first extracts "insights" and "concepts" from an agent's historical trajectories. It then employs an MCTS algorithm to search and optimize the content of these knowledge documents, identifying the version that maximizes task rewards . This process serves to offline-optimize an external knowledge base (KB). Ultimately, during execution, the agent can retrieve from this optimized knowledge base to achieve significant improvements in both performance (task precision) and speed (reduced API calls) on benchmarks such as OSWorld.

### Strengths
1. Formalizes the problem of constructing an optimal knowledge base (KB) as a state-space search problem, targeting the KB's content for optimization rather than model parameters.
2. Stores the agent's experiential knowledge in a human-readable knowledge base (KB) instead of opaque model weights, rendering the agent's behavior transparent, interpretable, and debuggable.
3. Offloads the computationally expensive state-space search optimization to an offline phase , ensuring that online execution only requires a lightweight retrieval step and thus incurs no additional inference latency.

### Weaknesses
1. The framework's effectiveness is heavily dependent on the quality of task grading. Inherent biases and noise from LLM-based evaluation could lead to the KB being constructed from skewed or suboptimal insights.
2. Scalability is questionable. The EG-MCTS algorithm requires running MCTS searches in parallel for every meta-concept. This optimization cost could become prohibitively high as the agent needs to learn thousands of concepts.

### Questions
1. What is the true computational cost of optimizing the knowledge base via MCTS? The total optimization cost is explicitly dependent on the number of meta-concepts. If this number is large (e.g., 1,000), the cost would be prohibitively high. Can the authors report the number of meta-concepts required for a given task and its corresponding MCTS cost?
2. Could the Reflector Agent and Integrator Agent be replaced by smaller, more cost-effective models? Using GPT-4.1 is quite expensive.

### Soundness
3

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
3

### Summary
This paper introduces BREW (Bootstrapping expeRientially-learned Environmental knoWledge), a framework that optimizes LLM-based agents by constructing and refining a structured, interpretable knowledge base (KB) from past interactions instead of fine-tuning model weights. BREW decomposes agent memory into conceptlevel documents and optimizes them via a novel Expand-and-Gather MCTS algorithm that jointly maximizes reasoning correctness and retrievability. Experiments on OSWorld, τ²-Bench, and SpreadsheetBench show consistent improvements over memory-based baselines, demonstrating that structured experiential memory can enhance efficiency and adaptability in long-horizon reasoning tasks.

### Strengths
1. The framework introduces a human-readable, document-based memory representation that enhances the interpretability and traceability of agent behavior.
2. The concept-level structured memory design is a novel contribution that enables modular organization and more efficient, task-aligned knowledge retrieval.
3. The use of the Expand-and-Gather MCTS for optimizing knowledge base construction is methodologically innovative and avoids the pitfalls of greedy or one-shot updates.

### Weaknesses
1. While the use of MCTS contributes to performance, the paper lacks a quantitative analysis of its computational overhead and does not compare resource consumption with baseline methods.
2. The performance improvements shown in Figure 3 are generally modest, with several task subcategories showing parity with or only minor gains over the baseline.
3. The framework is relatively complex, involving multiple agents and stages; its generalizability across different LLMs (especially open-source models) remains unclear and untested.

### Questions
1. Could you provide more details on the computational cost of the BREW framework, particularly regarding the MCTS optimization (e.g., runtime, hardware requirements, memory usage)?
2. Have you attempted to apply BREW on smaller or open-source LLMs? If so, how does performance compare, and what challenges arise?
3. For the OSWorld benchmark, several subtasks show limited gains—do you have further insights into what constrained BREW’s improvements in those cases?
4. While some prompt templates have been shared, would you consider releasing runnable code for the full pipeline, including data processing, reflection, and memory integration? This would be critical for reproducibility and further research.

### Soundness
2

### Presentation
2

### Contribution
2
