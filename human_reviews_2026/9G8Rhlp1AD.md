# Agent-REINFORCE: Searching Compute-Optimal Multi-LLM Collaboration Graph for Test-Time Scaling

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of \textit{searching for compute-optimal model combinations and architectures in TTS under a fixed budget.} We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirical insights into TTS collaboration graphs. Guided by these insights, we propose Agent-REINFORCE, an LLM-agent-augmented framework that mirrors the REINFORCE pipeline by mapping sampling–gradient–update to sampling–feedback–update, where feedback serves as a textual gradient to update the probabilistic graph and efficiently search for optimal multi-LLM collaboration graphs. Experiments show that Agent-REINFORCE outperforms both traditional and LLM-based baselines in sample efficiency and search performance, and effectively identifies optimal graphs under joint objectives of accuracy and inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes Agent-Reinforce, an algorithm to optimize a computation graph of multiple language models for collaborative problem solving. Empirical results show its promise against various baselins.

### Strengths
+ multi-llm collaboration is an important direction
+ formulating the problem as a graph optimization problem is nice

### Weaknesses
- The paper argues that many things are fixed in previous works (graph structure, etc.), while this work makes them flexible and optimizable. One thing becomes fixed in this work though: the role of model/agents. In section 3, the authors hand-crafted two roles: assistant and fuser, with their specific prompts and configurations. 1) Do you forsee any other potential role? 2) Is it possible to make roles a bit more flexible than hand-crafting prompts and manually designing roles? 3) If a new role or two is added, how would the system cope?

- How is "Theorem 1", line 186, a theorem? It's a conclusion about the cost of the computational graphs under a set of hyperparameters, not a theorem that was "proved" by anything?

- There does not seem to be any qualitative example (specific generated texts by the models in the computation graph), main paper or appendix. Maybe I missed it.

- The word "(test-time) scaling" was mentioned many times as a motivation for this approach. It would be nice to have some sort of scaling diagram, that by allowing more more models/larger graph sizes the performance goes up to some extent.

### Questions
please see above

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
3

### Summary
The work study the problem of searching for compute-optimal model combinations and architectures in test-time scaling under a fixed budget. It introduces Agent-REINFORCE, an LLM-agent framework that conducts budget-aware, feedback-driven search on collaboration graphs. Experiments show that the proposed algorithm outperforms baselines in both search efficiency and performance.

### Strengths
- Overall, I really like the proposed method. Using agent to search for model combinations and architectures in TTS is an interesting and well-motivated idea.

- The work explicitly incorporates budget constraints, either FLOPs or dollar cost, which provides a realistic perspective on multi-LLM collaboration in real deployment scenarios.

- The experimental setup is thorough, and the results are well-supported with clear justification, detailed analysis, and informative ablations.

- The paper is exceptionally well-presented. The writing is clear, figures and plots are informative, and the algorithmic components are introduced with clarity. I gave presentation score of 4.

### Weaknesses
The proposed method demonstrates strong performance in both accuracy and efficiency according to Table 1, substantially outperforming the baselines. That said, I have two concerns:

- Table 1 omits results for a single-model baseline, which makes it difficult to gauge how much of the gain is truly due to multi-LLM collaboration rather than simply using a stronger model. For instance, according to Figure 3(c), a 3B Mix model can achieve 54 on MATH, whereas a single LLaMA-8B reaches 64 on MMLU, which exceeds both the baseline and proposed method in terms of both accuracy and efficiency.

- The cost analysis appears to exclude the overhead associated with the search agent (i.e., DeepSeek‑R1), which is expensive.

### Questions
- what if we don't pre-assign roles?

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
4

### Summary
This paper addresses the problem of finding compute-optimal multi-LLM collaboration architectures for test-time scaling (TTS). The authors formulate TTS as a multi-LLM collaboration graph where nodes represent LLMs with assigned roles (fuser/assistant) and edges denote information flow. They identify three empirical insights about TTS behavior: (1) tasks prefer specific model family/size combinations, (2) width and depth have task-dependent optima, and (3) width-depth interdependence. Based on these insights, they propose Agent-REINFORCE, an LLM-agent-augmented framework that mirrors REINFORCE's sample-gradient-update pipeline using textual feedback as gradients. Experiments on MATH, MMLU, and HumanEval demonstrate superior search efficiency and performance compared to traditional and LLM-based baselines.

### Strengths
- Well-motivated problem: The paper identifies a real gap in existing TTS methods—fixed architectures and single-model usage—and proposes a principled solution.
- Comprehensive empirical analysis: The pilot experiments (Section 4) provide valuable insights that generalize across tasks and inform the algorithm design.
- Strong experimental results: Agent-REINFORCE consistently outperforms baselines in both search efficiency (convergence time) and final performance across all three benchmarks.
- Practical framework: The method handles real-world constraints (budget, latency) and supports multiple cost metrics (FLOPs, wall-clock time, monetary cost).

### Weaknesses
- Limited scope of evaluation: Only three tasks (MATH, MMLU, HumanEval) from limited domains (math, knowledge, code). Only LLaMA and Gemma models tested; unclear how insights generalize to other model families. No evaluation on truly large-scale models (e.g., 70B+) due to computational constraints.
- Computational cost of search: The paper does not report the cost of running DeepSeek-R1 as the search agent. Search time includes only environment evaluation, not LLM agent inference. For practical deployment, the amortization of search cost needs analysis.
- Insight generality: Insight 1 (model family preference) may be dataset/domain-specific. Insight 2 (non-monotonic scaling) is known in TTS literature but not cited adequately. Insight 3 (width-depth interdependence) is intuitive; the novelty is primarily empirical validation
- Statistical rigor: Limited discussion of variance across runs. Some results show confidence intervals (Fig. 3c-d) but most do not. No significance testing for performance differences.

### Questions
- Search cost amortization: How many test queries are needed to amortize the search cost? Can you provide a break-even analysis comparing search time vs. inference time savings?
- Insight generalization: How do the three insights hold for tasks outside your evaluation set (e.g., long-form generation, multimodal tasks, code debugging)? Have you tested on additional domains?
- Agent model choice: Why DeepSeek-R1? How sensitive are results to the choice of search agent? Have you tried smaller/larger agents or different model families?
- Graph constraints: The current formulation allows arbitrary DAGs. Would restricting to specific topologies (trees, chains) improve efficiency while maintaining performance?
- Budget sensitivity: How does performance change with budget granularity? The paper tests specific budgets (18, 42, 80) but not the full spectrum.
- Multi-objective optimization: The latency-aware optimization is mentioned briefly. Can you provide more details on the Pareto front and trade-off curves?
- Online adaptation: Can the searched graph adapt during deployment if data distribution shifts? Or is it fixed post-search?

### Soundness
2

### Presentation
2

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
This paper addresses an important yet underexplored problem in test-time scaling (TTS): searching for compute-optimal multi-LLM collaboration graphs under a fixed budget for specific tasks. The authors formalize this as a multi-LLM collaboration graph optimization problem, where nodes represent LLM primitives with role assignments (fuser/assistant) and model assignments, and edges represent information flow. Through pilot experiments, the authors identify three key insights: task-specific preferences for model combinations, task-dependent optima for width/depth beyond which additional compute yields negative returns, and interdependence between graph width and depth. Based on these insights, they propose Agent-REINFORCE, which maps REINFORCE's sample-gradient-update pipeline to sample-feedback-update, leveraging an LLM agent to incorporate prior knowledge for efficient search. Experiments demonstrate that the method significantly outperforms both traditional methods and LLM-based baselines on MATH, MMLU, and HumanEval datasets in terms of both accuracy and search efficiency, while effectively handling joint optimization objectives for accuracy and latency.

### Strengths
1. Novel and important problem formulation: This work is the first to systematically formalize test-time scaling as a multi-LLM collaboration graph optimization problem, filling a critical gap in existing work. While most TTS methods consider only fixed topologies and single models, this paper jointly optimizes architecture, role assignments, and heterogeneous model combinations, offering significant theoretical and practical value to the multi-agent collaboration community.

2. Deep and actionable empirical insights: The three insights discovered through systematic pilot experiments provide both strong algorithmic guidance and valuable empirical understanding of TTS behavior. These insights are thoroughly validated in ablation studies, demonstrating their crucial role in accelerating convergence and improving performance.

3. Clever method design with solid execution: The Agent-REINFORCE framework innovatively replaces REINFORCE's gradient updates with LLM-generated textual feedback, effectively combining the data efficiency of gradient methods with the semantic understanding of LLMs. The multi-stage initialization strategy is particularly clever in significantly reducing the search space, and the experimental design is comprehensive across multiple datasets and optimization scenarios.

4. Convincing experimental results: The method comprehensively outperforms multiple baselines across three representative tasks in both accuracy and search efficiency. Ablation studies clearly demonstrate the contribution of each component, and the method shows good generalization across different budget metrics and joint optimization objectives.

### Weaknesses
1. Incomplete baseline comparisons: While the paper compares against GPTSwarm and TextGrad, it misses important baselines in agent workflow optimization such as AFlow and ADAS, which have become standard comparison methods in this field. Including experimental comparisons and detailed discussions with these methods would better position the work and demonstrate its advantages.
2. The paper lacks quantification of the inference cost of the agent itself during the search process. It would be valuable to explicitly report the agent's FLOPs/dollar overhead and its proportion in the total search budget to comprehensively assess the method's practical efficiency.
3. The experiments use dated benchmarks for TTS research. Evaluating on more recent benchmarks such as LiveCodeBench would strengthen the contributions. Additionally, expanding the model families beyond LLaMA and Gemma to include more widely-used backbones like Qwen would enhance the generalization claims.

### Questions
As shown in weakness

### Soundness
2

### Presentation
3

### Contribution
3
