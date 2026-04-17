# GTA: Graph Theory Agent and Benchmark for Algorithmic Graph Reasoning with LLMs

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Language Models (LLMs) are increasingly applied to tasks involving structured data, such as graphs. However, their ability to perform complex algorithmic reasoning over graph-structured inputs remains under-explored. Existing benchmarks typically focus on basic reasoning over small graphs or code generation for graph tasks, but they do not provide models with direct access to graph-structured data, which limits a comprehensive evaluation of their graph reasoning capabilities.

To address this gap, we introduce **Graph Theory Bench (GT Bench)** , a challenging new benchmark featuring 44 diverse graph problem types (ranging from connectivity to minimum-cost flow) across over 100,000 instances with varied input representations (natural language, structured language, adjacency list, adjacency matrix). GT Bench is designed specifically to evaluate the ability of LLMs to perform multi-step algorithmic reasoning on graph-structured tasks.

Our evaluation uncovers a critical dependency between LLM performance and the chosen input graph representation, which varies with graph structure (e.g., density, topology). Based on these findings, we propose the **Graph Theory Agent (GTA)** , a novel framework that enhances LLM graph reasoning by employing an adaptive input representation selector and decomposing the algorithmic solution into manageable sub-steps. Experiments demonstrate that GTA significantly improves the ability of LLMs to solve complex graph problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces **GT Bench**, a large-scale benchmark designed to systematically evaluate large language models on algorithmic graph reasoning tasks. It covers 44 graph problem types and over 100k instances, presented in four input formats (natural language, structured language, adjacency list, and adjacency matrix). Based on insights from GT Bench, the authors further propose **Graph Theory Agent (GTA)**—a modular framework that adaptively selects the best input representation and decomposes high-level algorithmic plans into executable substeps. Experimental results show that GTA significantly enhances LLM performance across diverse graph reasoning tasks, highlighting the crucial role of input representation and structural adaptability.

### Strengths
S1. The paper presents a comprehensive, well-designed benchmark that fills a major gap in evaluating LLMs’ algorithmic reasoning abilities on graph-structured data, beyond simple code generation or node-level tasks.

S2. The introduction of GTA as an adaptive, modular reasoning framework is conceptually elegant and empirically effective. The decomposition pipeline (selector → generator → decomposer → executor) is well-motivated and aligns with recent trends in agentic reasoning research.

### Weaknesses
W1. The paper does not address large-scale graphs that may exceed the LLM’s context window. While GT Bench focuses on medium-sized graphs suitable for in-context reasoning, real-world graph problems (e.g., social networks) often involve millions of nodes and edges that cannot fit into a single prompt. The paper lacks discussion of how GTA or LLM-based reasoning could be extended to such out-of-context or distributed graph scenarios.

W2. The paper lacks an analysis of how model size affects performance. Although the results reveal clear performance gaps among LLMs under different input representations, the paper does not examine how model scale influences these differences or whether GTA’s improvements generalize consistently across models of varying sizes.

### Questions
Q1. How does GTA handle or plan to handle large-scale graphs that exceed the LLM’s context window? For example, could graph partitioning, retrieval-based reasoning, or external memory mechanisms be used to extend GTA beyond in-context settings?

Q2. How does model size influence performance trends across GT Bench? Can the authors provide a detailed analysis or visualization showing how scaling (e.g., 7B, 32B...) affects reasoning robustness, representation preference, or adaptation within GTA?

Q3. Table 4 shows that different models favor different input representations. In practice, how does GTA determine the most suitable representation for a specific downstream LLM?

### Soundness
3

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
5

### Summary
This paper introduces GT Bench, a benchmark comprising 44 graph-theoretic problem types with over 100,000 instances across four input representations, designed to evaluate multi-step algorithmic reasoning capabilities of LLMs. 
The authors demonstrate that LLM performance varies significantly with input representation.
 Based on these findings, they propose the Graph Theory Agent (GTA), a framework that adaptively selects optimal input representations and decomposes algorithmic solutions into sub-steps. 
Experiments show GTA outperforms baselines, including prompting strategies and automated agent frameworks.

### Strengths
1. The paper provided a comprehensive benchmark design addressing critical gaps in previous works.
2. The GTA framework works well in the benchmark and solves the problems mentioned.
3. GTA is also cost-friendly, introducing minimal overhead.

### Weaknesses
1. **Bad paper organization.** Table 2 presents the best representations per task without specifying which models were evaluated or whether the results represent averages. Experimental settings (models, hyperparameters) are deferred to Section 4.1 while representation sensitivity conclusions appear in Section 2.4. Table 3 shows accuracy across representations but provides no information about which model(s) produced these results. Section 4.5 explicitly uses the Appendix to extend the paper length, leaving the main body of the paper non-self-consistent. \
I understand that there has been a lot of work done in the paper, but it is still recommended that the authors reorganize the paper to make it clearer.

2. **Scalability** All experiments restrict graphs to ≤60 nodes, which is orders of magnitude smaller than real-world graphs. Figure 2 demonstrates significant degradation even within this range for GTA. 
When scaled up, methods like GraphTeam use code execution, which may offer superior scalability, yet current benchmarks may underestimate this line of work's superiority.

3. **Unfair baseline comparisons due to training data advantage.** GTA receives task-specific training exclusively on GT Bench while all baselines use only Phi-4 without GT Bench training, creating fundamental unfairness. No experiments train competitive baselines on GT Bench to isolate whether gains stem from architectural innovations or simply from task-specific fine-tuning. GraphWiz-DPO achieves only 32.5% but uses different training data and isn't adapted to GT Bench's four representations, making it unclear whether properly trained baselines would match GTA. \
**(Minor)** And it is questionable if MAS-based methods could yield stronger performance if combined with superior models, since they rely more on each agent's reasoning ability.

4. **Stronger pretrained models may obviate the need for GTA's complex framework.** Comparing Tables 4 and 5 reveals that strong pretrained models without specialized training achieve comparable performance, suggesting context capacity and general capabilities matter more than representation selection. Table 12 shows that GTA provides minimal gains for strong models.

5. **(Minor)** All main results report accuracy from single runs without standard deviations or confidence intervals.

6. As a benchmark, I would be excited if more detailed and clear analyzes is provided in the main paper. Currently, Table 2 provides raw results without further visualization, leaving the conclusions hard to interpret. 
**(Minor)** The unclear results presentation also makes the current GLbench somewhat too empirical, with little insightful analysis. 
One only learns that in each task, different templates outperform the others, but still don't know much about why.

### Questions
The paper provides no evidence that graph-theoretic reasoning tests capabilities orthogonal to existing benchmarks (MATH, GSM8K, HumanEval), leaving unclear what unique intelligence aspects GT Bench evaluates. 
Will an LLM be better at reasoning, also good at graphs? Or have any interesting contradictions been found in the benchmark? (e.g. If an LLM is good at math/reasoning but fails at graphs)

### Soundness
3

### Presentation
1

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
This paper introduces Graph Theory Bench (GT Bench) and Graph Theory Agent (GTA) to evaluate and improve LLMs abilities in algorithmic graph reasoning. GT Bench is a large-scale benchmark comprising 44 graph-theoretic tasks and over 100,000 instances presented in four input modalities, enabling systematic analysis of how representation formats and graph structures affect reasoning performance. Experiments show that LLM accuracy varies significantly across input representations, revealing strong representation sensitivity. Building on this insight, the authors propose GTA, a modular framework that adaptively selects input formats and decomposes problem-solving into structured sub-steps. Evaluations across multiple datasets demonstrate that GTA consistently outperforms existing prompting and agent-based baselines, with ablation studies confirming the contribution of its adaptive selection and decomposition modules. Overall, the paper provides a comprehensive benchmark, clear empirical insights, and an effective agentic framework that meaningfully advances the study of graph reasoning in LLMs.

### Strengths
- The paper offers a comprehensive empirical study. It covers 44 graph problem types and over 100,000 instances across four input modalities, includes easy/hard splits, and analyzes performance at the per-task and per-representation level. The authors also examine scalability with graph size, inference cost, and executor swaps, which gives a balanced view of where current LLMs succeed and fail. 

- The methodological contribution is concrete and grounded in the findings. Building on the observed representation sensitivity, the authors design the GTA framework with an adaptive input representation selector and a plan–decompose–execute pipeline. The decomposer is trained with SFT and DPO, and ablation studies show that each module contributes to the final gains. The approach improves accuracy without relying on external code execution or modifying the base executor.

- The writing is clear and well structured. Task definitions and ground-truth algorithms are provided, prompt templates are documented, and limitations are discussed. The case studies illustrating representation effects help readers understand why certain formats are preferable for specific tasks and graph structures.

- The empirical insight on representation sensitivity is carefully substantiated. The paper systematically shows that no single input format is universally best and that performance depends on graph density, topology, and task type. This motivates the adaptive selector in GTA and adds a useful perspective for future work on language-based graph reasoning.

### Weaknesses
- It is unclear why not simply input all of the four graph representations together as a baseline. Maybe the result will be better?

- The paper lacks a comparison with human expert performance, even on a subset of the tasks. Including such results would help readers understand how far current LLMs have progressed in graph reasoning relative to human capabilities.

- Table 4 shows that different models prefer different graph representations, yet the training of the Selector depends on a specific model. This implies that a new Selector must be re-trained each time when the model changes, which limits the generality and practicality of the proposed approach.

- The description of the baseline in Section 4.2 is confusing. Does Vanilla prompting mean using the same workflow without additional training, or does it refer to using a single model without the agentic framework? Clarifying this distinction would make the experimental setup easier to understand.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces GT Bench, a graph reasoning benchmark with 44 problem types and four input representations, and proposes GTA, an adaptive framework combining representation selection and algorithmic decomposition. While the scale is notable, the core ideas—multi-format evaluation and stepwise reasoning—are incremental.

### Strengths
The introduction of GT Bench offers a diverse dataset with 44 problem types across four input modalities. Its focus on multi-step algorithmic reasoning, rather than simple graph queries or code generation, fills a gap in existing evaluations.

The empirical demonstration of LLMs' sensitivity to graph input formats is rigorous. The findings, which link optimal representation to graph structure (e.g., dense graphs with AM, trees with SL), provide actionable insights for the community, moving beyond mere observation to offer practical guidance.

### Weaknesses
•Incremental Conceptual Novelty: The core ideas with multi-format evaluation and algorithmic decomposition are refinements of existing concepts. The heuristic selector is simplistic, and the learned version offers only minor gains, suggesting the representation selection problem may be less complex than claimed.

•Incomplete Baseline Comparisons: The omission of direct comparisons with strong, graph-specialized models (e.g., GraphWiz) is a notable gap. It remains unclear if GTA's gains stem from its adaptive framework or if they could be matched by a model fine-tuned specifically for graph reasoning.

•Limited Scalability Demonstration: Experiments are confined to small graphs (≤50 nodes), leaving scalability to real-world scales unexplored. The work does not address the critical challenge of handling larger graphs within finite context windows, limiting the claim of evaluating "realistic" complexity.

### Questions
1. The representation sensitivity is a key finding. Beyond the selector, did you explore if this insight could be used to create a more robust, representation-agnostic training curriculum for LLMs to improve their inherent graph reasoning capabilities? 

2. The heuristic selector performs reasonably well. In what specific scenarios or on which types of tasks does the learned selector provide the most significant advantage over the heuristic rules, and why? 

3. Can GTA generalize to graphs with >100 nodes, given context limits? Providing empirical evidence and discussions could be appreciated.

### Soundness
2

### Presentation
3

### Contribution
2
