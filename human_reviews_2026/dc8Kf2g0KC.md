# GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
This paper introduces GraphOmni, a comprehensive benchmark designed to evaluate the reasoning capabilities of LLMs on graph-theoretic tasks articulated in natural language. GraphOmni spans diverse graph types, serialization formats, and prompting schemes, substantially extending upon prior efforts in both scope and depth. Through systematic evaluation, we uncover critical interactions among these dimensions, revealing their decisive impact on model performance. Our experiments show that state-of-the-art closed-source models such as Claude-3.5 and o4-mini consistently lead overall, yet still leave considerable headroom, while open-source models display pronounced sensitivity to various design choices. Beyond the standard scope, larger graphs, real-world graphs, and additional NP-hard tasks are further discussed. We further analyze efficiency via output token usage, highlighting cost–accuracy trade-offs, and introduce a reinforcement learning-based optimizer that adaptively selects factor combinations, reducing evaluation cost by 75\% while retaining strong accuracy. This flexible and extensible benchmark not only deepens understanding of LLM performance on structured graph reasoning but also establishes a robust foundation for advancing model design and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a comprehensive benchmark for evaluating LLMs' ability on graph-theoretic tasks. The evaluation is conducted on six graph-theoretic problems, which are encoded with different serialization formats and prompting schemes. The main results show that current LLMs have substantial room for improvement on graph-theoretic problems, and there is no consistently good serialization and prompting strategies among different tasks and models. The paper further introduces a reinforcement learning-based optimizer that adaptively selects factor combinations.

### Strengths
The benchmark is comprehensive with multiple aspects of encoding strategies of graph-theoretic problems, covering both open-sourced and closed-sourced LLMs. The workload is heavy, and the experimental results are detailed.

### Weaknesses
1. Each part of the key components (Benchmark Tasks, Graph Types, Prompt Schemes, and Serialization Formats) was proposed and researched in previous works. Although this paper considers all four components, the experiments and analysis are still conducted individually, and more importantly, do not bring substantially **new** insights or findings.

2. For most experiments, results are reported without **in-depth interpretation**, and therefore, the inspiration provided is limited.

3. The **conclusions are limited to six graph-theoretic tasks** (Connectivity, Cycle detection, Diameter, BFS, Triangle, Shortest path). What are the differences among the six tasks, and do they measure different aspects of LLMs' graph reasoning ability? It's unknown whether the current conclusion changes when evaluating on other graph-theoretic tasks.

### Questions
1. Which main results are first found in this paper or are different from previous benchmarking papers?

2. What's the real reason for the LLM's unsatisfying performance on graph-theoretic problems? In Result 3, it says representative categories of errors are commonly the misinterpretation of serialization formats and incorrect reasoning about graph-theoretic concepts. However, I think this can not be claimed as "fundamental" and can be solved by prompt engineering.

3. Why do open-source models benefit from multi-shot exemplars, whereas they do not help closed-source models much?

4. What is the complexity of the six tasks (e.g., time complexity related to node and edge numbers), and does the ranking align with the performance of LLMs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes GraphOmni, a comprehensive benchmark for LLMs on Graph-theoretic tasks.
The benchmark makes breakthroughs in the number of instances, the number of investigated schemes, and the number of serializations.
Key findings from using GRAPHOMNI include: No single serialization format or prompt works best for all tasks; performance varies widely, and even state-of-the-art LLMs show significant room for improvement on graph reasoning tasks.
Motivated by these findings, the authors also propose an RL-inspired selector to dynamically choose the best settings for a given task, thereby improving performance.

### Strengths
1. The benchmark is comprehensive in both scale and involved methods, providing a solid exploration in graph reasoning.
2. The analysis is very solid and presents interesting viewpoints. The findings in Section 4.2 provide meaningful insights.
3. The paper is very clearly written overall. The presentation is excellent and the logic is complete.

### Weaknesses
1. The significance of comparing LLMs on graph reasoning is questionable. The challenges LLMs face in graph reasoning seem largely constrained by context length, rather than being focused on core reasoning abilities like mathematical reasoning. There is a lack of discussion on what specific deep capabilities of LLMs the study of graph reasoning is meant to reflect.
2. Expanding on point #1, a key characteristic of graph reasoning is that even humans often rely on external tools (e.g., writing code or drawing diagrams) to solve problems like finding shortest paths or Hamiltonian cycles. The inability of a human to solve these problems entirely "in their head" is not typically taken as a sign of deficient reasoning ability. Therefore, the practical significance of evaluating a single LLM's isolated reasoning capability on these tasks is limited.
3. The study only considers graphs with up to 50 nodes, which is still too small in scale. 
4. The description of the RL-based method is too brief. While I understand space constraints, the current description fails to adequately explain in the main text: 1) the core limitation it addresses, 2) the experimental setting, and 3) it lacks a comprehensive comparison with baselines. These elements should be fully described in the main body to ensure the paper is self-contained and coherent.

### Questions
See Weaknesses 1 & 2

### Soundness
4

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
2

### Summary
The paper introduces GraphOmni, a comprehensive and extensible benchmark for evaluating large language models on graph-theoretic reasoning tasks expressed in natural language. It systematically varies three critical dimensions—graph type, serialization format, and prompt scheme—across six canonical tasksand three difficulty levels. Experiments reveal substantial variability: no single model or prompt-serialization combination consistently dominates. The authors further introduce a reinforcement learning–based optimizer that adaptively selects optimal factor combinations.

### Strengths
S1. The benchmark framework is comprehensive, jointly considering three critical dimensions — graph types, serialization formats, and prompting schemes — to provide a multidimensional evaluation of LLMs’ graph reasoning abilities.

S2. The paper explores the use of reinforcement learning to search for optimal combination strategies, achieving significant cost reduction while maintaining high performance.

### Weaknesses
W1. The benchmark mainly focuses on classical graph-theoretic problems, which limits its applicability to real-world large-scale or labeled graphs that involve complex attributes, heterogeneous structures, or task-specific supervision.

W2. While GraphOmni reports accuracy across various tasks and factor combinations, the paper lacks a systematic analysis of how different task types interact with specific factor combinations.

### Questions
Q1. How do the authors ensure that the comparison between LLM-generated outputs and the ground truth is accurate? Given that the answers are evaluated from natural-language responses, what mechanisms or verification procedures are used to avoid misparsing or misjudging correctness during evaluation?

Q2. Could the authors provide a cross-factor analysis showing how different graph-theoretic tasks align with combinations of graph type, serialization format, and prompt scheme?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a large-scale benchmark (241,726 queries) for evaluating LLMs’ reasoning ability on graph-theoretic problems expressed in natural language. The framework systematically varies three dimensions—graph types, serialization formats, and prompt schemes—and assesses model performance across multiple difficulty levels and tasks (e.g., connectivity, BFS, cycle detection, triangle counting). The authors analyze open- and closed-source models, report detailed results, and propose a reinforcement learning–based optimization method (RL-Opt, RL-Scale) to reduce evaluation cost by 75% while maintaining accuracy.

### Strengths
1. The benchmark jointly varies graph type, serialization, and prompt scheme—more complete than prior work. This design provides strong coverage for structured reasoning.

2. Includes 241,726 queries across 7 graph generators and 9 prompt types; this ensures statistical reliability and generalizability.

3. Both open-source (Llama‑3, Qwen‑3) and closed-source (GPT‑4o, Claude‑3.5) models are evaluated to ensure representativeness.

4. Error analysis (Sec. 4.1, Result 3; Appendix E.3): Provides qualitative insights into misinterpretations (e.g., misunderstanding diameter definition), strengthening interpretability.

### Weaknesses
1. The benchmark provides empirical comparison but lacks analytical justification for observed behaviors—e.g., why serialization types yield specific effects. No direct evidence found in the manuscript.

2. The reuse of m, M, e, E in RL metrics (Sec. 4.4) is not clearly linked back to earlier sections or consistent definitions; may confuse readers.

3. No robustness check under prompt noise: All prompts assume fixed phrasing; real-world variance in natural language not examined. 

4. While model output tokens analyzed (Sec. 4.3), inference or hardware cost details (e.g., GPU hours) are missing.

### Questions
1. In Sec. 4.4; Table 4, you define Cost = e/E and Rate = m/M but provide no equation for the reward function or RL objective. Could you explicitly describe: the state/action/reward formulation of your RL process.

2. The symbol m is reused for “edge count” in Sec. 2 (Graph Generators) and again for “accuracy” in Sec. 4.4 (Rate = m/M). please clarify notation consistency.

3. Sec. 4.4 presents “RL-Scale” to assess scalability. What feedback signal or evaluation metric was used to balance computational cost and accuracy when adding new serialization factors?

4. The paper systematically varies prompt schemes but all examples appear deterministic (Sec. 2; Appendix A.5). Did you test robustness under paraphrased or noisy prompts to evaluate linguistic invariance?

5. Table 1 omits recent benchmarks in graph reasoning tasks (e.g., GraphWild in GCoder, 2025).

### Soundness
2

### Presentation
3

### Contribution
3
