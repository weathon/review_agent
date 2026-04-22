# Large Language Models as Topological Thinkers: A Benchmark on Graph Persistent Homology

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Persistent homology offers a principled way to capture multi-scale topological structures in graphs, yet it remains unclear whether large language models (LLMs) can understand and reason about such high-order topological concepts. To address this gap, we introduce LLM4PH, the first benchmark designed to evaluate the ability of LLMs to comprehend and apply persistent homology on graphs. Our benchmark decomposes the persistent homology pipeline into four progressively challenging task levels, ranging from simplicial structure understanding to real-world graph inference. It includes 9 sub-tasks spanning 3 synthetic graph sizes and 3 real-world graph datasets, each annotated with topological features such as connected components, simplices, filtrations, and persistence diagrams. We systematically assess LLMs' capabilities in recognizing topological features, reasoning over filtrations, designing filtration strategies, and applying persistent homology for classification. Beyond task-level evaluation, we perform cross-task ablations on prompt encoding and transfer, explore post-training effects, and construct a compositional PH pipeline to assess end-to-end performance. Our results provide the first in-depth view of how well LLMs bridge discrete graph structures with continuous topological abstraction, and offer insights into their potential for structure-aware scientific reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LLM4PH, the first benchmark designed to evaluate whether large language models (LLMs) can understand and reason about persistent homology (PH) — a core method in topological data analysis for capturing multi-scale graph structures. Results show that while LLMs handle low-level structural recognition (e.g., counting connected components) well, they struggle with higher-order reasoning such as dynamic filtration evolution and strategy design. The paper also explores prompt design, post-training effects, and integrated PH pipelines.

### Strengths
1. This paper is the first to test LLMs on topological reasoning, going beyond classical graph reasoning benchmarks. 
2. This paper includes proprietary and open-source LLMs with multiple analysis dimensions

### Weaknesses
1. The PH reasoning tasks, while theoretically rich, may have narrow real-world applicability beyond benchmark testing.
2. Most experiments are on small graphs (≤30 nodes), raising scalability concerns.
3. Results show high sensitivity to prompt format and phrasing, questioning benchmark robustness.
4. The study reports accuracy metrics but offers limited qualitative insight into why models fail at topological abstraction.

### Questions
1. How do you ensure that improvements correspond to genuine topological reasoning rather than pattern memorization of graph-text associations?
2. How scalable is the benchmark to larger, real-world networks (e.g. thousands of nodes)?
3. Have you considered integrating explicit topological modules (e.g., PH libraries) into LLM reasoning loops to test hybrid neuro-symbolic approaches?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a dataset of graphs and a set of tasks to asses the capacity of pretrained LLMs to understand topological tasks, such as predicting the connected components and subsets of persistent homology. The tasks consist of predicting connected components and birth and death times of cycles as well as predicting the critical edges / nodes for them. Harder tasks consist of finding the  best filter function to separate graphs.
A core part  of the paper lies in the evaluation of pretrained models on these task through either their API (for the proprietary models) or on a cluster when the model is open source.

### Strengths
Overall there is some merit in the idea that it can be interesting to understand how well an LLM can understand harder tasks such as predicting filter functions and computing higher order homology. The paper is well-written and the code is provided and well documented.

### Weaknesses
Overall the question why the predication of graphs could be potentially interesting is somewhat hard to understand. Hence the motivation and potential impact on either the field of TDA or machine learning seems somewhat hard to understand. In particular, fundamentally LLMs are designed to predict text rather than topological features and therefore evaluating them on this specific task would need more motivation.
The dataset provided consists of graphs, but the motivation to use exactly this set of graphs is a little bit unclear to me.

### Questions
In line with the weaknesses, there are some answers to be provided for a broader impact on the community.
- What would the reader, or user, compel to use their dataset, as compared to an arbitrary graph dataset? This question is a little unclear to the reviewer.
- What would be the broader impact of this work for either the TDA community or LLMs, if we would want to train a model for a particular problem, predicting it with an LLM would not be feasible anyway and also not expected and we would want to do this at runtime / training time as well.

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
4

### Summary
The paper introduces a benchmark designed to evaluate the ability of LLMs to reason about graph topology through the lens of persistent homology.  The benchmark comprises a hierarchically organised set of tasks and is accompanied by an in-depth comparative study of the performance of  multiple LLMs, as well as analyses of prompt formulation, graph representation, and recurring error types.

### Strengths
[S1] The paper is well written and generally easy to follow.

[S2] The benchmark is thoughtfully designed and covers a good range of PH-related reasoning tasks.

[S3] I particularly appreciate the error analysis presented in Section 3.4, which provides a clear window into model behaviour and failure modes.

### Weaknesses
[W1] My main concern lies in the motivation for this work. Persistent homology is a rather specialised area, and it is unclear why developing LLM benchmarks for PH reasoning is of broad importance. While the experiments are interesting, the contribution may be more suitable for a more focused venue (e.g., a workshop or conference on graphs and topology in AI) unless the authors can articulate a clearer connection to the general capabilities or limitations of LLMs or better justify the importance for LLMs to be able to perform these TDA computations.

[W2] The three “simple” tasks defined in the paper do not strongly reflect persistent homology reasoning—they could just as well appear in a general graph reasoning benchmark. It would strengthen the work to include additional tasks that more explicitly capture PH-specific concepts within this task family.

[W3] In Section 3.1, the finding that LLMs perform worse when PH-specific terminology is used seems unsurprising. This likely reflects the models’ limited familiarity with the jargon of a niche field rather than any deeper issue with topological reasoning itself. Similarly, the statement that “LLMs are capable of expressing meaningful topological reasoning only when guided by structured prompts and constrained options” may stem more from unfamiliar terminology than from fundamental reasoning limitations. If so, this insight is less significant than it initially appears.

### Questions
[Q1] Could you further justify why PH reasoning is an important or representative capability for evaluating LLMs? 

[Q2] Could you expand on the role of terminology in your results? You briefly mention this in Section 3.5, but it’s unclear whether your main experiments rely on PH-specific language. Why is it important, or interesting, for an LLM to understand this terminology rather than simply solving the task when phrased in more neutral terms? Couldn’t this familiarity be induced through an appropriate system prompt?

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
5

### Summary
This paper proposes LLM4PH, the first benchmark for evaluating large language models (LLMs)’ ability to understand and apply persistent homology (PH) on graphs. It decomposes the PH pipeline into 4 difficulty levels (10 subtasks), uses 3 sizes of synthetic graphs and 3 real molecular graph datasets (BZR, COX2, LDHFR), and assesses 9 LLMs. The work aims to bridge LLMs’ discrete graph reasoning and continuous topological abstraction, providing insights into structure-aware scientific reasoning.

### Strengths
1. Originality: Fills a gap by being the first benchmark targeting LLMs’ PH understanding, addressing the lack of topological reasoning evaluation in existing graph benchmarks.
2. Rigorous Experiments: Evaluates diverse LLMs (proprietary/open-source) with controlled graph sizes, and conducts cross-task ablations (prompt encoding, post-training) for in-depth analysis.
3. Clarity: Well-structured with clear task definitions, result tables, and appendices for reproducibility.
4. Significance: Addresses LLMs’ limitations in non-Euclidean topological reasoning, with implications for fields like molecular biology and social network analysis.

### Weaknesses
1. Lacks independent subtasks for "persistence diagram interpretation" and "topological distance reasoning"—core PH steps.
No Terminology Ablation: Fails to verify if low accuracy in "1D Simplex Counting" stems from "1-simplex" terminology (vs. "edge") via ablation.
2. Does not test prompt engineering (e.g., CoT) to fix errors like "context loss" in non-uniform filtration generation.
3. No Traditional PH Tool Comparison: Omits performance contrasts between LLMs, full traditional PH pipelines, and hybrid pipelines.
4. Lacks key works on simplicial complex identification (e.g., Hypergraph Isomorphism Computation, Reinterpreting Hypergraph Kernels: Insights Through Homomorphism Analysis).

### Questions
see weaknesses

### Soundness
4

### Presentation
3

### Contribution
4
