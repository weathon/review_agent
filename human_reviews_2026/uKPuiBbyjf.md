# Text2GraphBench: A Comprehensive Benchmark for Evaluating Text-Instructed Graph Generation with Large Language Models

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
The rise of Large Language Models (LLMs) is driving a paradigm shift in graph generation, from traditional statistical modeling to the emerging paradigm of Text-instructed Graph Generation. However, the development of this research field faces a critical bottleneck: a severe lack of benchmarks specifically designed for this new paradigm. This prevents a reliable and in-depth analysis of the capabilities of existing models. To address this issue, we introduce Text2GraphBench, a comprehensive benchmark designed to evaluate and analyze the performance of large models on this task. At the core of Text2GraphBench is a methodology for benchmark curation and evaluation centered on constraints. For dataset curation, we pioneer a ``graph-to-constraint, constraint-to-text'' generation pipeline, building a large-scale, multi-domain dataset that ensures every textual instruction corresponds to a precisely verifiable constraint. For the evaluation system, we propose a novel, constraint-based three-dimensional evaluation framework that moves beyond traditional similarity comparisons, assessing generated graphs from the perspectives of Validity, Semantic Fidelity, and Novelty in a thorough and quantifiable manner. We conduct extensive evaluations on a range of mainstream LLMs using Text2GraphBench, and our results provide the first systematic revelation of the current capabilities, strengths, and challenges of these models. We hope that Text2GraphBench will provide the community with a valuable tool to quantify model capabilities and inspire future research. Our datasets, code, and analysis results are fully open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors proposed a new benchmark named Text2GraphBench, which is used to evaluate the performance of LLM in generating a corresponding graph given constraints in the prompt. Specifically, authors collect multiple graph datasets, generate graphs that fit into each constraint type, and design corresponding QA tasks. Finally, the authors evaluate multiple existing LLMs on the proposed benchmark.

### Strengths
- The paper is well-written and easy to follow.
- The overall framework for constructing and evaluating the benchmark makes sense to me.

### Weaknesses
- I cannot see any major problem with the overall framework of the proposed benchmark. The biggest concern in my mind is how significant this benchmark is to both the LLM community and the graph community. In particular, the benchmark is used to examine the instruction-following ability of LLMs on graph generation. 
- For the LLM community, I would be willing to see a benchmark that is hard enough for current LLMs, even with enough training. However, most of the current LLMs do not train on extensive graph datasets, which means they generally fall short on graph generation, compared to other instruction-following baselines. Given this, I may want to see whether the LLMs are still struggling on the benchmark even when it was fine-tuned on enough graph datasets, to indicate an intrinsic limitation on current LLMs.
- For the graph community, it is kind of like an evaluation tool for researcher to pick the best LLM for generating their graphs. However, given the limited domain the benchmark covered, researchers can only use it as a reference if their graph distribution are similar to one of the dataset in the benchmark. 
- Could the author provide more evidence on the significance and usage of the proposed benchmark?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Text2GraphBench, a constraint-centric benchmark for text-instructed graph generation. It uses a Graph→Constraint→Text pipeline: raw graphs from multiple domains are paired with machine-verifiable constraints, then rendered into natural-language instructions via templates (plus polishing), enabling fully automated evaluation. The framework parses model outputs (adjacency/GML/DOT or executable Python) into NetworkX graphs, then computes a Constraint Pass Rate (CPR) as the core metric. Experiments evaluate several mainstream LLMs under a unified zero-shot prompt and common decoding settings (temperature, top-p, top-k), and report parsing validity and CPR across domains and constraint types. The claimed contributions are a multi-domain dataset, a three-dimensional constraint-centered evaluation, and a broad empirical study of current LLMs’ capabilities and limits on graph generation.

### Strengths
(1) Clear, verifiable evaluation core. Tying each instruction to explicit, machine-checkable constraints supports objective, reproducible scoring and reduces ambiguity compared to similarity metrics. 

(2) Multi-domain coverage with graded difficulty. The benchmark spans synthetic and real-world graphs (molecular, social, transport, IoT) and organizes tasks from simple to composite, probing capability boundaries more finely. 

(3) Reasonable experimental setup. The paper specifies zero-shot prompting and shared decoding parameters, which simplifies replication and isolates modeling differences from prompt engineering.

### Weaknesses
Issues: 

(1) Missing subgraph details. You sample subgraphs from DBLP and PEMS-BAY, but core specifications are missing. Please state the community detection algorithm, its parameters, and the target subgraph sizes. Explain how you avoid sampling bias, such as high-degree core inflation or modularity artifacts. Document the sampler, all hyper-parameters, and the full size distribution so results are reproducible.

(2) Constraint feasibility.
Constraint sets can conflict in practice. If some combinations are impossible to satisfy, the metric may penalize models unfairly. Please report the fraction of infeasible items per subset or certify feasibility. A clear feasibility check before instruction generation would resolve this concern.

(3) Instruction polishing drift.
The template → Qwen3-8B polishing step may change meaning. Numbers, ranges, and operators can flip under paraphrase. Please add a round-trip check that re-parses the polished text back into constraints and compares against the source. This verifies that semantics are preserved exactly.

(4) Potential model leakage/bias.
Qwen is used to polish the benchmark, and Qwen models are evaluated. This can introduce a stylistic advantage. Please justify this choice or provide a no-polish variant. Report the performance delta to rule out curation-model bias.

(5) Difficulty calibration not specified.
Table 1 lists Easy/Medium/Hard counts, but the mapping from constraints to levels is unclear. Please define the heuristic or scoring rule used to assign difficulty. Report per-level CSR to validate that the ladder reflects real task difficulty.

(6) Baseline coverage is narrow.
Key non-LLM graph generators are absent, such as GraphRNN, GraphVAE, and diffusion/GRAND variants. Add text-conditioned adapters or minimal controllers to include them as baselines. This will show whether LLMs are truly competitive on structure fidelity.


Suggestion

(1) Template diversity analysis.
Multiple instruction templates are defined, but results are aggregated. Please report performance by template type, e.g., declarative vs. composite. Models may overfit to simpler forms, and a per-template breakdown would reveal this.

(2) Important ablations missing.
Several ablations are necessary to understand the system. Compare polished vs. no-polish instructions. Vary the number of constraints per item. Separate output modes (text vs. code). Include few-shot and tool-use prompting to test robustness.

(3) CSR vs. CPR naming
Section 2.5.1 uses CSR, but Table 2 shows CPR. Please unify the terminology and confirm whether these are identical. Consistent naming avoids confusion in replication.

### Questions
(1) Feasibility control.
You sample 1–3 constraints per item. How do you prevent impossible or contradictory sets, such as BA requirements combined with ER p or chemistry that violates valency? Do you run a feasibility checker before instruction generation, and how effective is it?

(2) Metric definitions. What exact tests and cutoffs define small-world, scale-free, assortativity, and power-law fit? Please provide the equations, references, implementation choices, and thresholds. Precise definitions make the evaluation auditable.

(3) Subgraph sampling method.
Which community detection algorithm and parameters are used for DBLP sampling? How do you control for size, density, and modularity confounds across sampled subgraphs? A clear description will help others reproduce your datasets and compare fairly.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the lack of standardized evaluation benchmarks for Large Language Models (LLMs) in text-driven graph generation tasks by proposing Text2GraphBench, a comprehensive and scalable benchmark system. The authors construct a large-scale dataset covering multiple domains, including synthetic graphs, molecular graphs, social networks, traffic networks, and IoT security, through a "Graph-to-Constraint-to-Text" workflow. Each natural language instruction strictly corresponds to verifiable constraints. For evaluation, the paper proposes a three-dimensional assessment framework: structural correctness, semantic fidelity, and domain appropriateness, with Constraint Pass Rate (CPR) as the core metric. Through systematic experiments, the authors reveal for the first time the actual capabilities, advantages, and challenges of mainstream LLMs in this task. The dataset and code are fully open-source, aiming to advance the field.

### Strengths
1. This method introduces a multi-dimensional evaluation framework that not only focuses on the structural correctness of the generated results but also assesses semantic understanding and domain knowledge mastery, resulting in a more comprehensive and detailed evaluation.

2. The constructed dataset is rich and diverse, covering multiple application domains, with varied instructions and constraints and a reasonable difficulty level, which facilitates fine-grained capability analysis.

### Weaknesses
1. The models show significant differences in performance, but the analysis is limited: there is insufficient depth in the analysis of why some models perform extremely poorly in certain constraint types or domains.

2. Missing knowledge graph and soft engineering in evaluation.

3. Missing some graph-related references:
- GraphLLM: Boosting Graph Reasoning Ability of Large Language Model
- GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability
- InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment
- Towards Addressing Frontiers in Graph Generation
- Evaluating and Improving Graph to Text Generation with Large Language Models
- Demystifying the Power of Large Language Models in Graph Generation

### Questions
1. Does the dataset's instruction diversity adequately cover the complex needs of real-world applications? Has its naturalness been manually verified?

2. During the automatic constraint extraction and instruction generation process, is there any information loss or semantic shift? How is consistency between semantics and constraints guaranteed?

3. For models with extremely poor performance (such as Qwen3-32B), has there been an in-depth analysis of the reasons for their failure? Is it a bottleneck in model architecture, training data, or inference capability?

### Soundness
3

### Presentation
3

### Contribution
3
