# LG-Bench: A Graph-Structured Evaluation Benchmark for Life Sciences

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Traditional evaluation benchmarks reduce inherently interconnected scientific knowledge in life sciences into flat lists of questions, disregarding the underlying topological structure of the knowledge. We introduce LG-Bench, the first graph-structured benchmark for life sciences, featuring over 10,000 high-quality multiple-choice questions across medicine, biology, and chemistry. Our approach constructs a weighted evaluation graph using bidirectional matching and semantic similarity algorithms, where nodes represent questions and edge weights capture their semantic relationships. Leveraging this graph topology, we design two novel evaluation metrics. The Global Coherence Score (GCS) measures a model’s consistency within semantically related neighborhoods, while Knowledge Balance Score (KBS) analyzes how model errors are distributed across the graph to reveal conceptual blind spots. LG-Bench facilitates fine-grained comparison of LLMs by surfacing differences in conceptual coherence and patterns of knowledge organization across models. Our framework shifts the evaluation paradigm from flat accuracy metrics to structure-aware analysis, offering a new lens for diagnosing and improving LLM performance in the life sciences domain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LG-Bench, that contains over 10,000 high quality multi-choice questions capturing topoplogical structure of knowledge in the domain of biology, chemistry and medicine. The authors further introduce GCS and KBS to track performance across knoweldge clusters, providing deeper insights of performance on the knowledge distribution rather than plainly reporting accuracy by treating each individual questions as flat and independent entities. Evaluation and Results demonstrate that large-sized closed-sourced models perform quite well in preserving knowlegde within a single and across different domains. Further insights reveal that fine-tuning alters the knowledge landscape by re-adjusting the weights.

### Strengths
- The paper is well written with proper flow and clear/readable diagrams.
- Experiments conducted are well-rounded and covered all bases. Interesting plots have been demonstrated highlighting the key impact of the evaluation metric. Overall the need for the proposed evaluation metric is motivated well.
- The convolution of evaluation over a knowledge graph is novel and can inspire future research to gauge models on different levels of topology.

### Weaknesses
- The weighted accuracy w(v,u) in line 269 has not been explained properly.
- The motivation behind the design choices made towards stage-2 of the benchmark construction pipeline (Graph construction) is not provided. Were there other keyword representation you tried? How does the breakdown and computing semantic similarity over the keyword beneficial? How many keywords were extracted per question and what was their granularity? Moreover, what was the reason for choosing a weighted combination of bi-directional similarity and core-similarity? It would be great if you can demonstrate some ablation studies to justify the design choices made.

### Questions
1) What speciaized LLM was used for Guided Question Generation (line 154)?
2) How were the hyper-parameters decided? Was it a theoritical choice or empirically based?
3) Can you share the prompts used for Guided Question Generation and Multi-Model Committee Assessment?
4) How many questions were passed on from Multi-Model Assessment to expert human verification? What was the portion of questions were retained from the committee check and what portion was introduce after expert verification?

### Soundness
3

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
The paper introduces LG-Bench, a novel, large-scale, graph-structured evaluation benchmark for LLMs in the life sciences. It comprises over 10,000 multiple-choice questions derived from recent scientific literature. The benchmark transforms this corpus into a weighted evaluation graph and introduces two new graph-based metrics—the Global Coherence Score (GCS) and Knowledge Balance Score (KBS), designed to measure structured understanding and conceptual balance beyond standard accuracy metrics.

### Strengths
## 1. High Novelty and Vision

- The move from flat-list evaluation to a graph-structured framework is original and timely, addressing a known limitation in LLM evaluation.

## 2. Sophisticated Data Pipeline

- The multi-stage generation and filtering pipeline—combining multiple LLMs and expert review—is impressive, though the scope of human validation remains unclear.

## 3. Potential Diagnostic Insight

- **GCS** and **KBS** are conceptually appealing and could become useful diagnostic measures if shown to align with human judgments of coherence.

### Weaknesses
## 1. Graph Construction Sensitivity and Excessive Density 

- Average degree of ~586 suggests over-connectedness, risking that GCS/KBS reflect lexical similarity rather than meaningful conceptual coherence.
- Sensitivity to hyperparameters (γ, θ, κ) is not shown.
- **Require sensitivity analysis and justification for graph density.**

## 2. Lack of Human Validation for Metrics 

- No human evaluation demonstrates that GCS/KBS correlate with expert assessments of model coherence.
- **A correlation study with expert judgment is essential.**

## 3. KBS Justification and Baseline Comparison 

- The KBS amplification factor (α = 100) is arbitrary and unvalidated.
- Missing comparisons to simpler graph-aware baselines or knowledge-graph (KG) reasoning benchmarks.
- **Include ablation and KG-based comparisons (e.g., “"A scalable llm framework for therapeutic biomarker discovery: Grounding q/a generation in knowledge graphs and literature." ICLR 2025 Workshop on Machine Learning for Genomics Explorations. 2025.”).**

## 4. Circularity, Reproducibility,

- The same or similar LLMs are used for graph construction and evaluation, risking architectural bias.
- Missing implementation details (γ, θ, κ, embeddings, prompts).
- **Include full hyperparameters, model details, responsible release plan, and a formal ethics/biosecurity statement.**

### Questions
## 1. Validation Scope

- Did all 10,000 questions receive human validation, or was this a subset?
- What proportion were verified by experts versus filtered automatically by LLMs?

## 2. Expert Agreement

- Was inter-rater agreement measured among the four PhD reviewers?
- If not, how consistent were their judgments?

## 3. Graph Density Justification

- Why is such a high connectivity (avg. degree ≈ 586) appropriate for a domain-knowledge graph?
- What happens to GCS/KBS if the threshold θ or parameter γ is adjusted to produce sparser graphs?

## 4. Metric Interpretation

- How do **GCS** and **KBS** correlate with accuracy or with human-perceived coherence?
- Is there evidence that a higher GCS reflects better reasoning rather than answer clustering?

## 5. KBS Amplification (α)

- Why was α = 100 chosen?
- How sensitive are results to this factor?
- Would normalization or unamplified variance produce comparable results?

## 6. Knowledge Graph Baseline

- How does LG-Bench differ empirically from a traditional knowledge-graph-based evaluation (e.g., using DrugBank or Hetionet relations)?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LG-Bench, a novel graph-structured benchmark for evaluating large language models in the life sciences. The authors argue that traditional "flat list" benchmarks fail to test an LLM's understanding of interconnected scientific concepts.

They construct the benchmark questions by creating questions from peer-reviewed scientific papers, which serve as the graph's nodes, and then connecting these nodes with weighted edges that represent the semantic similarity between the questions.

This graph structure is then used for evaluation. The paper introduces two new metrics that go beyond simple accuracy:

-Global Coherence Score (GCS): Measures a model's consistency by assessing whether it correctly answers clusters of related questions.

-Knowledge Balance Score (KBS): Analyzes the distribution of a model's errors to identify "conceptual blind spots" or uneven knowledge.

The authors show that these metrics can reveal insights into model performance, such as the trade-offs of domain-specific fine-tuning, that traditional accuracy scores cannot.

### Strengths
* Clear thesis and focus, as well as well-articulated findings
* As far as I am aware, their graph proposal is a novel concept for a life sciences LLM benchmark, in using graphs to identify more fine-grained nuances in LLM life sciences 
* Logical argument for testing the quality of data

### Weaknesses
* Not thorough in assessing substitute and alternative approaches
* The authors do not describe the detailed effects (quantitatively, fine-grained breakdown) of the expert human review. There can be valuable insights here for the community (for example regarding patterns in LLM hallucination in question'

### Questions
Can you study alternatives more deeply?

What is the percentage of questions that were rejected or rewritten by the human experts? What are the total number of questions generated by the AI before the human validation stage?

While this appears to be a useful, unique benchmark, it is often the follow-up questions (and follow-ups to those follow-ups etc) that more akin to real life science work. Could you extend your method to evaluate this in any way?

### Soundness
3

### Presentation
3

### Contribution
3
