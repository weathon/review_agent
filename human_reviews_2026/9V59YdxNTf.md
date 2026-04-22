# Hierarchical Sequence Iteration for Heterogeneous Question Answering

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Retrieval-augmented generation (RAG) remains brittle on multi-step questions and heterogeneous evidence sources, trading accuracy against latency and token/tool budgets. This paper introduces **Hierarchical Sequence (HSEQ) Iteration** for **Heterogeneous Question Answering**, a unified framework that (i) linearize documents, tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structural tags, and (ii) perform structure-aware iteration to collect just-enough evidence before answer synthesis. A Head Agent provides guidance that leads retrieval, while an Iteration Agent selects and expands HSeq via structure-respecting actions (e.g., parent/child hops, table row/column neighbors, KG relations); Finally the head agent composes canonicalized evidence to generate the final answer, with an optional refinement loop to resolve detected contradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-hop, and agentic RAG baselines, alongside higher efficiency. Beyond aggregate metrics, HSEQ exhibits three key advantages: (1) a **format-agnostic unification** that enables a single policy to operate across text, tables, and KGs without per-dataset specialization; (2) **guided, budget-aware iteration** that reduces unnecessary hops, tool calls, and tokens while preserving answer quality; and (3) **evidence canonicalization for reliable QA**, improving consistency and auditability of the generated answers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on hybrid question answering systems involving heterogeneous information. 
It aims to standardize retrieval information of different structures to enable unified utilization of heterogeneous data, and also improves the retrieval efficiency with multi-agent systems.

The paper proposes the Hierarchical Sequence (HSEQ) to address the above issues:
1. The authors convert heterogeneous information (e.g., plain text, tables, knowledge graphs) into a unified hierarchical sequence format, using identifiers to locate target information across different sources (e.g., rows and columns in tables), achieving a universal data representation.
2. A multi-agent collaborative framework is proposed for retrieval, with different agents responsible for global and local tasks respectively. This ensures sufficient retrieval evidence while avoiding resource waste.
3. The paper represents evidence in a structured manner, providing traceability and enhancing faithfulness during reasoning.

The authors conducted experiments on question-answering datasets with different structures, including pure text, tables, and knowledge graphs.

### Strengths
1. This paper focuses on leveraging heterogeneous retrieved information. Through an adapter approach, it unifies information of varying structures into a hierarchical sequence representation, enabling seamless integration of heterogeneous data.

2. The paper proposes a multi-agent collaborative framework, where different agents handle macro-level global planning and micro-level detail processing respectively. This improves retrieval accuracy while avoiding resource waste.

3. This paper conducts experiments on several heterogeneous datasets, including text, tables, and knowledge graphs, to validate its claims.

### Weaknesses
1. The author uses hierarchical sequences to represent heterogeneous unified information. However, the differences between representations of institutional information, such as plain text, tables, and knowledge graphs, are significant. Can this hierarchical sequence representation, serving as middleware, provide accurate guidance for subsequent retrieval? How much error is introduced during this transformation process, and how much retrieval noise does it cause? The article does not discuss the noise and information loss brought about by this intermediate transformation process.

2. The multi-agent-guided retrieval method proposed in the article is not novel. Using multi-agents for high-level planning and low-level specific operations is already a relatively common approach in the current field of llm agents, and its contribution is limited.

3. The writing of this article is somewhat disorganized, mainly reflected in the methodology section (Section 2).
    - The structure is messy and redundant: Core concepts (e.g., the unified sequence of HSEQ, the three-module design) are repeated across subsections (2.1, 2.2, 2.3), lacking logical coherence. This forces readers to piece together the framework themselves rather than following a linear and coherent explanation.
    - Key components (e.g., the fragment format of the HSEQ-Adapter, the "structure-aware neighborhood operator" of HSEQ-I) lack concrete examples or clear definitions—there are no sample fragments for tables/knowledge graphs, nor is it explained how iterative logic (e.g., parent-child jumps, window refreshing) operates in practice

### Questions
Please see weakness section

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates hybrid question answering over heterogeneous data sources. The main goal is to establish a unified way to handle information from multiple structures—such as text, tables, and knowledge graphs—while improving retrieval performance through a multi-agent system.
The authors introduce a Hierarchical Sequence (HSEQ) framework that converts heterogeneous information into a unified representation. Each piece of data, regardless of its original form, is encoded as a hierarchical sequence with explicit identifiers (e.g., for table cells or graph nodes), enabling consistent access across sources.
To improve retrieval efficiency, a multi-agent coordination mechanism is adopted: different agents are responsible for global and local retrieval tasks. This setup aims to balance retrieval thoroughness with computational efficiency. The authors further claim that this structure-aware representation enhances reasoning faithfulness and allows traceable evidence construction.
Experiments are performed on question-answering benchmarks covering multiple modalities, including textual, tabular, and knowledge-graph-based datasets.

### Strengths
The paper tackles an important problem—how to effectively retrieve and integrate information from heterogeneous sources for QA—by putting forward an interesting idea of using a hierarchical unified representation for multiple data types (which could simplify downstream reasoning), proposing a multi-agent retrieval strategy that provides a modular way to combine global exploration with local evidence refinement, and evaluating its approach on diverse datasets covering different structural formats to offer a broad view of its applicability，and this approach aligns with the framework introduced in the paper that unifies text, tables, and knowledge graphs into a reversible hierarchical sequence and uses a Head Agent and an Iteration Agent for guided, budget-aware retrieval and reasoning.

### Weaknesses
The proposed hierarchical sequence representation may oversimplify structural differences among text, tables, and knowledge graphs. It is unclear whether this abstraction preserves sufficient semantics for accurate retrieval. The paper lacks analysis on information loss or noise introduced by this transformation.

The writing in the methodology section of this paper is somewhat confusing. Could the authors clarify how each module of the proposed method interacts and operates? Providing some examples would be helpful, as the current presentation makes comprehension difficult.

The overall novelty of this paper is relatively limited. Neither the unified retrieval intermediary nor the multi-agent collaborative RAG workflow demonstrates significant methodological novelty or offers new insights.

### Questions
I have no further questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a unified framework for question answering (QA) that linearizes heterogeneous knowledge sources (text, tables, and knowledge graphs (KGs)) into a single, reversible interface. It enables structure-aware, budgeted evidence collection before synthesizing an answer. Experiments on multiple benchmarks demonstrate improved performance over several baselines.

### Strengths
* The paper is well structured.
* Ablation studies are reported, demonstrating the contribution of each component in the proposed framework.

### Weaknesses
* Unfair comparison with prior work: The paper does not employ the same base model when comparing with the existing works. For example, the TAT-LLM baseline [1] in Table 2 is based on Llama 2, while the closest base model used in this paper is Llama 3.1, and the base models of the proposed framework in Table 2 are not from the Llama series. This makes it difficult to attribute performance gains solely to the proposed framework rather than differences in underlying model capacity.
* Limited coverage of heterogeneous QA challenges in the chosen datasets: The experimental evaluation does not fully capture the diversity of heterogeneous QA. In particular, MetaQA is the only benchmark involving KGs, and it does not combine KGs with other knowledge sources. Incorporating more comprehensive benchmarks (e.g., CRAG [2]) could strengthen the motivation and empirical justification for the proposed approach. 
* Lack of clarity in experimental details and results: Please see the questions below.

[1] TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data.

[2] CRAG -- Comprehensive RAG Benchmark.

### Questions
* Why the results in Table 2 and Table 3 are inconsistent (e.g., the F1 of HybridQA for HSEQ (best))?
* Why some metrics are not reported (i.e., those dashes) in Table 2 and Table 5?

### Soundness
2

### Presentation
2

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
The paper introduces Hierarchical Sequence (HSEQ) Iteration, a unified, format-agnostic framework for multi-step, multi-source QA. It linearizes heterogeneous sources into a hierarchical sequence and runs budget-aware iterative retrieval with a planning/answering head. Results show solid accuracy and improved efficiency on several multi-hop QA benchmarks.

### Strengths
1. Unified interface across text, tables, and KG. Consistent control and auditing.

2. Competitive or SOTA accuracy on diverse multi-hop datasets.

3. Clear efficiency gains vs. graph-heavy pipelines on comparable quality.

### Weaknesses
1. **Complexity/Cost issue.** The iteration processing increases the complexity and cost of learning.

2. **The unified interface, which is the core contribution of this paper, lacks novelty.** Positioning vs. prior unified/structural RAG is not fully convincing. The authors should clarify what is technically new and why it matters.

3. Lack of ablation studies on the multi-agent design.

### Questions
1. Could the author analyze the efficiency trade-off between LLM-only and Graph ToG?

### Soundness
2

### Presentation
2

### Contribution
1
