# PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks---HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG---demonstrates that our approach consistently outperforms strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces PRISM which is an agentic retrieval system for multi-hop QA that explicitly separates precision and recall in each iteration step with Selector and Adder. The system consists of the Question Analyzer to decompose the complex query into sub-queries, the Selector to filter retrieved candidates to maximize precisions, and the Adder that revisits the discarded candidate to recover recall. The retrieved information after a few iterations is fed into the Answer Generator to give the final answer. The system is evaluated on 4 multi-hop datasets in comparison with OneR and IRCoT.

### Strengths
- The RAG system with modules are clearly introduced.

### Weaknesses
- The system design is costly and there is no budget and latency analysis with baselines. Please provide #retrievals, #LLM calls, tokens, latency per query for all methods.
- MultiHopRAG Table 1 lists PRISM as 24.74/40.64, while the surrounding text states 28.18/42.22
- The novelty is limited, as the main modules in the system have been extensively explored in prior RAG work. For example, IRCoT already covers the 'decompose, retrieve and update' approach, works as Self-RAG, QD-RAG, RAPTOR, RankRAG also have retrieve, critique/select, and generate approach. 
- The Adder largely resembles a naive recall expansion and the paper should ablation Adder (not together with Selector), and the paper should also report results of precision and recall with varying k retrievals with a fixed re-ranker.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper targets multi-hop question answering, and argues that existing retrieval methods struggle to balance precision (removing distractors) and recall (capturing all necessary evidence).
The paper proposes PRISM, an agentic retrieval framework that employs three LLM-based agents, Question Analyzer, Selector, and Adder, in an iterative loop to refine evidence through precision–recall balancing.
Across four QA benchmarks, PRISM achieves higher retrieval accuracy and downstream QA performance than strong baselines such as IRCoT and SetR.

### Strengths
1. The paper is clearly written and easy to follow, with a good presentation of the motivation and framework.

2. The core idea of separating precision- and recall-oriented retrieval through multiple agents is reasonable and intuitively makes sense for multi-hop QA.

### Weaknesses
The technical contribution is limited. The proposed framework can largely be seen as a variant of existing agentic retrieval paradigms such as ReAct or IRCoT. In essence, its three-agent loop (Analyzer–Selector–Adder) fits naturally into the standard structure where an LLM iteratively decomposes complex queries, performs retrieval, summarizes or filters relevant evidence, and then issues follow-up sub-queries for missing information. As such, the framework does not introduce a new mechanism or learning objective, but rather re-organizes established steps in a slightly more structured form.

The comparison to prior work is too limited. For instance, the paper overlooks completeness-oriented retrieval frameworks such as ARM: An Alignment-Oriented LLM-Based Retrieval Method (Chen et al., 2025), which performs a “retrieve-all-at-once” alignment between questions and structured data representations. It also cites but does not thoroughly contrast with recent iterative agentic retrieval methods (e.g., ReAct-style or similar architectures), where the main differences are not rigorously analyzed. As a result, it remains unclear whether the proposed framework represents a substantive advance over these approaches or simply reorganizes familiar retrieval-reasoning patterns within a slightly different agentic structure.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a multi-agent RAG framework comprising three key components - Question Analyzer, Selector, and Adder. The method iteratively tries to balance precision (using selector) and recall (using adder). Evaluations are presented on HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG consistent improvements over zero-shot baselines like IRCoT.

### Strengths
1.  The explicit separation between precision (Selector) and recall (Adder) is intuitive and aligns with known trade-offs in retrieval-augmented reasoning.

2. The paper is fairly well written and is easy to follow.

3. Improvements in retrieval translate to gains across multiple wikipedia-style QA datasets.

4. The ablations confirm that the Q Analyzer and Selector-Adder loop indeed contribute to improvements.

### Weaknesses
1. Limited Novelty & Prior Art not fully acknowledged: The Selector and Adder roles are similar “critic” or “augmenter” agents explored in prior work. The paper could better situate itself relative to context compression and summarization methods like RECOMP (Zhang et al., 2024), which similarly aim to prune irrelevant evidence. Overall the technical novelty of the paper is limited, conceptually the contribution lies mainly in the modular packaging of ideas existing in prior literature. Accordingly, the contributions should be toned down.

2. Missing comparisons: Key state-of-the-art agentic retrievers are missing from comparisons: CoRAG (Wang et al), R1-Searcher (Song et al), O2Searcher (Mei et al), which report stronger retrieval–generation integration with SLM finetuning. The baselines used (IRCoT, RankZephyr) are somewhat dated and primarily zero-shot. Without inclusion of finetuned systems, the claim is slightly overstated.

3. Missing implementation details: The paper briefly notes using a dense retriever based on BGE-M3, but omits critical implementation specifics such as the indexing method, retrieval depth (k), and whether all baselines share the same retriever configuration and corpus index. Without these details, it is difficult to verify that comparisons are conducted under identical retrieval settings, which weakens the empirical rigor of the results.

4. Scalability is not addressed: The method involves multiple LLM calls per query (Analyzer + multi-step Selector/Adder), which might scale poorly for large corpora or smaller models. There is no analysis of cost, latency, or performance when replacing LLMs with smaller models.

5. The authors are encouraged to test their approach on more recent multi-hop QA datasets beyond wikipedia style corpora which are already seen by LLMs during training.

6. PRISM reports P/R/F1 for its approach but does not report the same retrieval metrics for every baseline at the same retrieval depth (or at varying k). Reporting only aggregate QA EM conflates retrieval and generation effects.

7. The paper might benefit from some qualitative examples. For instance, the authors could provide qualitative example cases where Selector removed correct but weak evidence.

8. Adding sensitivity analyses for iterations N, prompt templates, and LLM backends would strengthen the paper.

### Questions
Kindly see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1
