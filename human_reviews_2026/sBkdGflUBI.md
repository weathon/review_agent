# Temporal Reasoning with Large Language Models Augmented by Evolving Knowledge Graphs

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Large language models (LLMs) demonstrate impressive capability in natural language understanding, yet they remain limited when reasoning over knowledge that evolves. A common remedy is to augment LLMs with knowledge graphs (KGs), which provide structured access to factual information. However, most existing approaches rely on a static snapshot of the KG and fail to account for temporal evolution and conflicting updates that naturally arise in real-world knowledge. To address these challenges, we present EvoReasoner, a temporal-aware multi-hop reasoning algorithm that integrates global-local entity grounding, multi-route decomposition, and time-sensitive scoring to support robust inference. Complementing this, we introduce EvoKG, a noise-resilient graph evolution module that continuously updates the KG from unstructured documents using confidence-aware contradiction handling and temporal trend tracking. We evaluate our framework on temporal QA benchmarks and a new end-to-end setting where the KG is dynamically updated from raw text. Our method consistently surpasses prompting-only and static KG-augmented baselines, and notably enables an 8B-parameter model to achieve accuracy on par with a 671B model trained seven months later. These findings underscore the necessity of unifying temporal reasoning with KG evolution to ensure LLMs remain accurate and up-to-date. Code and data are released at: anonymous.4open.science/r/TREK-434C.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles a key LLM weakness: reasoning over knowledge that changes over time. It proposes a complete system with two main parts: EvoKG, which continuously updates a knowledge graph (KG) from new text documents, and EVOREASONER, a new algorithm that performs complex, time-aware reasoning on that dynamic KG. The core idea is to combine a noise-resilient KG construction pipeline with a robust, multi-step reasoning process. A standout result is showing that their framework allows a small 8B model to match the performance of a massive 671B model on temporal questions.

### Strengths
- This is not just another reasoning algorithm. It is a full-stack solution that smartly connects the dynamic updating of a KG from raw text with the downstream reasoning task.
- The multi-route approach in EVOREASONER is a good feature. It explores several ways to answer a question, making the system more likely to find a correct answer even if some information is missing or ambiguous.
- EvoKG's method for handling contradictions is very practical. Instead of simply overwriting old facts, it uses confidence scores based on how often and how recently a fact appears. This makes the KG much more resilient to noise from real-world text.

### Weaknesses
- Potential for high latency. The reasoning process is multi-layered and involves numerous LLM calls (for planning, scoring entities, reranking paths). This could make it slow and expensive for real-time applications.
- Risk of cascading errors: The framework relies heavily on the LLM at each step. An error made by the LLM early on (e.g., in the initial planning phase) could derail the entire reasoning process with no clear way to recover.
- The global initialization step, which matches a query against all entities in the KG, might not scale well to truly massive, web-scale knowledge graphs with billions of nodes.

### Questions
1. Could you provide an analysis of the inference latency and computational cost of your method compared to simpler baselines like RAG or Plan-on-Graph?
2. How does the system handle cascading errors? For instance, what happens if the initial route planning by the LLM is flawed? Are there any self-correction mechanisms?
3. The EvoKG module's success depends on accurately extracting temporal data from text. How sensitive is the system's overall performance to noise or errors in this initial time-extraction step?

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
This paper introduces EvoReasoner, a temporal-aware multi-hop reasoning algorithm that augments large language models with dynamic, evolution-resilient knowledge graphs via the EvoKG module, enabling robust inference over changing knowledge and outperforming static KG-based and prompting-only baselines on temporal QA tasks, thereby highlighting the importance of unifying temporal reasoning and continuous KG updates for keeping LLMs accurate and current.

### Strengths
1. The paper astutely identifies current methods' limitations: the treatment of temporal knowledge as static metadata and the lack of robustness to contradictions during KG updates, and effectively addresses them.

2. The temporal-aware local reranking of neighbors ensures that the exploration is guided by both semantic relevance and temporal alignment.

3. Efficiency analysis shows the proposed method has low time complexity, enabling scalability to more complex scenarios.

### Weaknesses
1. When computing $Cost(r_j)$, "the estimated number of hops required to complete the subgoal $h_t$" is used to calculate each subgoal's traversal complexity. However, how $h_t$ is estimated lacks clear explanation.

2. While the paper provides a theoretical complexity analysis, it does not report actual runtime, memory usage, or cost for the full pipeline, making it difficult to assess the practical feasibility of deploying the system at scale.

3. The use of a simple exponential decay for recency may not be optimal, as it assumes a fixed rate of knowledge obsolescence, whereas in reality, the importance of recency can vary greatly depending on the fact type.

### Questions
See Weakness

### Soundness
4

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
This paper introduces a unified framework that pairs a temporal multi-hop reasoning algorithm (EVOREASONER) with a noise-resilient KG evolution module (EVOKG). The goal is to reason over evolving knowledge while continually updating the KG from unstructured text. Evaluations span temporal KGQA and an end-to-end setting where the KG is dynamically updated, with consistent gains over strong LLM-only, KG-enhanced, and graph-reasoning baselines. The authors further claim that the approach narrows the small–large model gap (e.g., enabling an 8B model to approach a much larger model’s accuracy after KG updates).

### Strengths
1. **Clear motivation and problem scope.** The work articulates the limitations of static-snapshot methods on temporally evolving facts and frames the need for temporal validity and ordering.

2. **Coherent system design.** The split between a temporal reasoning engine and a KG evolution component makes the pipeline modular and interpretable.

3. **Broad evaluation coverage.** Results include classical temporal KGQA benchmarks and an end-to-end “continual update → reasoning” scenario.

### Weaknesses
**W1 — Missing empirical complexity/efficiency experiments.**
The paper discusses complexity qualitatively but lacks measured runtime/throughput and memory results. For deployment feasibility, please report wall-clock latency (per query and per KG-update batch), throughput, peak memory/VRAM, and token cost (prompt vs. generation), and include scaling curves vs. |V|, |E|, document batch size, route width/depth, etc. Compare these costs with the main baselines under identical hardware and decoding settings.

**W2 — Reference formatting is inconsistent.**
Entries mix styles for venue names, arXiv identifiers, capitalization, URLs, and sometimes include non-standard fields. Please unify the bibliography to the venue’s official style (e.g., ICLR’s iclr2026_conference.bst / NeurIPS format), and ensure consistent punctuation and page ranges.

**W3 — Missing comparisons to recent methods (e.g., ROG).**
The tables do not include several recent reasoning/retrieval–reasoning coupling methods, such as ROG[1].

**Reference:**
[1] Luo L, Li Y F, Haffari G, et al. Reasoning on graphs: Faithful and interpretable large language model reasoning[C]// ICLR, 2024.

### Questions
Sensitivity: How sensitive are results to contradiction-resolution and decay hyperparameters? Provide guidance for tuning.

SOTA coverage: Can you include comparisons to ROG (and other very recent systems)? If not, please justify and consider a minimal, matched experimental slice.

Runtime & memory: What are the end-to-end numbers per query and per update batch, and their breakdown by stages (extraction, alignment, merging, reasoning)?

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
This paper presents a framework to address the inability of static LLMs to reason over evolving, real-world knowledge. The authors propose a dual-component system: EvoKG, a noise-resilient module that constructs and continuously updates a knowledge graph from unstructured text, and EvoReasoner, a multi-hop, time-aware reasoning algorithm that queries this dynamic graph. EvoKG's novelty lies in its confidence-aware contradiction handling and tracking of temporal fact validity. EvoReasoner introduces robustness through multi-route decomposition and time-sensitive path scoring. The central contribution is the tight integration of these two components, demonstrating through end-to-end experiments that this unified approach outperforms static baselines.

### Strengths
1.	The authors convincingly argue that a temporal reasoner is ineffective without a KG that is actually evolving, and an evolving KG is useless without a reasoner that can handle its temporal, noisy, and contradictory nature. The unified framework is a well-motivated and strong contribution.
2.	The use of a confidence score based on both frequency and temporal decay  is a principled and practical method for managing conflicting information extracted from diverse, unstructured sources.
3.	The multi-route decomposition strategy is an effective method for improving the robustness of LLM-driven reasoning. 
4.	The authors simulate a practical scenario by downsampling the original KG to force incompleteness, then demonstrate that EvoKG can heal this graph from raw documents, leading to significant performance gains.

### Weaknesses
1.	The complexity analysis O(n log N + m log M) only covers the search/merge step, completely ignoring the dominant cost of the LLM-based information extraction. More transparent analysis of the computational and financial cost of this continuous evolution is needed.
2.	Shallowness of "Temporal-Aware" Reasoning: The temporal reasoning in EvoReasoner, while effective, appears to be primarily a reranking and filtering mechanism rather than deep temporal logic. The system uses time-augmented embeddings (Enc(m_i, τ_i)) and scores verbalized triplets (s_triplet) to check if paths align with the query's temporal scope. This is useful, but it's unclear if the framework can handle truly complex temporal queries involving relative ordering ("before X but after Y"), overlapping intervals, or reasoning about durations. The paper provides no qualitative examples of such complex logic, and the chosen benchmarks may not be sufficient to evaluate it.
3.	In a real-world scenario, new documents will introduce not just new facts but potentially new types of entities and relations. The "Synonym Matching" (Section 4.2) is a simple normalization step, but it doesn't handle the emergence of a truly novel, valid relation type. The paper does not discuss how EvoKG would differentiate a new, useful schema concept from simple extraction noise.
4.	Does the 671B model's performance also significantly increase when grounded in the evolved KG, or does its vast internal knowledge make the external graph redundant? This result is essential for understanding the true interplay between a model's internal parametric knowledge and this system's external, dynamic knowledge.
5.	The related work coverage are insufficient, as several directly relevant studies[1][2] are not discussed.

[1] Test of time: A benchmark for evaluating llms on temporal reasoning

[2] Temporal Knowledge Question Answering via Abstract Reasoning Induction

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
