# KG-QUEST: Knowledge Graph–Enhanced Question Answering and Reasoning in Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Large Language Models (LLMs) achieve strong results on medical and open-domain QA but remain limited by static retrieval and parametric memory, which hinder adaptation to evolving ontologies and multi-hop reasoning. We present KG-QUEST, a framework for Knowledge Graph–Enhanced QA that grounds questions in Entity–Attribute–Value (EAV) and Entity–Relation–Entity (ERE) triples and dynamically constructs an answer-specific knowledge graph during inference. A query graph is softly matched to a global biomedical KG and expanded via hop-limited frontier search with predicate weights, synonym and inverse alignment, and negation-aware pruning to form a minimal, high-support subgraph. Phase I (KG generation) fine-tunes LLaMA 3.1 (8B) with ensemble refinement to produce ontology-aligned triples; answers are then selected by dual grounding—scoring KG paths (and optional text evidence) with hop decay and abstention—yielding explicit evidence chains. On MedQA (USMLE) and MMLU medical subsets, KG-QUEST achieves new state-of-the-art results (93.7% and 92.0% accuracy, respectively), surpassing GPT-4 and Med-PaLM 2 while maintaining verifiability. Beyond medical QA, KG-QUEST demonstrates how LLMs can not only retrieve but also construct and navigate structured knowledge graphs for complex reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main contribution of this paper is an LLM+Knowledge Graph approach that extracts triples from a question, then expands the graph around these triples in a KG to build a neighborhood (this is the "dynamic" part of the method). This dynamic graph is then used to drive answer selection by scoring paths from nodes in the graph to the answers. Experimental results show very strong performance on MMLU medical subset and MedQA.

### Strengths
+ The "dynamic discovery" stage where the idea is to expand a seed set to find a local neighborhood (and to also consider boundary detection) makes a lot of sense to me.

+ The experiments showcase some impressive results on the MMLU medical subset compared to several baselines and strong results on MedQA versus a bunch of baselines.

### Weaknesses
- The model itself is quite complex, with many steps and many design choices that are not clearly motivated nor illustrated. 

- Based on the complexity of the model, I expected to see more detailed analysis and ablations in the experiments. There are many design choices, parameters, and so forth that should really be more fully explored so we can have confidence in the method components.
 -- For example, what is the impact of the initial ERE and EAV extraction? What if there are errors ... do they propagate? How robust is this component?

- The paper has three phases, but several places mention Phase 1 without continuing on to mention the other two phases (in the abstract, top of page 2 in the introduction, maybe elsewhere). 

- There is some missing highly-relevant work -- KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques https://arxiv.org/abs/2403.05881 -- that also uses LLMS + KGs.

### Questions
See above

### Soundness
2

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
3

### Summary
The paper presents KG-QUEST, a medical QA framework that turns each question into EAV/ERE triples, matches them to a biomedical KG, and then dynamically discovers a compact, answer-specific subgraph via query-graph alignment, hop-limited frontier expansion, synonym/inverse handling, and negation-aware pruning. Answers are selected by dual grounding plus an abstention rule; the overall procedure is formalized as Algorithm 1.
On MedQA (USMLE) and MMLU medical subsets, the authors report new SOTA results arguing that the synergy of dynamic KG discovery and dual grounding improves both correctness and verifiability in ontology-heavy domains. Datasets and metrics are described, along with KG generation statistics to support scale and connectedness claims.

### Strengths
While drawing on prior KG-augmented retrieval, the paper’s end-to-end, question-constrained pipeline is cleanly specified and practically sensible for clinical QA. The mechanics give the method a reproducible algorithmic core rather than a purely prompt-level recipe.
The narrative is direct, and the emphasis on auditable evidence chains and abstention aligns with safety requirements in healthcare. Reported gains on MedQA/MMLU subsets support the claim that explicit KG grounding is especially effective where ontologies are rich.

### Weaknesses
Conceptually, the method is close to graph-augmented RAG lines (GraphRAG, Chain-of-Knowledge, ToG-2); the paper frames differences but does not provide head-to-head comparisons under matched conditions, making it hard to isolate what the proposed boundary-discovery and query-constrained expansion add beyond existing graph-guided retrieval. A tighter differential would strengthen claims.
Results focus on MedQA/MMLU medical, leaving open generalization beyond medicine. Ablations are largely component removals; the study lacks design-alternative tests and error decomposition across extraction, alignment, and expansion. The dual-grounding module’s calibration and the abstention policy’s operating characteristics are also under-analyzed.

### Questions
Can you run matched comparisons against existing methods such as GraphRAG/ToG-2/Chain-of-Knowledge on a shared corpus/KG, reporting accuracy, evidence size, and latency, to quantify the added value of query-constrained boundary discovery?

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
5

### Summary
The paper proposes KG-QUEST, a framework that dynamically generates a question-specific subgraph from a knowledge graph and uses dual-grounding (graph + text evidence) to score reasoning chains. The method consists of a fine-tuned LLaMA-3.1-8B based triple extractor (Phase I) and a query-graph driven frontier expansion with hop-limited traversal and predicate weighting (Phase II). The system then scores answer candidates via a graph-based reward and abstention mechanism, and reports strong results on medical QA benchmarks. While the overall direction is compelling and addresses an important challenge in evidence-grounded reasoning, the paper lacks sufficient analysis in several areas including the quality of KG generation, robustness under missing knowledge, and fair baseline comparisons.

### Strengths
- The integration of an LLM-based triple extractor for open-domain KG expansion enables adaptability beyond fixed KG edges.
- The proposed dual grounding of path-based and text-based evidence, combined with a hop-decay/abstention strategy, shows attention to reasoning interpretability and safety.

### Weaknesses
1. The seed triples derived via the LLaMA extractor appear to be central to performance, yet there is no standalone evaluation (precision/recall) of this component. Without such analysis, it is hard to understand how error propagation from the extractor affects downstream performance.
2. The paper assumes that relevant entities and edges exist in the underlying KG. It is unclear what happens when no similar entities are found, or when the KG has significant gaps. A robustness study under varying KG completeness would strengthen the work.
3. What is the backend LLM of KG-QUEST? LLaMA3.1-8B is only for phase I, which is to extract triplets. What is the backend LLM of the whole model?

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1
