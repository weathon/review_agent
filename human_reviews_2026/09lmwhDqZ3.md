# Automated Formalization via Conceptual Retrieval-Augmented LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Interactive theorem provers (ITPs) require manual formalization, which is labor-intensive and demands expert knowledge. While automated formalization offers a potential solution, it faces two major challenges: model hallucination (e.g., undefined predicates, symbol misuse, and version incompatibility) and the semantic gap caused by ambiguous or missing premises in natural language descriptions. To address these issues, we propose CRAMF, a Concept-driven Retrieval-Augmented Mathematical Formalization framework. CRAMF enhances LLM-based autoformalization by retrieving formal definitions of core mathematical concepts, providing contextual grounding during code generation. However, applying retrieval-augmented generation (RAG) in this setting is non-trivial due to the lack of structured knowledge bases, the polymorphic nature of mathematical concepts, and the high precision required in formal retrieval. We introduce a framework for automatically constructing a concept-definition knowledge base from Mathlib4, the standard mathematical library for the Lean 4 theorem prover, indexing over 26,000 formal definitions and 1,000+ core mathematical concepts. To address conceptual polymorphism, we propose contextual query augmentation with domain- and application-level signals. In addition, we design a dual-channel hybrid retrieval strategy with reranking to ensure accurate and relevant definition retrieval. Experiments on miniF2F, ProofNet, and our newly proposed AdvancedMath benchmark show that CRAMF can be seamlessly integrated into LLM-based autoformalizers, yielding consistent improvements in translation accuracy—achieving up to 62.1% and an average of 29.9% relative improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the task of automatic formalization in theorem proving, which currently faces two major challenges: model hallucination and the semantic gap caused by ambiguous or missing premises in natural language descriptions. To address these issues, the authors propose a framework called CRAMF (Concept-driven Retrieval-Augmented Mathematical Formalization). The framework consists of three main components: (1) construction of a concept definition knowledge base, (2) mathematical concept extraction, and (3) definition retrieval. Experimental results demonstrate that CRAMF can effectively extract information useful for formalization, thereby improving overall accuracy.

### Strengths
The paper is very well-written, easy to follow and well-organized.

This paper introduces a framework for automatically constructing a concept-definition knowledge base based on Mathlib4, addressing the problem of a lack of structured knowledge bases in the automated formalization domain.

This paper introduces the CRAMF-Guided Autoformalization process, which first extracts math concepts from question and then retrieve definitions from pre-defined knowledges. Then, based on the informal statement and definition, realize autoformalization, thus improving the accuracy.

Experiments on different datasets show the effectiveness of the CRAMF framework. Through the ablation study, it also shows the reasonability of each module in this framework.

### Weaknesses
Although the CRAMF method substantially improves both retrieval accuracy and auto-formalization performance, I still have some concerns. 

The paper uses MathBERT for semantic encoding, but it is not clear whether the authors have compared it with other encoding models that might achieve better performance. 

In addition, regarding the baselines, there already exist several retrieval approaches for Lean based on natural language, such as LeanSearch and LeanExplore. It would strengthen the paper if comparisons with these methods were included.

### Questions
Please refer to the Weakness section.

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
This paper proposes CRAMF, a concept-driven retrieval-augmented framework for automated statement formalization in Lean 4. It builds a concept–definition knowledge base from Mathlib4 and uses hybrid retrieval with reranking to provide context for LLMs. Experiments on miniF2F, ProofNet, and a new AdvancedMath dataset show improved compilation and formalization accuracy across models.

### Strengths
- The paper applies RAG to statement formalization tasks, and the retrieved contextual information effectively improves model performance.
- Code and benchmark data are provided in supplementary material, supporting reproducibility.

### Weaknesses
- Lack of novelty. It is already well established that RAG improves performance on knowledge-intensive tasks, and similar ideas have been explored in auto theorem proving and auto formalization in Lean, e.g., ReProver [1], LEGO Prover [2], and recently Hilbert [3]. It is unsurprising that RAG also works in statement formalization. Moreover, key components such as reverse translation of Mathlib4 entities, query enhancement, and dual-pathway hybrid retrieval have already been used in Mathlib retrieval systems like LeanSearch [4] and LeanExplorer [5].
- The range of models evaluated is limited. More recent and stronger models such as Gemini-2.5-pro, GPT-5, Claude-4.1, and Goedel-Formalizer-V2 (8B/32B) should be included for comparison. Their inclusion would strengthen the experimental section. It should also be noted that Kimina-7B is trained on miniF2F and ProofNet, making comparisons on these benchmarks less fair.
- Formalization accuracy is evaluated only through automated checks, without human expert verification, which weakens the reliability of the results.
- More explanation is needed for the AdvancedMath dataset. Although it is included in the supplementary material, its motivation and basic characteristics should be summarized in the main text. It seems all problems come from a single source and focus on calculus, which should be clarified.

[1] Yang, Kaiyu, et al. "Leandojo: Theorem proving with retrieval-augmented language models." Advances in Neural Information Processing Systems 36 (2023): 21573-21612.

[2] Wang, Haiming, et al. "Lego-prover: Neural theorem proving with growing libraries." arXiv preprint arXiv:2310.00656 (2023).

[3] Varambally, Sumanth, et al. "Hilbert: Recursively Building Formal Proofs with Informal Reasoning." arXiv preprint arXiv:2509.22819 (2025).

[4] Gao, Guoxiong, et al. "A semantic search engine for Mathlib4." arXiv preprint arXiv:2403.13310 (2024).

[5] Asher, Justin. "LeanExplore: A search engine for Lean 4 declarations." arXiv preprint arXiv:2506.11085 (2025).

### Questions
- Does “Formalization Accuracy Rate@10” in Table 2 refer to the probability that at least one of the ten outputs passes the semantic consistency check, or to the proportion of successful outputs among the ten? It would also be helpful to clarify whether the metric is calculated on all runs or only those that pass compilation.
- The notion of “concept” is somewhat vague. What exactly is its relationship with Mathlib definitions, classes, or structures? Is it a one-to-one mapping, or a more abstract layer? How do “concepts” influence the retrieval process?

### Soundness
3

### Presentation
2

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
This paper presents CRAMF, a retrieval-augmented framework that enhances LLM-based automated formalization for theorem proving in Lean 4. CRAMF builds a structured concept-definition knowledge base from Mathlib4 (over 26,000 definitions and 1,000 core concepts) and retrieves relevant definitions to reduce hallucinations and semantic gaps during autoformalization. Experiments on miniF2F, ProofNet, and a new dataset show consistent improvements, with up to 62.1% accuracy gain.

### Strengths
1. The paper precisely identifies two major obstacles in automated formalization—model hallucination and the semantic gap arising from ambiguous or incomplete natural language premises.
2. This paper introduces a concept-level retrieval-augmented generation (RAG) mechanism and builds a concept–definition knowledge base from Mathlib4.
3. This paper demonstrates through experiments on miniF2F, ProofNet, and the newly proposed AdvancedMath benchmark that CRAMF can be seamlessly integrated into LLM-based autoformalization pipelines, consistently enhancing translation accuracy

### Weaknesses
1. The paper is primarily system-oriented and lacks formal justification or complexity analysis of why the hybrid retrieval pipeline improves semantic grounding. For example, the mapping from informal expressions to formal concepts could benefit from a more rigorous evaluation of ontology precision and recall.
2. The experimental comparison focuses mainly on RAG-style retrievers (e.g., BM25, HyDE), without including other autoformalization-specific retrieval systems
3. While the retrieval strategy is well adapted to formal reasoning, its methodological foundation largely follows existing RAG frameworks. The paper’s novelty lies primarily in domain adaptation to mathematical formalization rather than introducing fundamentally new retrieval theory or mechanisms.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

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
This paper proposes CRAMF, a retrieval-augmented framework that enhances LLM based automated formalization in theorem provers such as Lean 4. 
The method builds a structured concept-definition knowledge base from Mathlib4 by aligning formal definitions with natural language expressions through reverse translation, concept extraction, and embedding via MathBERT. CRAMF retrieves relevant formal definitions for a given theorem through a dual pathway hybrid retrieval.

### Strengths
The motivation of model hallucination and conceptual polymorphism is sound, and the solution through the integration of retrieval and concept-level grounding is persuasive.
The ontology-based KB design is novel in the formal reasoning domain and technically sound
Experiments are comprehensive, covering multiple datasets (miniF2F, ProofNet), models (GPT-4o, DeepSeek-V3, Herald-7B), and ablations, and the newly released AdvancedMath benchmark adds value to the community.
There are consistent gains across heterogeneous base models

### Weaknesses
1. The conceptual extraction and rewriting modules rely heavily on opaque LLM prompting, I'm curious about the reproducibility and generalization.
2. comparison to premise/statement retrieval specialized for Lean is missing, which makes it hard to situate CRAMF against strongest formal-specific retrievers.

### Questions
1. could the authors report latency and context budget
2. How sensitive are results to KB noise?

### Soundness
3

### Presentation
3

### Contribution
3
