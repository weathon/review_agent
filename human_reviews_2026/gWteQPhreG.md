# RASRAG: A DOMAIN-SPECIFIC RAG FRAMEWORK AND BENCHMARK FOR ROBOTIC-ASSISTED SURGERY

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Robot-assisted surgery (RAS) has significantly improved patient outcomes by reducing blood loss, shortening hospital stays, and accelerating recovery. Despite these benefits, the widespread adoption of RAS has been slowed by a shortage of trained robotic surgeons and limited access to robotic systems. One of the major limitations is access to academic materials and expertise in this domain, which are
mostly limited to private company programs or a few textbooks. In this regard, foundation models and large language models (LLMs) have been shown to excel in both information retrieval and knowledge synthesis. However, none have been specifically adapted to the complexities of the RAS domain. To address this gap, we introduce RASRAG, a RankLLaMA-based Tree Retrieval-Augmented Generation framework that leverages a hierarchical structure derived from the source textbook. Our contributions are: (1) a novel tree-based RAG architecture in which RankLLaMA jointly performs agentic exploration and reranking along the hierarchy (“forest of knowledge”), yielding more relevant retrieval than embedding only baselines, fine-tuned models, and alternative RAG methods; (2) a publicly available, first-of-its-kind question–answer benchmark curated by seven surgeons and two physicians, reflecting real-world RAS clinical inquiries; and (3) clinically grounded evaluation protocol, including blind grading of both model and human answers by surgeons and RAG-specific measures of retrieval and answer quality. RASRAG with significantly smaller models matches or outperforms state-of-the-art LLMs, fine-tuned LLMs, and existing RAG architectures in terms of precision and relevance for domain-specific tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents RASRAG, a novel domain-specific RAG framework designed to address the knowledge acquisition barriers in Robot-Assisted Surgery. Unlike traditional vector retrieval methods, RASRAG constructs authoritative RAS textbooks into a hierarchical knowledge tree and employs RankLLaMA for exploration and reranking. The work contributes the first publicly available RAS Q&A benchmark curated by a surgical team and establishes a rigorous clinical evaluation protocol. Experimental results demonstrate that RASRAG significantly outperforms conventional RAG approaches, state-of-the-art LLMs, and human expert responses across both automated metrics and blinded evaluations by independent surgeons.

### Strengths
- The paper introduces a novel tree-based RAG architecture that employs RankLLaMA for agentic exploration and reranking within a hierarchical knowledge structure.
- The work creates the first question-answering benchmark for RAS. This benchmark is curated by a seven-member clinical expert team including five surgeons.
- RASRAG demonstrates retrieval effectiveness largely independent of model size. Even smaller parameter models (e.g., Qwen2.5-1.5B) in RASRAG achieve retrieval performance comparable to larger models.

### Weaknesses
- The benchmark represents a core contribution, yet the paper inadequately describes its construction process. Key methodological details are absent, including: question generation procedures ; sources of "standard answers" (expert-authored vs. textbook excerpts?); and quality control and validation workflows.
- The core evaluation is conducted on a benchmark derived from the same textbook used to construct the knowledge base. While the paper includes a small-scale test on a second textbook, the framework's robustness across broader knowledge bases remains insufficiently validated.
- The work is highly specialized in robot-assisted surgery. The core architecture relies on a hierarchical tree structure derived from a specific textbook. Despite authors' claims of methodological generalizability, the paper provides no evidence of scalability or broader applicability.

### Questions
See Weakness

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
This paper introduces RASRAG, a domain-specific Retrieval-Augmented Generation (RAG) framework designed for Robotic-Assisted Surgery (RAS). It integrates a tree-structured retrieval system based on a surgical textbook and leverages RankLLaMA for semantic reranking at each node. The framework aims to enhance precision and contextual accuracy in surgical knowledge retrieval. The authors also propose the first RAS-specific QA benchmark, curated by surgeons and physicians, with over 300 questions reflecting real-world clinical scenarios. Extensive evaluations—including RAGAS, NVIDIA Answer Accuracy, and expert surgeon grading—demonstrate that RASRAG outperforms traditional RAG methods, fine-tuned models, and general-purpose LLMs in both factual accuracy and clinical relevance.

### Strengths
**Novel Domain-Specific Architecture**:
The paper effectively adapts the RAG framework to a highly specialized medical context, introducing a hierarchical “tree-of-knowledge” retrieval system that mirrors clinical reasoning structures. This design improves interpretability and retrieval precision.

**High-Quality Benchmark Creation**:
The curated QA dataset, built by surgeons and doctors, is a major contribution. It provides a reliable foundation for evaluating medical QA systems, addressing the lack of standardized evaluation in RAS.

**Comprehensive Evaluation**:
The authors conduct a multi-angle evaluation using both automated and human assessments. The inclusion of surgeon-based grading lends strong credibility and clinical grounding to the results.

**Strong Empirical Performance**:
Across all metrics, RASRAG consistently outperforms both open and proprietary baselines (e.g., GPT-4o, GPT-5), demonstrating that architecture and retrieval quality can rival model scale.

### Weaknesses
**Poor presentation**:
The reviewer feels uncomfortable that there is no Introduction section at the beginning and conjectures that the presentation quality seems to be quite below the expectations of the ICLR conference.

**Limited Generalization Beyond Textbook Sources**:
The framework heavily depends on a single structured textbook as its knowledge base. While a second book test is mentioned, broader generalization to heterogeneous or unstructured data (e.g., surgical notes, videos) is not explored.

**Computational Overhead**:
The use of RankLLaMA for multi-stage reranking introduces a latency of ~15 seconds per query. Although acceptable for research, this may hinder real-time clinical deployment.

**Lack of Comparison with More Recent Agentic or Planning-Based RAGs**:
The study does not deeply compare against modern multi-hop or agentic retrieval approaches (e.g., Tree-of-Thought RAG, planner-verifier pipelines), which could contextualize RASRAG’s innovation more sharply.

**Evaluation Bias Toward Structured QA**:
The benchmark and evaluations focus on structured factual questions. Open-ended, reasoning-intensive queries (e.g., decision-making or surgical risk prediction) remain underrepresented.

### Questions
**Scalability and Adaptation**:
How does RASRAG handle updates or integration of new medical knowledge, such as new surgical techniques or guidelines? Would retraining or structural expansion be required?

**Multimodal Extension**:
Since RAS inherently involves visual data (e.g., endoscopic imagery), could this hierarchical retrieval method be extended to incorporate multimodal (text + image/video) sources?

**Clinical Validation Path**:
Beyond expert grading, are there plans to test RASRAG’s utility in real surgical training or decision-support settings, potentially measuring time saved or error reduction?

**Model Transparency and Trust**:
Given that the framework emphasizes “traceability,” how effectively does RASRAG allow surgeons to verify retrieved evidence? Could future versions integrate explainable retrieval pathways?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes RASRAG, a domain-specific Retrieval-Augmented Generation (RAG) framework for Robotic-Assisted Surgery (RAS). The method operates by structuring a source RAS textbook into a hierarchical knowledge. It then employs a RankLLaMA-based model to perform semantic reranking and navigation through this hierarchy, moving from high-level procedures (BTUs) down to specific text chunks (STUs) . A key contribution is the introduction of a new, 305-pair question-answer (QA) benchmark curated by a team of seven clinicians. The framework's performance is assessed using automated metrics (RAGAS, NVIDIA Answer Accuracy) and a blind evaluation conducted by three independent surgeons.

### Strengths
- The paper focuses on a clear and important real-world problem domain: Robotic-Assisted Surgery (RAS). This field faces distinct challenges, including a shortage of trained surgeons , barriers to training, and limited access to specialized academic materials.
- The creation and release of first-of-its-kind QA benchmark curated by clinical experts (surgeons and physicians) is a valuable contribution, providing a new resource for future research in this area.
- The explanation of the current status of RAS and its critical challenges is reasonable and supports the understanding of the RAS environment and the intent of the framework. 
- The evaluation design is comprehensive, incorporating automated RAG metrics, clinical answer accuracy metrics, and a blind human expert evaluation, which represents a robust approach to validation.

### Weaknesses
### Limited Methodological Novelty and Mismatch with Domain
The core method, a hierarchical search through a structured corpus is fundamentally just a structured search over a single textbook's table of contents. The validation for this (Appendix A.1) demonstrates properties of a well-organized textbook, not unique properties of the *RAS domain*. The authors themselves concede the method's generality ("This methodology could generalize well beyond RAS"), which undermines the central claim of domain-specific innovation.

### Under-described Benchmark
A primary contribution, the 305-pair QA benchmark, is presented with insufficient detail. Section 2 merely states *who* created it (7 clinicians) and *what* it is (305 pairs). Critical information regarding the protocol for question generation, the quality assurance process, answer curation process. This lack of transparency makes it difficult to assess the benchmark's quality or reproducibility.

### Worries of overfitting to the retrieval corpus
The framework relies on heuristic, rule-based procedures (e.g., select definite and candidate passages). Because the approach was tuned to the specific retrieval corpus, its evaluation primarily demonstrates properties of that corpus’s organization (and the chosen search heuristics) rather than unique features of the RAS domain. The resulting complexity might introduces extra engineering burden. While the complexity of the RAS domain is understandable, the heuristic-based approach may limit the method’s extensibility; therefore, the paper should provide a stronger justification to address this concern.

### Needs more baseline 
The authors rely on a latency-heavy tree-search structure to achieve gains, but do not sufficiently evaluate stronger or refined similarity-based baselines (e.g., Qwen-based retriever, re-ranking with lightweight cross-encoders, iterative RAG loops, or search-agent framework for complicate queries). Though the authors includes a few variations in table 2 (MedGraph, PaperQA), I wonder whether it is sufficiently represent the potential of existing RAG researches (and also think it should be included in table 1.)

### Questions
### Chapter-level independence as a specific property of RAS
The "Chapter-level conditional independence" seems to be the main justification for the hierarchical tree structure. Can you elaborate on why this is a specific property of RAS knowledge, rather than a general property of any well-structured textbook? How would this method perform on a non-hierarchical corpus, such as 10,000 individual surgical case reports?

### Regarding the benchmark
What was the detailed protocol given to the 7 clinicians for generating questions and answers? What quality control measures were in place to ensure the answers were correct, consistent, and comprehensive before using them as ground truth?

### Correlation with general performance of models
In table 1, unlike my expectation, large-scale models (including close-sourced) have relatively lower performance (context precision, context recall) tendency than smaller open-source models. It would be helpful for me to get a justification of this.

### Latency-Performance Tradeoff
The work needs a more rigorous, quantitative comparison that measures both quality gains and latency/compute costs. Could the authors limit the number of search call (or iterations) and evaluate the performance?



### paper error
In table 1, the performance of context precision is wrongly highlighted. the best performance is Qwen2.5-1.5B-Instruct (0.8918), not MedGemma (0.8829).

### Soundness
3

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
4

### Summary
RASRAG is introduced as a domain-specialized retrieval-augmented generation framework aimed at improving medical report generation and question-answering in a specific clinical domain. The method builds a hierarchical “forest of knowledge” from a key domain textbook, allowing an LLM agent to iteratively explore and rerank relevant sections, much like an expert searching through a textbook. 
The authors also contribute a new expert-curated benchmark of question–answer pairs reflecting real clinical queries, along with an evaluation protocol.

### Strengths
The paper tackles a well-identified gap by focusing on a specialized medical domain where general LLMs underperform. The motivation is explained with real-world context (e.g. limited access to expert knowledge in the domain), making the case that a domain-specific model is needed and valuable.
The paper contributes a new expert-curated QA benchmark for the domain, which is a valuable resource for the community.

### Weaknesses
1) The approach is tailored to a specific domain and relies on a structured hierarchy for retrieval. This dependence means that applying RASRAG to a different domain would require a similarly well-structured knowledge source. If the domain knowledge is not organized as this, the performance may degrade.  
2) While the results are strong, the paper could benefit from deeper ablation studies or analysis of each component in the pipeline.   
3) The custom QA benchmark, while valuable, is relatively small in scale (on the order of a few hundred expert-curated questions). This raises a concern that the evaluation, though high quality, might not cover the full diversity of real-world queries.

### Questions
please address the concerns in weakness sections.

### Soundness
2

### Presentation
2

### Contribution
2
