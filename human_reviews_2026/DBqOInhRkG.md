# RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Retrieval-Augmented Generation (RAG) enhances recency and factuality in answers. However, existing evaluations rarely test how well these systems cope with real-world noise, conflicting between internal and external retrieved contexts, or fast-changing facts. 
We introduce Retrieval-Aware Robustness Evaluation (RARE), a unified framework and large-scale benchmark that jointly stress-tests query and document perturbations over dynamic, time-sensitive corpora. One of the central features of RARE is a knowledge-graph-driven synthesis pipeline (RARE-Get) that automatically extracts single and multi-hop relations from the customized corpus and generates multi-level question sets without manual intervention. Leveraging this pipeline, we construct a dataset (RARE-Set) spanning 527 expert-level time-sensitive finance, economics, and policy documents and 48295 questions whose distribution evolves as the underlying sources change. To quantify resilience, we formalize retrieval-conditioned robustness metrics (RARE-Met) that capture a model’s ability to remain correct or recover when queries, documents, or real-world retrieval results are systematically altered. Our findings reveal that RAG systems are unexpectedly sensitive to perturbations. Moreover, they consistently demonstrate lower robustness on multi-hop queries compared to single-hop queries across all domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Retrieval-Aware Robustness Evaluation (RARE), a comprehensive framework designed for evaluating retrieval-augmented generation systems. Without requiring manual curation, RARE automatically constructs time-sensitive benchmarks via knowledge graph extraction and traversal, and systematically assesses robustness under query, document, and real-world retrieval perturbations. Extensive experiments across financial, economic, and policy domains with over 48,000 queries demonstrate that RARE effectively reveals critical limitations in RAG systems, particularly in multi-hop and domain-specific scenarios.

### Strengths
1. The authors introduce Retrieval-Aware Robustness Evaluation (RARE), a comprehensive framework designed for evaluating retrieval-augmented generation systems.

2. Extensive experiments across financial, economic, and policy domains with over 48,000 queries demonstrate that RARE effectively reveals critical limitations in RAG systems, particularly in multi-hop and domain-specific scenarios.

3. The analysis dimensions are thorough and insightful.  The paper not only presents overall performance comparisons but also compares robustness under query perturbations, document perturbations, and real retrieval scenarios.

### Weaknesses
1.  Insufficient technical details in the knowledge graph construction process: The paper does not provide enough description of the core steps of knowledge graph construction, particularly the specific implementation of relation normalization (detailed process using E5-Mistral-7B-Instruct), the quality control mechanisms for handling extracted conflicting or incorrect triples, and how to manage entity alignment and conflict resolution during cross-document knowledge graph merging.

2.  The comparison across unique dimensions is not clearly explained.  In terms of time-sensitivity (time-sens), multi-hop questions (MH), and cross-document retrieval (Dynamic), the improvements are not significantly different from existing datasets, and the innovation in these dimensions is insufficient.

3.  Limited coverage of perturbation types: The current perturbation types mainly focus on surface-level and semantic-level changes but lack factual perturbations (e.g., inserting contradictory information into documents), structural perturbations (e.g., changing document order or format), and handling of multimodal documents (e.g., cases where tables and text are mixed in images).

4.  Practical guidance could be strengthened: The experiments identified several important phenomena (e.g., the vulnerability of multi-hop queries), but there is a lack of in-depth analysis of the underlying causes of these phenomena and no specific system improvement suggestions based on the findings.

### Questions
1. The core innovation of the paper relies on large models like GPT-4 for knowledge graph triple extraction and QA pair generation.      Considering the large scale of the benchmark datasets, what are the specific computational costs and time overheads of this automated process?      Have more cost-effective alternatives been explored?      Providing estimates and discussion on this aspect would help the community replicate and extend this work.

2. On the realism and intensity balance of perturbations: In document perturbations, "lexically similar but answerless" is achieved by directly deleting the answer sentence, which may be too extreme and easy to identify in real retrieval scenarios.      Have more natural and subtle perturbation methods been considered?      Additionally, how can it be ensured that different types of perturbations have roughly comparable "interference intensity" to avoid overly harsh or lenient evaluation of certain perturbation types?

4. On the fairness of model comparison and prompt engineering: Experiments compared various open-source and closed-source models, but different models may have different sensitivities and optimization needs for prompts.      Were the unified prompts used in the paper sufficiently optimized or validated for each evaluated model?      If not specifically optimized, could this potentially lead to an underestimation of some models’ performance?

5. On the phenomenon that "model robustness does not strictly increase with scale": The paper observes an interesting phenomenon where Qwen3-32B shows lower robustness than the smaller Qwen3-8B and 4B models, attributing it to "architecture design and training methods."      This explanation could be more specific and detailed.      Can more nuanced hypotheses be proposed?

6. On the basis for quality assurance threshold selection and data bias analysis: In the quality assurance stage, Claude 3.5 was used to score generated QA pairs, with a retention threshold set at "all dimension scores above 6."      What considerations were behind this specific threshold?      Can the score distribution of filtered-out QA pairs be provided, analyzing which dimensions mainly had deficiencies?      This would help readers understand the quality boundaries of the dataset and potential biases.

If the authors can address all of the concerns, I am willing to consider raising my rating.

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
4

### Summary
This paper introduces a framework for evaluating RAG system robustness under realistic perturbations. RARE addresses critical gaps in existing benchmarks by simultaneously examining three key dimensions: dynamics (handling changing information), query complexity (single-hop to multi-hop reasoning), and robustness measurement (quantifying performance degradation). The framework consists of RARE-Get (automated KG-driven synthesis pipeline), RARE-Set (questions spanning multiple documents and complexity levels), and RARE-Met (retrieval-conditioned robustness metrics). Evaluation reveals fragility in current RAG systems.

### Strengths
- The proposed method is highly scalable: RARE-Get uses KG-driven synthesis to automatically generate multi-hop questions without manual curation, focusing on specialized, time-sensitive corpora appropriate for real-world RAG applications.
- This paper proposes well-defined metrics: RARE-Met distinguishes memorization from retrieval-based reasoning and tests robustness across query perturbations (typos, paraphrasing) and document perturbations (lexical/answer variations). Moreover, its evaluation presents useful insights: model size doesn't guarantee robustness, multi-hop queries underperform single-hop, and domain factors significantly impact performance.

### Weaknesses
- The paper does not analyze error propagation which harms its reliability. RARE-Get chains multiple LLMs (GPT-4.1 for extraction and generation, Claude variants for filtering and evaluation), where errors compound at each stage. The paper omits failure rates, data discarded during quality checks, and how extraction errors corrupt downstream question generation.
- The paper assumes generated questions require multi-hop reasoning without validation—questions may be answerable through single-chunk retrieval despite their structure. The robustness definition is debatable: it penalizes models that follow perturbed context, but robust systems could instead detect and reject unreliable retrieved information, yielding opposite interpretations of the same behavior.

### Questions
Could you comment on the failure rate or amount of data discarded during the LLM-based quality assurance step? What percentage of extracted triplets are factually correct? For instance, sample 100-200 triplets and verify against source documents can give a better sense of the raw quality of initial LLM-based generation and how errors propagate through the pipeline.

### Soundness
2

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
3

### Summary
This paper focuses on evaluating existing Retrieval-Augmented Generation (RAG) systems. It first identifies the limitations of current evaluation frameworks and then introduces a novel approach for automatically generating complex evaluation data. Specifically, the paper presents RARE-Get, a synthesis pipeline that constructs time-sensitive evaluation data via knowledge graphs; RARE-Set, a multi-domain dataset created using this pipeline; and RARE-Met, an evaluation metric designed to assess RAG performance under query and document perturbations. Experimental results demonstrate that current RAG systems exhibit fragility under certain perturbations and that robustness does not always increase proportionally with model size.

### Strengths
- This paper identifies key limitations in existing RAG evaluation datasets and introduces a dynamic data generation pipeline that leverages knowledge graphs to automatically construct both single-hop and multi-hop questions. The proposed pipeline has strong potential value for the community, as it can generate diverse and complex datasets from unstructured data. In addition, the proposed dataset **RARE-Set** and the robustness evaluation metric contribute useful resources for assessing RAG systems.
- The paper is well-written, and the analysis is thorough, supported by extensive experiments.

### Weaknesses
Although the paper includes experiments with various LLMs, the baseline methods are limited to standard RAG setups. Recent state-of-the-art RAG variants, such as adaptive or noise-robust models, are not included, which makes it difficult to fully assess the effectiveness of the proposed benchmark and evaluation metric. Incorporating results from these stronger baselines would significantly enhance the rigor and credibility of the evaluation.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
