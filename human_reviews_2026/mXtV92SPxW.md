# UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To this end, we introduce UniversalRAG, designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose modality-aware routing that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks of multiple modalities, showing superiority over modality-specific and unified baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces UniversalRAG, a retrieval-augmented generation framework designed to handle corpora across multiple modalities (text, image, video, table) and granularities (paragraph, document, clip, video). Instead of a unified multimodal embedding space, the authors propose a modality-aware and granularity-aware routing mechanism that dynamically determines the most suitable corpus for retrieval per query. The method is evaluated across eight datasets spanning different modalities, showing improved accuracy over unimodal or unified embedding-space baselines. Variants include both trained routers and a training-free one.

### Strengths
- The paper is clearly written, well structured, and presents the motivation, method, and results in a coherent way. Figures are well-designed and help illustrate the concept.
- The problem motivation is strong and convincing. The authors correctly identify that current universal multimodal retrieval systems have failed to truly unify retrieval across modalities. Existing unified-embedding approaches still often suffer from modality bias, where queries tend to retrieve results from the same modality due to embedding-space clustering. I agree with this diagnosis and find the proposed modality- and granularity-aware routing to be a reasonable and promising alternative to address this limitation.
- The comparison between training-based and training-free (zero-shot) routers is thorough, and the inclusion of analyses such as router accuracy, latency, and scalability is commendable.

### Weaknesses
- Although the paper’s goal is to evaluate retrieval and generation across diverse modalities, each benchmark is strictly modality-specific (e.g., all queries in a dataset are either text-, image-, or video-based). This design makes the routing problem trivial, the router effectively learns to identify which dataset corresponds to which modality, not to infer modality from ambiguous, mixed, or cross-modal queries. As a result, the claimed “universal” capability is not meaningfully tested.
- Moreover, a key expectation for a multimodal RAG system is to operate effectively when multiple modalities are present within a single corpus. However, no experiment examines this setting. Without retrieval or routing under multi-modal candidate pools (e.g., setups similar to InfoSeek[1] or E-VQA[2], where text, image, and video evidence coexist), it remains unclear whether the router performs genuine modality reasoning or simply dataset-level selection.
- The benchmark design seems inherently favor UniversalRAG. Because datasets are modality-specific, the routing decision is effectively trivial. This makes the experimental gains less meaningful, as they mainly reflect the advantage of choosing the correct unimodal retriever rather than demonstrating improved generalization or reasoning across modalities. In practice, UniversalRAG’s performance relies too much on the strength of existing modality-specific retrievers.
- The paper omits fair comparisons in terms of model size, memory footprint, and latency across modalities. UniversalRAG combines multiple modality-specific retrievers plus an additional router, which almost certainly increases the total number of parameters and inference cost. Yet, only video-corpus latency is analyzed, with no consistent comparison against other modality-specific or unified baselines. This makes the efficiency analysis incomplete and weakens the fairness of the claimed advantage
- The authors highlight out-of-domain evaluation as an indicator of generalization. However, other baselines are not explicitly evaluated under in-domain vs. out-of-domain separation, making the comparison asymmetric and inherently favorable to UniversalRAG. In the appendix (Table 11), the out-of-domain results show relatively small gains compared to baselines, suggesting that the model’s advantage diminishes outside the curated in-domain setup. In fact, generalization seems to rely heavily on GPT-4.1-based routing; without it, the trained routers (DistilBERT, T5) show clear degradation. This raises concerns about whether the claimed universality stems from the framework or from the use of a powerful external model.
- The authors state that InternVideo2 serves as the vision-specific encoder, yet Table 1 shows discrepancies between VideoRAG results and InternVideo2-based retrieval on the same datasets (VideoRAG-Wiki, VideoRAG-Synth). It is unclear whether these differences arise from implementation variance. This lack of clarity undermines confidence in the experimental reproducibility.

[1] Chen et. al., Can pre-trained vision and language models answer visual information-seeking questions?, EMNLP 2023.
[2] Mensink et. al., Encyclopedic vqa: Visual questions about detailed properties of fine-grained categories. ICCV 2023.

### Questions
- The authors argue that current multimodal retrieval suffers from imperfect alignment across modalities. Do the authors view UniversalRAG as a temporary and pragmatic workaround until better multimodal alignment models are achieved, or as a fundamentally necessary paradigm, implying that full alignment across modalities is infeasible and that routing-based retrieval will remain essential in the long term?
- The paper mentions using InternVideo2 as the vision-specific encoder, but Table 1 shows discrepancies between VideoRAG results and InternVideo2-based baselines on the same datasets. Could the authors explain this inconsistency?
- Could the authors provide the total parameter count, memory footprint, and latency of UniversalRAG (router + all retrievers + LVLM) compared to unified baselines such as GME or UniRAG? Without these numbers, fairness and scalability remain unclear.
- I believe the out-of-domain (OOD) evaluation should appear in the main paper, not only in the appendix. The current in-domain results may be biased toward in-domain datasets, so showing OOD performance side-by-side with baselines is essential for fair evaluation.
- Could the authors explain whether the current setup enforces an all-or-nothing assumption, i.e., if the router predicts the wrong modality, the model cannot access any information from that modality? In real-world scenarios, even “incorrect” modalities may contain partially relevant or redundant cues (e.g., a text caption describing a video frame). Does UniversalRAG allow partial credit or soft routing across modalities?
- When the router misclassifies a query’s modality, does the system completely fail to retrieve the correct information? In real-world scenarios, relevant information might exist in other modalities (e.g., a text description of an image).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper replaces the common practice of "retrieving in a unified embedding space" with the execution trajectory of "querying Self-Adaptation routing to modality-granularity-corpus", directly addressing a pain point in real systems (different problems require evidence from different sources and units). From the perspective of experimental coverage and analysis accuracy, the paper is of high quality. The novelty is mainly reflected in unifying "modality selection", "granularity selection", and "non-retrieval" into routing decisions, and validating the implementation of this paradigm in real settings of multi-granularity/multi-corpus.

### Strengths
1. Unifying "Modal Selection", "Granularity Selection", and "Non-Retrieval" into query-level routing decisions and systematically validating them in real-world environments of MultiModal Machine Learning/multi-granularity/multi-corpus is a powerful alternative to the existing "Unified Embedding" paradigm. Compared with the multi-modal RAG approach centered on "unified space + unified metric", emphasizing a workflow centered on "query-task requirements" is innovative at both the methodological and systematic levels.

2. Helps to promote the engineering implementation of Multimodal RAG: Compared with the unified embedding paradigm, it is easier to integrate new modalities (by extending routing logic and dedicated retrievers rather than recalibrating the shared space).

### Weaknesses
1. Route Cost and System Throughput: Under industrial-scale (>100M entries) and high concurrency, what are the total latency and throughput of route invocation + dedicated retrieval? Can the router be distilled into a smaller model or cached to further reduce online overhead?

2. Learning objective of granularity selection: Has there been an attempt to bind "retrieval granularity" with "generation quality/cost" to a joint objective (e.g., learning with metric-cost Pareto optimality as a constraint), so that the router explicitly weighs the cost during inference?

3. Evidence Aggregation and Conflict: When cross-modal evidence conflicts (image and text are inconsistent), how does the current system make a decision?

### Questions
Please refer to Weaknesses.

### Soundness
3

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
2

### Summary
The paper proposes UniversalRAG to handle diverse multi-modal knowledge corpora and multiple retrieval granularities (e.g., paragraph vs. document, clip vs. full video). The core insight is that existing multimodal RAG systems suffer from a modality gap: queries (typically textual) are biased toward retrieving items of the same modality, even when other modalities contain more relevant information. UniversalRAG introduces a modality- and granularity-aware routing mechanism that dynamically selects the most appropriate corpus (or corpora) for each query, which is training-free manner.

### Strengths
1. The paper makes a conceptually clean and timely contribution by reframing multimodal RAG as a routing problem rather than a unified embedding problem
2. The experimental design is comprehensive, constructing a multi-modal, multi-granularity benchmark suite that covers realistic query types (factoid, multi-hop, visual, temporal)

### Weaknesses
1. The performance of UniversalRAG hinges critically on the router’s accuracy. While the paper explores multiple router implementations, it does not sufficiently address failure modes in ambiguous or cross-modal queries, as shown in Table 16.
2. Although the paper includes a “Cross-GPT-4.1” variant that retrieves from multiple modalities, the majority of benchmarks are unimodal by design (e.g., NQ is text-only, WebQA is image-only). The true value of cross-modal retrieval is only demonstrated on HybridQA and WebQA subsets. A dedicated benchmark with intrinsically multimodal queries (e.g., “Compare the architectural style shown in this image with the historical description in this paragraph and the example video”) would better validate the framework’s full potential.
3.Training-free router limitations: While GPT-4.1 shows strong out-of-domain generalization, its use as a router raises concerns about cost, latency, and reproducibility (as acknowledged in the Reproducibility Statement).

### Questions
Some highlighted performance scores in Table 1 are problematic: (1) NQ F1 (47.89 of ParagraphRAG > 47.86 of UniversalRAG (Cross-GPT-4.1)); (2) HoptpotQA F1 (28.49 of UniversalRAG (Cross-GPT-4.1) > 27.56 of UniversalRAG (T5-Large).

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
1

### Summary
The paper targets the “modality gap” that arises when all items are embedded into one unified space, showing that text-typed queries bias retrieval toward text even when image or video evidence is needed.It proposes a router that first picks modality, then granularity (paragraph vs document; clip vs full video; plus a no-retrieval path), and retrieves only within the chosen corpora using modality-specific encoders.

### Strengths
1. Modality+granularity routing is well specified, including a training-based router and a training-free LLM-prompted router, and the formulation includes a clear “no retrieval” option.

2. The modality-gap diagnosis is concrete and visually supported, which justifies routing instead of naive unification.

### Weaknesses
1. The “automatic” label construction for router training maps datasets to routing targets, which risks encoding benchmark priors into the router and inflating in-domain gains.

2. The evaluation relies on specific retrievers per modality (e.g., bge-large for text and InternVideo2 for vision) without sensitivity analyses to retriever choice or corpus indexing settings.

3. Cross-modal scenarios are underrepresented in common benchmarks, and even the paper notes stronger gains where queries genuinely require multi-source aggregation.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2
