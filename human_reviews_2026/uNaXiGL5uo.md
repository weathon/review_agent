# Text summarization via global structure awareness

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 2, 8, 6

## Abstract
With the explosive growth of information, the volume of long documents has surged, and the cost of processing them continues to rise, making text summarization increasingly important. Existing studies primarily focus on model enhancements and sentence-level pruning based on contextual dependencies and semantic patterns. Although some approaches leverage large language models (LLMs) for text summarization and achieve higher accuracy, they incur substantial computational costs and often overlook global structural modeling. Consequently, summarized texts may lose critical logical chains, disrupting coherence and weakening downstream task performance. To address these issues, we propose GloSA-Sum, a novel text summarization framework that performs global structural analysis of texts via topological data analysis (TDA), enabling efficient summarization while preserving semantic cores and logical dependencies. Specifically, we construct a semantic-weighted graph from sentence embeddings, where persistent homology identifies core semantics and logical structures, preserved in a ``protection pool'' as the backbone for summarization. We design a topology-guided iterative strategy, where lightweight proxy metrics approximate sentence importance to avoid repeated high-cost computations, thus preserving structural integrity while improving efficiency. To further enhance long-text processing, we propose a hierarchical strategy that integrates segment-level and global summarization. Experiments on multiple datasets demonstrate that GloSA-sum reduces redundancy while preserving semantic and logical integrity, striking a balance between accuracy and efficiency, and further benefits LLM downstream tasks by shortening contexts while retaining essential reasoning chains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GloSA-sum, a novel extractive text summarization framework that leverages Topological Data Analysis (TDA), specifically persistent homology, to preserve global semantic structures and logical dependencies in long documents. The core idea is to perform a one-time TDA computation on a semantic-weighted graph built from sentence embeddings, identifying persistent features (H0 for semantic clusters/themes, H1 for logical cycles/loops) that form a "Protected Pool" as the document's backbone. Subsequent iterative compression uses lightweight proxy metrics (topological connectivity and task relevance) to remove redundancy without recomputing TDA, ensuring efficiency. For ultra-long texts, a hierarchical strategy segments the document for parallel local summarization followed by global integration. Experiments on datasets like GovReport, ArXiv, PubMed, and CNN/DailyMail show improvements over baselines (e.g., TextRank, BART, BigBird) in ROUGE, BERTScore, QAFactEval, and human evaluations for coherence, informativeness, and conciseness. The method also enhances downstream LLM tasks by shortening contexts while retaining reasoning chains.

### Strengths
Technical Soundness:  Integrating TDA into text summarization, providing a fresh global perspective that explicitly models semantic clusters (H0) and cross-paragraph logical dependencies (H1) is novel and original. Prior TDA applications in NLP (e.g., contradiction detection, discourse coherence) are classification-focused, making this a meaningful extension to generation/compression tasks. The "Protected Pool" and proxy-based iteration cleverly address TDA's computational overhead, differentiating it from graph-based methods like TextRank or optimization-based approaches.

### Weaknesses
1- Limited Justification for H1 in Text Data: While H0 (clusters) intuitively maps to themes, H1 (loops) is less straightforward for linear text. The paper claims it captures "cross-paragraph logical dependencies," but examples (e.g., case studies) are vague.  How do loops manifest in sentences? Empirical evidence (e.g., ablation on H1 alone) is missing, and higher dimensions (H2+) are dismissed without deeper analysis.

2- Scalability Concerns: TDA is computed once, but for ultra-long docs (e.g., >10k sentences), even one-time persistent homology on high-dimensional embeddings could be prohibitive (O(n log n) graph + complex homology). The paper's max length (8,192 tokens in A.2) is modest; tests on book-length texts would strengthen claims. Hierarchical segmentation is "fixed-length or semantic," but details (e.g., how semantic segments are detected) are underspecified.

3- Baseline Gaps and Evaluation Choices: Some baselines (e.g., BERTSum, MatchSum) are absent from certain datasets (e.g., GovReport), potentially biasing comparisons. LLM baselines (A.6) rely only on human eval, no ROUGE/BERTScore, limiting objectivity. QAFactEval is useful for factuality, but its question generation could be biased toward extractive methods. No analysis of failure cases (e.g., noisy embeddings) or robustness to embedding models beyond all-mpnet-base-v2.

4- Potential Overfitting to Datasets: Gains are strongest on scientific/long docs (ArXiv, PubMed), but modest on news (CNNDM). Hyperparameters (e.g., α=0.5, λ=0.7) are tuned on GovReportm; cross-dataset generalization isn't explicitly tested.

5- Extensive Experiment is required: In the era of LLMs (decoder only), summarization is quite a simple task. Please emphasize the motivation and the comparison of traditional methods, like this study, compared with LLM-based methods.

### Questions
Please check the comments in the Weaknesses Section

### Soundness
2

### Presentation
2

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
The paper proposes GloSA-sum, a global-structure-aware framework for long document summarization that uses Topological Data Analysis (TDA) to identify and preserve the semantic and logical backbone of a document. Experiments on GovReport, ArXiv, PubMed and CNN/DailyMail show higher ROUGE-L than several extractive and long-context baselines, and the authors claim that the structure-preserving summaries help LLM downstream tasks.

### Strengths
- First work to apply TDA to summarization.

- The Protected Pool + proxy scoring design avoids repeated TDA, this enables scalability

- High QAFactEval scores suggest better factual consistency than many abstractive models.

### Weaknesses
- The paper says it is the first to bring TDA into summarization. That might be acceptable wording, but a lot of what is actually done after the TDA step looks like a graph based extractive summarizer with a protected set plus shortest-path based importance.

- The ablation study removes the Protected Pool but does not compare against alternative global-structure-aware summarizers (e.g., graph-based methods with community detection, discourse parsers, or transformer-based long-range attention proxies).

- Is the performance gain due to TDA specifically, or just the idea of preserving a global backbone? Could a non-TDA method (e.g., spectral clustering + cycle detection) achieve similar results more cheaply?

- There is no point in comparing the performance gains if the latest baseline is 6 years old. Please provide baselines from newer LLMs ( at least LLMs with <8B parameters )

### Questions
- Have you compared GloSA-sum against a variant that uses non-TDA global structure detection (e.g., Louvain clustering for H₀, dependency parsing or RST for H₁)? Is TDA truly necessary, or is the gain from the Protected Pool concept alone?

- Have you explored using GloSA-sum as a preprocessing step for LLMs in a retrieval-augmented generation (RAG) pipeline? Does it reduce hallucination?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents GloSA-sum, a new method for summarizing long documents by preserving their overall structure. Its core idea is to use Topological Data Analysis (TDA), a mathematical technique, to create a high-level map of the document's main topics and logical connections. The most important sentences that form this structural "backbone" are placed in a "Protected Pool" and are never deleted. The system then iteratively removes less important sentences from around this core. For extremely long texts, it uses a divide-and-conquer strategy. Experiments show that GloSA-sum produces more coherent summaries and is more computationally efficient than strong existing methods, especially on long and complex documents.

### Strengths
1. The paper’s primary strength is its novel use of Topological Data Analysis (TDA) to formally model a document's global structure. This allows the method to identify core semantic themes and logical connections in a principled way, moving beyond traditional local similarity graphs.
2. The framework is cleverly designed to be both highly effective and computationally efficient. The one-time TDA analysis and "Protected Pool" mechanism avoid costly repeated calculations, making the method scalable for summarizing very long documents.
3. The method is supported by comprehensive experiments against numerous strong baselines across multiple challenging datasets. The rigorous evaluation, including a detailed ablation study, provides convincing evidence that each component of the framework is essential for its success.

### Weaknesses
1. The paper posits that H1 cycles correspond to "logical loops" or "recurrent argumentative structures." While this is a compelling intuition, the connection is not explicitly demonstrated. The work would be significantly strengthened by a qualitative analysis that visualizes a few high-persistence H1 cycles from the data and shows the exact sentences that form them, explaining how they constitute a logical loop. Without this, the interpretation remains a plausible but unproven claim.
2. The entire analysis is highly dependent on the quality of the initial sentence embeddings. The paper does not investigate the model's sensitivity to different sentence encoders, making it unclear how robust the identified structures are across various embedding spaces.
3. The "Protected Pool" mechanism, while efficient, may be too rigid. It unconditionally preserves sentences based on an initial analysis, which could lead to keeping structurally important but contextually redundant information without any way to reconsider them.

### Questions
1. Regarding the interpretation of H1 features: Could you provide more examples from your experiments where H1 cycles clearly map to specific reasoning chains or logical dependencies? Is it possible that some of these cycles are artifacts of the embedding space rather than true semantic structures?
2. In the ablation study (Table 5), the result for "w/o Hierarchical" is missing. Could you run this ablation on a shorter-document dataset like CNNDM to quantify how the hierarchical approach affects ROUGE scores compared to a non-hierarchical global analysis?
3. How does the method handle documents with very flat or simple structures, where there might be few persistent H1 cycles? Does the performance rely heavily on the presence of these complex features, or does the H0 backbone suffice for good performance?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GloSA-sum, a text summarization method that applies topological data analysis (TDA) to preserve the global semantic and logical structure of long documents during compression. The approach constructs a semantic graph from sentence embeddings and uses persistent homology to identify robust topological features. Such features are collected in a "Protected Pool" that guides subsequent compression through lightweight proxy metrics, avoiding repeated expensive TDA computations. A hierarchical strategy enables scalable processing of long documents.

### Strengths
- The paper is among the first to employ topological data analysis for text summarization, explicitly modeling and preserving semantic clusters and logical dependencies.
- Tables 2 and 3 provide a thorough analysis of computational complexity and runtime, demonstrating efficiency gains from the proposed one-time Protected Pool mechanism and hierarchical pipeline. The method achieves a favorable balance between computational cost and performance, which is particularly valuable for long-document scenarios.
- The paper presents a thorough empirical validation. The downstream experiments particularly enhance the practical relevance of the approach. Additional experiments demonstrate that GloSA-sum summaries provide effective context for LLM-based downstream tasks, indicating practical relevance beyond the summarization task itself and suggesting broader applicability of the method.

### Weaknesses
1. **Unclear presentation and interpretation of main results:** The results in Table 1 demonstrate competitive performance, but do not clearly establish consistent advantages over strong baselines across all datasets and metrics. The improvements are often modest and inconsistently distributed across evaluation metrics. The paper would be strengthened by statistical significance testing to confirm that observed gains are reliable rather than artifacts of random variation. Moreover, the interpretation does not address cases where baselines perform comparably or better. A more nuanced interpretation would help readers better understand the method's contributions and limitations.

2. **Potential limitation in semantic representation through independent sentence encoding:** The paper employs sentence-level encoding where each sentence is embedded independently. This design choice raises a concern: cross-reference information, such as pronouns, anaphora, and other referential expressions, is encoded without their antecedents, potentially resulting in semantically incomplete representations. Moreover, sentence-level encoding may not fully capture document-level phenomena such as topic progression, rhetorical structure, or long-range dependencies. Since TDA operates on these independent local embeddings, the method might miss some genuine semantic relationships that require broader discourse context. This limitation warrants discussion, particularly given that the method's core claim is to preserve "global structure" through operations on what are fundamentally local representations.
An ablative analysis examining different text granularities—such as clause-level, sentence-level, and multi-sentence chunks—would provide more insightful details about the method's robustness to encoding choices. Such an analysis would also help address the problem of short and isolated sentence embeddings by revealing whether coarser or finer granularities better capture semantic structure for TDA-based analysis. Additionally, the choice of `all-mpnet-base-v2` as the sentence encoder is concerning, as this relatively modest model may not provide sufficiently rich semantic representations for a method whose effectiveness critically depends on the quality of the initial embeddings—particularly when identifying nuanced topological structures that distinguish genuine semantic relationships from spurious patterns.

3. **Incomplete empirical validation of H₁ cycles' contribution:** While H₁ cycles are described as essential for preserving logical flow, their actual contribution could be more thoroughly validated. The ablation study in Table 5 evaluates the removal of the Protected Pool, TopoScore, and hierarchical components, but does not isolate the differential impact of H₀ versus H₁ features within the Protected Pool itself. Specifically, a comparison between P = P_H₀ only versus P = P_H₀ ∪ P_H₁ would directly test whether incorporating H₁ cycles meaningfully improves performance beyond semantic clusters alone.


4. **Limited analysis of hierarchical segmentation strategy:** The hierarchical compression strategy is described as partitioning documents into segments that are processed independently; however, a systematic comparison of segmentation methods (semantic versus fixed-length versus paragraph-based) would strengthen the paper. The sensitivity of summary quality to segmentation choices and the method's robustness to documents with non-standard structure remain unexplored. Such analysis would help clarify whether the hierarchical design genuinely contributes to performance improvements.


5. **Miss related work:** Lack of pertinent literature for graph construction to retain only important sentences:
    - Graph-based Abstractive Summarization of Extracted Essential Knowledge for Low-Resource Scenario (ECAI 2023)
    - Cross-Document Distillation via Graph-based Summarization of Extracted Essential Knowledge (IEEE TASLP 2025)

### Questions
1. Could the authors describe the sentence splitting procedure in detail? Which tools or methods are used to determine sentence boundaries, and how might different splitting strategies affect the resulting topological features?
2. Have the authors analyzed whether sentences in the Protected Pool exhibit systematic patterns with respect to their position in the original document (e.g., beginning, middle, end)? Such analysis would help validate that TDA captures semantic structure rather than positional biases.
3. How are shortest paths computed in the weighted graph when calculating TopoScore (Equation 7)? Specifically, how does the algorithm handle potentially disconnected components, and how are ties broken when multiple sentences have identical scores?

### Soundness
3

### Presentation
2

### Contribution
3
