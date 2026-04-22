# UniDoc-Bench: A Unified Benchmark for Document-Centric Multimodal RAG

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Multimodal retrieval-augmented Generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented—focusing on either text or images in isolation, or simplified multimodal setup, failing to capture document-centric multimodal use cases.
In this paper, we introduce UNIDOC-BENCH, the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across 8 domains.
Our pipeline extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. 
To ensure reliability, 20% of QA pairs are validated by multiple annotators and expert adjudication.
UNIDOC-BENCH supports apples-to-apples comparison across four paradigms --- 1) text-only, 2) image-only, 3) multimodal text–image fusion and 4) multimodal joint retrieval --- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. 
Our experiments show that multimodal text–image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding–based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. 
Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This introduce UniDoc-Bench, a large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across 8 domains. It supports 4 different types of retrieval paradigms: text-only, image-only,  multimodal text–image fusion and multimodal joint retrieval. The paper further offers analysis comparing visual and textual context usage in sota multimodal LLMs.

### Strengths
1. The paper introduce a novel benchmark UniDoc-Bench which contains 1, 600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries.
2. The paper introduce a fair and reproducible evaluation framework by fixing candidate pools across modalities , and measuring retrieval effectiveness, answer faithfulness, and completeness end-to-end across different RAG systems.
3. This paper conducts a systematic comparison of text-retrieval, image-retrieval, text–image fusion, multimodal joint retrieval pipelines, analyzing which retrieval strategy performs best under different question types, evidence modalities.

### Weaknesses
1. The significant contribution that I take away from this paper vs the previous work is unified evaluation and multiple reference. Therefore, 
What are the benefits of unified evaluation?
2. How is multiple reference used  during evaluation to enhance the groundness?

### Questions
1. See the in weakness.
2. Will this paper opensource their evaluation framework?

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
2

### Summary
This paper introduces UniDoc-Bench, a benchmark for document-centric multimodal retrieval-augmented generation (RAG). The dataset includes 1,600 synthetically generated multimodal QA pairs, of which 20% are validated by multiple annotators and expert adjudication to ensure quality. The benchmark standardizes evaluation across four paradigms: (1) text-only, (2) image-only, (3) multimodal text–image fusion, and (4) multimodal joint retrieval, enabling fair, apples-to-apples comparisons. Experimental results show that multimodal text–image fusion RAG systems perform best.

### Strengths
1. Provides a systematic and unified comparison of text retrieval, image retrieval, text–image fusion, and multimodal joint retrieval pipelines under consistent settings.
2. Ensures data reliability by validating 20% of the QA pairs through multiple annotators and expert adjudication.

### Weaknesses
The QA data are synthetically generated (GPT-4.1 and Gemini-Pro-2.5), which may introduce bias toward the LLMs used. This limits the benchmark’s ability to fully assess model generalization beyond this generation and templates distribution.

### Questions
Have you compared LLM-generated QA performance against fully human-written QA pairs to confirm generalizability?

### Soundness
3

### Presentation
3

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
The paper UniDoc-Bench introduces a large-scale benchmark for multimodal retrieval-augmented generation (RAG) across text, image, and table modalities, providing standardized evaluation for both open-source and commercial models.
It demonstrates that multimodal RAG improves performance and cost efficiency compared to text-only RAG, particularly in domains rich in visual content such as finance and construction.

### Strengths
1. Covers diverse modalities (text, image, multimodal, table-required) and question types, providing a unified evaluation framework.
2. Offers granular insights into the relative strengths of text and image retrieval and the role of domain-specific content richness.
3. Highlights cost and latency trade-offs in multimodal RAG, providing actionable insights for system optimization.

### Weaknesses
1. It would be beneficial to add more SoTA models' results such as Gemini and Claude with and without retrieved instances. Also, it'd be great to perform a more detailed comparison between different SoTA models.
2. Overrepresentation of finance and construction may bias generalization to less visually complex domains.

### Questions
How would different SoTA models (e.g., GPT, Gemini, Claude) perform on this benchmark and what types of errors do these models make?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents UniDoc, a unified benchmark designed to evaluate multimodal document understanding across diverse input forms, e.g., scanned pages, forms, and charts. UniDoc aggregates and harmonizes multiple existing datasets into a standardized schema, supporting different types of tasks. The benchmark construction is illustrated, and relevant experiments are executed.

### Strengths
- The benchmark is inclusive. As shown in Table 1, UniDoc contains the tasks from multiple domains, and it has the largest number of queries and pages of documents.
- The dataset curation section is clear to some extent, and the further release and deployment seem promising.

### Weaknesses
- The paper mostly reuses and reprocesses existing datasets, offering engineering unification rather than new data collection or annotation.
- 20% generated QA pairs are validated by annotators or experts. More can increase the credibility of the benchmark.
- A clear task definition should be formally expressed along with the proposed benchmark.
- The tested baseline model scope should be enlarged to increase the credibility of the proposed benchmark. So far, only 4 main models are included in the paper. Although tables 4 and 5 involve 6 models, this is still not adequate.

### Questions
In addition to the weakness, other questions are listed
- Is it necessary to measure the difficulty of tasks when unifying different datasets?
- Is there any plan for including reasoning chain annotations to assess step-by-step interpretability?

### Soundness
2

### Presentation
2

### Contribution
2
