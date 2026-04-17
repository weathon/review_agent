# Cross-Lingual Multimodal Retrieval-Augmented Generation for Open Question Answering in Tamil and Yoruba

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
As large language models (LLMs) with retrieval augmented generation (RAG) gain traction in multimodal knowledge base question answering (KBQA), concerns about their transfer to low resource languages (LRLs) remain unaddressed. We introduce LR-MMQA, a benchmark evaluating multimodal cross lingual retrieval and reasoning in LRLs. Using the hardest examples from WebQA and MultimodalQA, we build a high quality LRL benchmark through LLM assisted translation, human validation, and culturally aligned rewriting that reflects native speaker phrasing (i.e. what a native speaker would naturally ask) while preserving answerability. We also present XM-RAG, a cross lingual multimodal RAG pipeline for LRLs that reaches 38.1 answer accuracy, more than 7.8 points above the next best baseline. LR-MMQA exposes major performance gaps and failure modes in current systems. Notably, all baselines perform far below top English results (WebQA 64.4 and MultimodalQA 73.48), showing that existing methods still struggle with complex tasks in LRL settings. By releasing LR-MMQA and XM-RAG, we offer a resource to evaluate and address these gaps and guide progress toward equitable multimodal KBQA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Based on the existing WebQA and MultimodalQA datasets, this paper developed the knowledge-based question answering dataset LR-MMQA for Tamil and Yoruba through difficult sample selection, filtering, and post-processing. Next, the authors designed a training-free method, XM-RAG, which directly encodes the original low-resource language query without translation. The retrieved image and text evidence are then compressed and fed into mT5 along with the query and language tags to obtain the answer. Results on LR-MMQA show that XM-RAG outperforms several other baseline models.

### Strengths
Exploring multimodal RAG methods for knowledge-intensive questions in low-resource languages ​​is significant and offers new prospects for research and application.

### Weaknesses
1. In the method comparisons of Table 2: (1) For methods such as GPT-4o that are not specifically designed for low-resource language scenarios, the questions in low-resource languages ​​should be translated into English for answering, and then the answers should be translated back to the target language to reflect the advantages of this method over translation-based methods, both in terms of performance and efficiency; (2) More open-source/closed-source general-purpose MLLMs should be included, such as Gemini-2.5 flash/pro, Claude, Qwen-2.5/3-VL, InternVL-3/3.5, etc. (3) For RAG methods, recent advanced methods such as text-only Search-R1 should also be compared.
2. Regarding method design: (1) Why not input the image caption and then combine the passage and question into an older language model mT5, instead of some of the latest LLMs such as Qwen3 or LLaMA4? (2) Why not directly input the original image (without caption process), passage and question into an MLLM, such as the Qwen-2.5/3-VL mentioned above?
3. The meanings of $d_i$, $s_i^{text}$, $v_j$, and $s_j^{img}$ in L252-254 are not explained.
4. In retrieval, the commonly used metric seems to be R@k. How are the precision, recall, and F1 used in this paper defined?

### Questions
The reference format used in this paper seems unusual. For example, "text-only methods Suri et al. (2025); Chen et al. (2022); Ling et al. (2025)" in L34 is often written as "text-only methods (Suri et al., 2025; Chen et al., 2022; Ling et al., 2025)."
I have some concerns about the method and experimental comparisons in this paper, which I have detailed in the weaknesses section. I would be happy to discuss this with the authors to further consider my final score.

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
This paper creates LR-MMQA, a new multi-modal cross-lingual knowledge-base question answering dataset by translating 718 queries from English WebQA and Multimodal QA datasets to Tamil and Yoruba (documents are still in English). They then propose a multimodal RAG baseline consisting of 1) M-CLIP encoding of the query and image; 2) cross-modal retrieval via FAISS indices; 3) cross-modal ranking based on a textual and visual similarity; 4) late fusion using BLIP-2 to generate two-sentence visual summaries and compressing high-resource language passages; 5) answer generation with mT5. The proposed method is shown to outperform text-only and monolingual RAG baselines, a closed-book GPT-4o, and RAGVL adapted with MT.

### Strengths
1. The study tackles an important and challenging problem, multi-modal cross-lingual retrieval.
2. The authors evaluates their method on two under-studied linguistically diverse low-resource languages, Tamil and Yoruba.
3. The study presents a new dataset for low-resource languages.
4. The proposed method outperforms the baselines.

### Weaknesses
1. The created dataset is created using translations. As a result it is biased towards English and Western-centric knowledge and does not reflect authentic question that speakers of Tamil and Yoruba would actually ask. There are already a lot of translation-based datasets; what the community is lacking is datasets containing queries reflecting the real-world needs of speakers of under-represented languages.
2. By only translating the queries and using English ground truth documents, the proposed dataset side-steps the real-world challenge of dealing with source documents in multiple languages, which was tackled in prior work such as XOR-QA.
3. The newly proposed RAG pipeline is only evaluated on the new dataset so it’s unclear to what extent it has been tuned for this setup and how well it generalizes to other settings. It should be evaluated on the original WebQA and MultimodalQA for comparison to the English setting as well as other multilingual multimodal QA datasets such as Kaleidoscope ([https://arxiv.org/abs/2504.07072](https://arxiv.org/abs/2504.07072)), WorldMedQA-V ([https://arxiv.org/abs/2410.12722](https://arxiv.org/abs/2410.12722)), or CulturalGround ([https://arxiv.org/abs/2508.07414](https://arxiv.org/abs/2508.07414)).
4. The ablation study only ablates a single setting, the impact of the cross-encoder reranker. Other factors such as the choice of different encoding or answer generation methods, multimodal fusion and evidence compression, and the heuristic reranking are not ablated. So it’s unclear which aspects are actually important in this setting.
5. The proposed RAG baseline consists of several components, many of which have been used in prior work. It's unclear which aspects are novel and how they compare to design choices in prior work.

### Questions
1. None of the examples shown in the Appendix include images. How important is the visual understanding component as part of the overall task?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LR-MMQA, a benchmark for cross-lingual multimodal question answering in low-resource languages. The dataset is constructed by identifying and translating the hardest question-answer pairs from WebQA and MultimodalQA into Tamil and Yoruba, two typologically and semantically distant low-resource languages. The paper further proposes XM-RAG, a cross-lingual multimodal retrieval-augmented generation pipeline, which shows improved answer accuracy compared to baseline systems on this benchmark.

### Strengths
- The benchmark targets low-resource multimodal QA, a setting that is currently understudied.

- The choice of Tamil and Yoruba introduces high linguistic diversity, which improves the benchmark’s ability to test cross-lingual robustness.

- The proposed XM-RAG pipeline demonstrates measurable improvements in answer accuracy over existing systems on this benchmark.

### Weaknesses
- Many of the techniques described in Section 4 are already well-established, so the methodological contribution is limited.


- The resulting dataset is small, especially given the multimodal cross-lingual QA setting, and would likely benefit from further expansion.


- Because the dataset is directly translated from WebQA and MultimodalQA, it inherits entity and cultural biases. Inspecting the dataset showed that the questions included content with the Point Skyhawks logo and Danish Vikings. These questions may not reflect the cultural distributions of Tamil or Yoruba user queries.


- There is no comparison to other multilingual QA datasets, which would help contextualize the contribution.

- The experiments are also limited to very few models. Expanding the work to cover models is needful.

- Some relevant citations appear missing, including the original WebQA and MultimodalQA citations, as the primary is needed.

### Questions
1. In line 173, did you intend to mention BARTScore instead of BARTScore? Citing the actual metric would be useful to the readers.

2. Section 4 is stretched across multiple pages. Could this section be condensed or made more concise to better reflect the actual contribution? Most of the bullet points there are not needed.

$~$

### **Suggestions**

CVQA (Romero et al., 2024) might be a better starting point. It includes culturally grounded entities across multiple regions and may help increase cultural relevance and diversity in your dataset. It also contains questions for Tamil that you can benchmark against.

$~$
### **References**

Romero, David, et al. *"CVQA: Culturally-diverse multilingual visual question answering benchmark."* arXiv preprint arXiv:2406.05967 (2024).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new benchmark LR-MMQA for evaluating multimodal cross lingual retrieval and reasoning in a low resource language scenario for Tamil and Yoruba. The paper also propose a possible RAG pipeline for low resource cross lingual and multimodal setting, where the  KG is in high resource language and the query is in low resource language. The main contributions of this paper are:
1. A new benchmark (translation based benchmark) to evaluate cross lingual retrieval and reasoning.
2. A cross lingual multimodal RAG pipeline for low resource languages.

### Strengths
The main strengths of the paper are as follows:
1. Human validation for filtering out flawed queries from the original WebQA and MMQA datasets.
2. Involving humans during validation of the final translated dataset will ensure quality .
3. Good idea to have a multilingual, multimodal benchmark for evaluating cross lingual retrieval and reasoning.

### Weaknesses
The main weaknesses of the paper are as follows:
1.  Machine translated data lacks cultural and social aspects of the language and it also lacks originality. 
2. Machine translation errors might creep in, reducing quality of the benchmark dataset.
3. The scale of the benchmark dataset is very small, 718 instances might be too less. 
4. Challenge query selection based on performance of just Two QA models might not be the correct approach as question challenging to these two baselines might not be challenging at all.
5. Benchmark availability for only limited languages (to be precise only  two languages)
6. In XM-RAG fastest might have its own errors, which might have propagated.
7. Limited Novelty of XM-RAG pipeline.
8. Multimodal fusion might not be required for WebQA subset of dataset, but it is presented as a general sub-module of the pipeline .
9.Performance comparison with strong baselines are missing.

### Questions
1. Performance of frontier models on text subset of LR_MMQA is surprisingly low, have you done some prompt tuning and used RAG properly to test?
2. How is the performance using sonnet model?
4. Retrieval and re-ranking performance evaluation using ranking based metrics like NDCG@10 is missing.

### Soundness
2

### Presentation
3

### Contribution
2
