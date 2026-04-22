# Cross-modal RAG: Sub-dimensional Text-to-Image Retrieval-Augmented Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture, necessitating the integration of retrieval methods. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy—combining a sub-dimensional sparse retriever with a dense retriever—to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in the retrieval and further contributes to generation quality, while maintaining high efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Cross-modal RAG addresses a core limitation of existing text-to-image RAG systems: global image-text similarity often fails when complex queries span multiple fine-grained aspects and no single image satisfies all constraints. The paper proposes a sub-dimensional framework that decomposes both the textual query into subqueries and candidate images into aligned visual sub-dimensions. Retrieval is cast as a multi-objective problem that jointly balances lexical subquery satisfaction (sparse stage) with fine-grained semantic alignment (dense stage) to select a Pareto-optimal set of complementary images. Generation is then guided by a MLLM with subquery-aware instructions so that only the relevant visual components from each retrieved image condition the final synthesis.

### Strengths
The paper advances RAG for text-to-image generation by dual decomposition—subqueries on the textual side and sub-dimensional visual embeddings on the image side—which enables targeted alignment at a finer granularity than global CLIP-like matching. The multi-objective formulation and Pareto-optimal image selection are conceptually elegant and differ from common top-k retrieval pipelines. Compared to fine-grained retrieval works that crop regions or use patch-level alignment, the proposed adaptor derives sub-dimensional embeddings without explicit region cropping and aligns them to subqueries, which is both efficient and robust to noisy region proposals. The subquery-aware generation via MLLM prompting (“Use only [Qj] in [Ij]”) is a pragmatic and model-agnostic way to control conditioning at inference.

### Weaknesses
1. Reliance on dataset captions for the sparse stage raises questions about robustness in settings where captions are incomplete, noisy, or absent. While the dense stage helps, the algorithm filters to by sparse satisfaction, which could discard relevant images that use synonyms or paraphrases not captured by lexical matching. The approach assumes successful query decomposition—errors or ambiguities in subquery extraction (or in converting ti to core concepts Ti) can propagate to both retrieval and generation.

2. Lack of ablation study on evaluation of text-to-image generation on WikiArt, CUB, and ImageNet-LT. The authors are advised to provide more results under the settings of w/o RAG, w/o Stage 1, w/o Stage 2, w/o Stage 3, etc.

### Questions
1. What is the sensitivity of retrieval and generation quality to β and the α schedule? Beyond the theoretical bound, does a learned α improve coverage/quality for user-specific priorities?

2. How reliable is subquery decomposition across domains? Do you have error analyses showing typical failure modes (entity conflation, attribute omission), and how the dense stage compensates when decomposition is imperfect?

3. How large can the Pareto set P be in practice, and how does generation quality/latency scale with the number of in-context images? Are there diminishing returns beyond 2–3 images, and do you prune overlapping subquery coverage proactively?

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
This paper proposes Cross-modal RAG, a framework for text-to-image generation that uses sub-dimensional decomposition of queries and a hybrid retrieval strategy to handle complex prompts. It aims to improve generation by retrieving a set of complementary images and guiding an MLLM with subquery-level instructions.

### Strengths
1.The dual decomposition of queries and images into sub-dimensions is creative and distinguishes it from prior work. 

2. ​The multi-stage framework is well-explained, and the algorithm is detailed with complexity analysis. 

3. The proofs are logically structured and align well with the framework’s design.

### Weaknesses
1.While the authors mention FineRAG briefly in Related Work, no experimental results are provided to directly contrast the two methods.

2.The approach relies heavily on LLMs or query decomposition. Evaluating robustness to different decomposers would strengthen the work.

3. Minor Errors: 

	(1) Line 249, Redundant use of “by” in the sentence: “dominated by by any other image...”

	(2) Line 277, The term “cos” appears abruptly without prior definition; it should likely align with the earlier notation "sim" (cosine similarity) introduced in Section 3 for consistency.

	(3) vji is occasionally written as vj,i (Theorem 3.2), which may cause confusion.

### Questions
1. See weaknesses above.

2. The selected MLLM baselines are already highly capable models. Could the observed gains be partly attributed to the inherent knowledge of these models rather than the proposed retrieval framework? To clarify this, could the authors provide ablation results where weaker generators are integrated with Cross-modal RAG?

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
4

### Summary
This paper proposes Cross-modal RAG, a retrieval-augmented generation framework that decomposes both queries and candidate images into sub-dimensional components for subquery-aware retrieval and image generation. The method combines a sub-dimensional sparse retriever and a dense retriever under a multi-objective optimization formulation, aiming to retrieve a Pareto-optimal set of images that collectively cover all semantic aspects of a text query. Experiments on MSCOCO, Flickr30K, WikiArt, CUB, and ImageNet-LT show notable quantitative improvements.

### Strengths
1. The paper presents a clear and novel perspective by modeling retrieval as a multi-objective optimization over subqueries.
2. The hybrid sparse–dense retrieval design is technically sound and efficiently implemented.
3. The paper provides extensive experimental results and clear visualizations to support the proposed idea.
4. The Pareto-optimal formulation and subquery-aware conditioning in generation are conceptually elegant and potentially generalizable.

### Weaknesses
1. The reported BLIP-2 results on MSCOCO deviate substantially from those in the original BLIP-2 paper, raising concerns about the fairness or correctness of baseline reproduction.
2. Evaluation on retrieval benchmarks (e.g., MSCOCO, Flickr30K) is insufficient to reflect the RAG ability for generation. These datasets primarily test retrieval performance, not retrieval-augmented reasoning or compositional synthesis, which are central to RAG.
3. The design of Stage 1 and Stage 2 is very complicated but only performs the function of query decomposition, which has been widely investigated and applied in previous works. This paper lacks comparison with naive query decomposition methods, such as leveraging an LLM as the query decomposer.

### Questions
same as the weaknesses.

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
This paper proposes Cross-modal RAG, a retrieval-augmented framework for text-to-image generation. The core idea is to decompose a textual query into multiple sub-queries and retrieve corresponding image fragments in a cross-modal space through a hybrid sparse–dense retrieval mechanism. Subsequently, a multimodal large language model (MLLM) conditionally fuses these retrieved images to produce a composite visual output. The authors evaluate the approach on MS-COCO, Flickr30K, WikiArt, and ImageNet-LT, showing that it outperforms existing methods such as BLIP-2, SigLIP, and UniRAG in both fine-grained retrieval and image generation performance.

### Strengths
1. Proposes a sub-dimensional decomposition mechanism for cross-modal retrieval-augmented generation that is simple yet effective.
2. Achieves consistent improvements in both retrieval and generation.
3. The hybrid retrieval design (sparse + dense) achieves a good balance between accuracy and efficiency.

### Weaknesses
1. The paper’s novelty is not sufficiently articulated. The concept of Cross-modal RAG has appeared in prior studies (e.g., VisRet), so the authors should more clearly highlight their unique contribution—particularly the multi-dimensional decomposition mechanism for complex multimodal semantics. Comparative or visualization-based analyses (e.g., subquery–feature alignment) would help strengthen the differentiation.
2. The proposed “Pareto-optimal hybrid retrieval” remains largely heuristic and lacks theoretical analysis. The paper does not explain how the strategy guarantees or approximates Pareto optimality under given hyperparameter settings. The authors are encouraged to include sensitivity analyses of α and β or provide theoretical justification to demonstrate optimization stability and soundness.
3. The efficiency evaluation focuses mainly on parameter size but ignores the additional computational cost from multiple subquery retrievals and multi-image generation.
To ensure a fair comparison, the authors should additionally report:
• inference time and GPU memory usage under the same number of retrievals;
• baseline performance and latency when multi-retrieval is also applied.

### Questions
1. Have you analyzed the impact of α and β parameters on the balance between retrieval and generation?
2. How does the inference latency under multi-retrieval conditions compare with BLIP-2 or SigLIP?

### Soundness
3

### Presentation
3

### Contribution
3
