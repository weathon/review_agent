# Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 2, 8

## Abstract
Text embeddings from Large Language Models (LLMs) have become foundational for numerous applications. However, these models typically operate on raw text, overlooking the rich structural information, such as hyperlinks or citations, that provides crucial context in many real-world datasets. This paper introduces and systematically evaluates a new paradigm for generating structure-aware text embeddings by integrating these structural relations directly into the LLM's internal encoding process, rather than relying on traditional post-hoc aggregation. We investigate two primary in-process methods: sequential concatenation and parallel caching. Through extensive zero-shot experiments across retrieval, clustering, classification, and recommendation tasks, we demonstrate that our structure-aware approaches consistently outperform both text-only and post-hoc baselines. Our analysis reveals critical trade-offs: sequential concatenation excels with noisy, moderate-length contexts, while parallel caching scales more effectively to long, high-signal contexts but is more susceptible to distractors. To address the challenge of noisy structural data, we also introduce and validate two effective techniques: Context Distillation and Semantic Balancing. This work provides the first comprehensive analysis of in-process structure-aware encoding, offering a blueprint for building more powerful and contextually aware embedding models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies “structure-aware” text embeddings by injecting graph/relational context (e.g., linked neighbors, co-occurrence, citations) during encoding rather than aggregating representations post-hoc. Two inference-time, training-free variants are proposed: a sequential concatenation strategy (Seq) that encodes the target text with retrieved neighbors as a single sequence, and a parallel key–value strategy (Par) that supplies neighbor representations as external attention sources without serial concatenation. Extensive zero-shot evaluations across retrieval, clustering, classification, and recommendation show that structure-aware encoding generally outperforms both plain “individual” encodings and after-the-fact pooling.

### Strengths
- Clear articulation of “encoding-time structure injection” vs post-hoc aggregation, with two complementary designs (Seq vs Par) and principled discussion of trade-offs (robustness to noise, long-context behavior, and compute).
- Evaluations span multiple tasks/datasets and report consistent gains for structure-aware embeddings relative to individual encodings and standard aggregation baselines.
- Methods are training-free at inference time and thus easy to bolt onto existing embedders; simple techniques (context distillation, semantic balancing) are effective in practice.

### Weaknesses
- Positioning vs encode-time RAG/PRF is under-specified (mechanism-level overlap).
1. The Seq pathway is closely related to retrieval-augmented fusion that injects evidence during encoding/decoding, e.g., Fusion-in-Decoder (FiD) [1], REALM [2], RETRO [3], and black-box prepend methods such as REPLUG [4]. PRF-style contextual expansion with BERT encoders (BERT-QE) [5] also fuses feedback text at encoding time.
2. The Par pathway resembles dense-retrieval PRF that augments the query with additional feedback vectors/keys in parallel, e.g., ANCE-PRF [6], subsequent reproductions/enhancements [7], multi-representation PRF [8], and ColBERT-PRF/FairPRF [9].

Although this paper targets general-purpose embeddings rather than generation/reranking, the encode-time fusion mechanisms are substantially similar. A more systematic, controlled comparison is needed (same backbone, retrieval pool, neighbor budget, and fusion budget) to sharpen novelty claims.

- If any interpolation/weighting hyperparameters are tuned on evaluation splits, that inflates reported effectiveness vs deployment-realistic settings; stricter selection (validation only) or self-tuning heuristics would strengthen claims.

[1] Izacard & Grave. Leveraging Passage Retrieval with Generative Models (Fusion-in-Decoder). 2021.

[2] Guu et al. REALM: Retrieval-Augmented Language Model Pre-Training. ICML 2020.

[3] Borgeaud et al. RETRO: Improving Language Models by Retrieving from Trillions of Tokens. PMLR 2022.

[4] Shi et al. REPLUG: Retrieval-Augmented Black-Box Language Models. NAACL 2024.

[5] Zheng et al. BERT-QE: Contextualized Query Expansion for Document Re-Ranking. Findings of EMNLP 2020.

[6] Yu, Xiong, Callan. ANCE-PRF: Improving Query Representations for Dense Retrieval with Pseudo-Relevance Feedback. CIKM 2021.

[7] Li et al. Improving Query Representations for Dense Retrieval with PRF: A Reproducibility Study. ECIR 2022.

[8] Wang et al. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. arXiv 2021.

[9] Jaenich et al. ColBERT-FairPRF: Towards Fair PRF in Multiple-Representation Dense Retrieval. ECIR 2023.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is motivated by the question of how to integrate structural information with the internal knowledge of LLM encoders to improve text embedding quality. It explores structure-aware text embedding by integrating structural information directly into the LLM’s internal encoding process. Through experiments, the paper shows the advantages of the two methods, Struc-Emb-Seq and Struc-Emb-Par. The paper is well structured and easy to follow. However, the lack of theoretical support and comprehensive experiments, such as fine-tuning and few-shot settings, limits its soundness.

### Strengths
1. This paper proposes structure-aware text embedding by integrating structural information directly into the LLM’s internal encoding process.
2. Through experiments, the paper shows the advantages of the two proposed methods, Struc-Emb-Seq and Struc-Emb-Par.
3. The paper is well structured and easy to follow.

### Weaknesses
1. The experimental results do not show a large advantage for these two methods; individual embeddings also show the best performance among the experiments.
2. This paper lacks an analysis of computational comparisons.
3. The paper focuses on zero-shot experimental results, lacking experiments such as fine-tuning and few-shot, which limits its soundness.
4. The captions of figures and tables lack clear notation and summarization of the results.

### Questions
1. Can this framework be evaluated with more extensive experiments, such as fine-tuning and few-shot learning, to strengthen its soundness?
2. What is the computational difference between LLMs and structure-aware models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present an approach to integrate structural information (such as links or citations) directly into the encoding process of pre-trained language models.
They propose two versions: Struc-Emb-Seq, which concatenates the target text with related texts   and Struc-Emb-Par, which uses parallel caches to incorporate the context more efficiently.
They also introduce two improvement techniques, Context Distillation and Semantic Balancing, aimed at reducing noise and preserving the meaning of the main text.
Experiments on several datasets show moderate improvements.

### Strengths
The idea that structured texts contain useful relationships is valid. The two proposed strategies, sequential and parallel, are well defined and easy to reproduce.
The analysis is systematic, as the authors test different types of datasets and provide a clear comparison with the baselines. The writing is clear, and the paper uses standard metrics and presents consistent results.

### Weaknesses
- Integrating structural context or linked documents into LLMs is not a new idea.
- The authors propose two well-implemented, but quite straightforward, approaches for adding context to the model. Concatenating texts or reusing parallel caches is not a significant conceptual innovation, but rather a simple variation in input processing.
- Maybe I missed it, but I could not find any analysis explaining why the use of structural context works or how it interacts with semantic representation; everything remains at the implementation level.
- The improvements are present but small, showing no clear qualitative leap—only marginal gains over the baselines.

The paper is well-executed and experimentally thorough, but lacks novelty. I would describe it as incremental work: it shows that certain practical choices (integrating structure during encoding) can yield slightly better results, but it does not advance the state of the art. It neither opens new research directions nor provides a theoretical framework. The paper is not poorly done, but it is not sufficiently innovative for this venue.

### Questions
It would be helpful if the authors could further stress the conceptual novelty of their approach in relation to previous studies.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces STRUC-EMB, a framework for generating structure-aware text embeddings by integrating relational information (such as hyperlinks, citations, or co-purchase links) directly into an LLM’s encoding process. It proposes two methods—sequential concatenation and parallel KV caching—and enhances them with context distillation and semantic balancing to handle noise and preserve target semantics. Across retrieval, clustering, classification, and recommendation tasks, these in-process methods consistently outperform text-only and post-hoc aggregation baselines, showing that directly encoding structural context yields more effective and scalable embeddings.

### Strengths
- Well-written paper and very pleasant to read.
- Comprehensive and carefully designed experiments.
- Investigating structure-aware encoding is both timely and genuinely valuable.

### Weaknesses
- The core findings are somewhat expected (e.g., structural information helps, and one must balance target embeddings against noisy context), though the proposed techniques—Context Distillation and Semantic Balancing—are thoughtful and appreciated.
- I might appreciate a deeper discussion or qualitative/mechanistic analysis of how post-hoc aggregation fails to capture low-level interactions.

Minor:
- Line 200, ’structurlly’

### Questions
- Apologies if I missed this, but could you clarify the experimental configurations used for the Mean Neighbour and Weighted Mean Neighbour baselines?

### Soundness
3

### Presentation
3

### Contribution
3
