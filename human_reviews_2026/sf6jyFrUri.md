# Beyond Chunking: Efficient Global Pooling for Holistic Long-Document Representation

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Effectively representing long documents is a persistent challenge in natural language processing, as foundational encoders are constrained by limited context windows. Prevailing methods like chunking create fragmented representations that sever long-range dependencies and lose crucial global context, hindering downstream task performance. To overcome this, we introduce **Spectral Attention Token Pooling (SATPool)**, a novel, encoder-agnostic module that generates a single, holistic vector for a document of any length. SATPool operates in two stages: it first uses an efficient linear attention mechanism to capture global token interactions across the entire document, then employs a novel **Spectral Token Compression (STC)** technique to compress these globally-aware token representations into a compact, context-aware vector. We demonstrate that SATPool consistently and significantly outperforms established baselines through extensive experiments on diverse tasks, including long-document classification, Retrieval-Augmented Generation (RAG), multimodal RAG, and factuality consistency evaluation. Our work presents a practical, plug-and-play solution that unlocks the full potential of pre-trained encoders for long-form text without requiring costly retraining, enabling more robust document-level understanding and retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Encoding and representing long documents with Transformer encoder-based models, such as BERT, remains a significant challenge. A common solution is to split long contexts into smaller chunks, encode each chunk into an embedding, and then pool these embeddings to obtain a final document representation. This paper introduces a novel and efficient pooling strategy within this framework, demonstrating substantial improvements over existing pooling methods in both performance and parameter efficiency.

Given a set of chunk embeddings for a long document, the proposed method first applies linear attention to all embeddings, leveraging techniques from the Performer architecture. It then introduces a discrete-Fourier-transform-like approach, incorporating sine and cosine coefficients into the embedding representations. This projects the original token embeddings onto a set of K bases with varying frequencies. The final document representation is formed by concatenating these frequency-based bases with the result of standard mean pooling.

The authors conduct extensive experiments on four long-context datasets, comparing their method against three to four traditional pooling strategies. Results indicate that the proposed approach consistently outperforms others on metrics such as recall@K, F1, and exact match. Additionally, ablation studies reveal that the method achieves better performance with fewer parameters compared to mean pooling.

### Strengths
1. The introduction of sine and cosine coefficients into the representation is a novel and interesting idea that helps preserve structural information within the document. The experiments convincingly demonstrate the feasibility of this approach in both retrieval and certain question answering tasks. 
2. The proposed method is model-agnostic and parameter efficient. While it is particularly effective for long-context encoding, it is also broadly applicable to a wider range of use cases beyond just long document processing.

### Weaknesses
1. The overall contribution of the paper is limited. There is no evidence that the proposed pooling strategy improves end-to-end question answering performance or achieves state-of-the-art (SOTA) or near-SOTA results on the selected datasets. It would be informative to see how the proposed approach performs when integrated with existing SOTA methods.
2. The evaluation does not include comparisons with strong baselines. The use of mean pooling and MaxP as baselines is rather simplistic, and the Late Chunking approach does not appear to be a robust alternative, either. The results would be more convincing if the proposed method were compared against established SOTA approaches.

### Questions
n/a

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
The paper presents Spectral Attention Token Pooling (SATPool), an encoder-agnostic module to produce a single fixed-size, holistic vector for arbitrarily long documents. SATPool first applies an efficient Performer-style linear attention over the concatenated chunk embeddings to produce attention-reweighted token representations, then applies Spectral Token Compression (STC): it concatenates the mean embedding and the projections of the sequence onto K sine/cosine bases (i.e., spectral coefficients) to form a compact vector which is linearly projected to the final document embedding. Experiments on long-document classification (20News, EURLEX), RAG (NQ, HotPotQA), multimodal RAG (OKVQA, EVQA) and factuality-consistency tasks show consistent gains over standard pooling, MaxP, and a Late-Chunking baseline.

### Strengths
1. **Clear, intuitive two-stage idea.** Combining a linear global attention to collect cross-chunk interactions with a spectral pooling that explicitly encodes coarse positional structure is conceptually neat and interpretable (reconstructed signals / heatmaps). The STC visualization helps build intuition.
2. **Empirical gains across multiple tasks.** The paper reports consistent improvements in retrieval and classification metrics across several datasets and encoders (BERT, RoBERTa, GTR-T5, CLIP-DPR), supporting generality.

### Weaknesses
1. **Limited theoretical justification for STC.** STC uses fixed sinusoidal bases and concatenated coefficient vectors; while empirically effective, the paper lacks a deeper theoretical analysis on what spectral components capture in semantic terms and when such bases are sufficient (vs. learned bases or other positional encodings). The robustness test of paragraph reordering is helpful but not fully convincing as a general explanation.
2. **Efficiency / runtime / memory evaluation is shallow.** The paper motivates linear attention for scalability, but lacks a thorough wall-clock / memory benchmarking comparing SATPool (Performer + STC) to alternatives (late-chunking, Longformer, LongT5) under the same hardware and realistic document lengths. Practical adoption will depend heavily on throughput/latency tradeoffs for indexing and retrieval.
3. **Reproducibility choices and dataset sampling.** For some RAG experiments authors train on subsets (e.g., 20k training, 1k test) "for efficiency" — this raises the question whether gains hold at full scale and whether hyper-parameters are overfit to small subsets. More details on hyper-parameter search ranges and robustness across seeds would strengthen claims.
4. **Generation-side impact remains unclear.** While retrieval metrics improve, generation EM/F1 does not always follow; the paper notes the “needle-in-a-haystack” issue but stops short of analyzing why improved retrieval representation fails to translate to generation improvements (e.g., quality of retrieved passages, summary compression step, generator capacity). Additional analysis (cases where retrieval improved but generator still failed) is needed.
5. **Comparisons to latest baselines missing or limited.** The paper compares to mean, MaxP, Late Chunking and some long-context models, but does not include recent SOTA long-document dense retrievers (e.g., MemLong, Longtriever variants with similar experimental settings) beyond a subset of references; stronger, consistent baselines would contextualize gains.

### Questions
1. How sensitive is SATPool to the chunking strategy (chunk size, overlap / stride)? Is sliding-window chunking necessary, or do non-overlapping chunks produce similar H? Please report an ablation.
2. Please provide wall-clock throughput / memory comparison for SATPool vs. Late Chunking and Longformer-based encoders on a realistic document corpus. What are r and K choices in practice for long web-scale corpora?
3. Why use fixed sinusoidal bases rather than direct learnable data-driven basis? Or have you tried learned frequency bases or PCA/low-rank bases over token embeddings? Do you have some empirical comparison?
4. For the RAG experiments, when retrieval improves but EM doesn't, can you provide qualitative examples and an analysis that separates (a) retrieval error, (b) summarizer failure (if any in RAG system), and (c) generator misuse of retrieved context? This would illuminate the "needle-in-a-haystack" discussion more clearly.
5. Could STC be combined with late-interaction (dot-product over chunk-level tokens) approaches (i.e., hybrid fine-grained matching)? Any experiments or discussion is welcomed.

### Soundness
2

### Presentation
3

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
This paper aims to address the issue of representation fragmentation in long-document encoding caused by context window limitations. To this end, the authors propose SATPool, a plug-and-play two-stage pooling module. This module first employs efficient linear attention to capture global token dependencies, and then distills the sequence into a fixed-size vector containing semantic and structural information through STC.

### Strengths
1. SATPool is designed as a lightweight, encoder-agnostic module. This implies that there is no need to retrain expensive foundational models, thereby reducing application costs.
2. Employing efficient linear attention to capture global dependencies across the entire document is theoretically superior to simple mean pooling.
3. Spectral Token Compression is proposed to compress globally-aware sequences into fixed-size vectors that encompass both semantic and structural information.

### Weaknesses
1. The linear attention mechanism used in the paper is an off-the-shelf component. Although the authors utilize it for post-processing of a frozen encoder, this is not a conceptually novel invention.
2. In the fields of signal processing and information retrieval, using the low-frequency components of frequency-domain transformations to create content fingerprints is a well-established classical technique. In this paper, STC merely applies this classical technique to the embedded sequences of a Transformer and concatenates it with the mean vector to serve as a trainable pooling layer, which represents an incremental application.
3. SATPool still relies on chunking in Stage 0. When processing the first chunk, it has no knowledge of the content of the fifth chunk. It attempts to guess and reconstruct the global context from a pile of already fragmented information, which presents practical difficulties.
4. SATPool is a lightweight module that is feasible in engineering but lacks sufficient innovation. Experimentally, it mainly demonstrates superiority over simple mean pooling.

### Questions
1. To what extent can the linear attention in Stage 1 truly recover the semantic dependencies that were already lost during the encoding phase by performing post-hoc reweighting on an already information-impaired sequence? Is the theoretical performance upper bound of this method inherently lower than that of a long-context model that can access the global context during encoding?
2. SATPool introduces two key hyperparameters: the number of spectral components K and the low-rank dimension r of the linear attention. Do these two hyperparameters exhibit interaction effects? Does the high sensitivity of this method to K undermine its claimed plug-and-play convenience?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a technique for generating long-document representations. It uses an efficient linear attention mechanism, Performer (Choromanski et al., 2020), to capture global token interactions, followed by a novel Spectral Token Compression method to compress these globally-aware token representations into a single vector.

The proposed method is evaluated on tasks such as document classification, retrieval, and factual consistency detection. The authors demonstrate that it outperforms several baselines, including Mean-Large, while using a similar number of trainable parameters.

### Strengths
The contribution is clearly explained, and its effectiveness is shown across multiple tasks.

The authors provide an intuitive explanation of the process, along with ablation studies on key design choices.

### Weaknesses
The claim that the method is a "plug-and-play solution" and works "without requiring costly retraining" seems somewhat misleading. The second stage, which involves reconstructing the signal, still requires training.

### Questions
Would suggest using more strong baselines, e.g., hierarchical transformers described in [1].

[1] Dai, Chalkidis, Darkner, and Elliott, "Revisiting Transformer-based Models for Long Document Classification", in Findings of EMNLP , 2022.

### Soundness
3

### Presentation
3

### Contribution
3
