# Embedding-Based Context-Aware Reranker

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) systems rely on retrieving  relevant evidence from a corpus to support downstream generation. 
The common practice of splitting a long document into multiple shorter passages enables finer-grained and targeted information retrieval. 
However, it also introduces challenges when a correct retrieval would require inference across passages, such as resolving coreference, disambiguating entities, and aggregating evidence scattered across multiple sources.
Many state-of-the-art (SOTA) reranking methods, despite utilizing powerful large pretrained language models with potentially high inference costs, still neglect the aforementioned challenges.
Therefore, we propose Embedding-Based Context-Aware Reranker (EBCAR), a lightweight reranking framework operating directly on embeddings of retrieved passages with enhanced cross-passage understandings through the structural information of the passages and a hybrid attention mechanism, which captures both high-level interactions across documents and low-level relationships within each document.
We evaluate EBCAR against SOTA rerankers on the ConTEB benchmark, demonstrating its effectiveness for information retrieval requiring cross-passage inference and its advantages in both accuracy and efficiency.
Our source code is available at https://github.com/BorealisAI/EBCAR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes EBCAR (Embedding-Based Context-Aware Reranker), a lightweight reranking framework that operates entirely in the embedding space. Experiments on the ConTEB benchmark demonstrate that EBCAR achieves strong performance in tasks requiring cross-passage entity disambiguation, coreference reasoning, and structural dependency modeling, while maintaining a favorable trade-off between accuracy and efficiency.

### Strengths
1. The method operates directly in the embedding space, avoiding repeated LLM-based text processing and improving inference speed.
2. The experiments are conducted on the ConTEB benchmark, which emphasizes cross-passage reasoning, and include both in-distribution and out-of-distribution evaluations, with clear and comprehensive results.

### Weaknesses
1. Since embeddings represent compressed information, fine-grained textual details may be lost. The paper does not conduct a quantitative analysis of which types of information are most susceptible to loss.
2. The model keeps the query embedding unchanged, yet no comparison is provided with an alternative setting where the query representation is updated. An analysis could clarify whether contextualizing the query improves or destabilizes semantic interactions.
3. The paper would benefit from qualitative analyses. Specifically, visualizing hybrid attention heatmaps for typical success and failure cases to illustrate how the model routes information within and across documents, thereby improving interpretability.
4. Most SOTA rerankers (e.g., RankVicuna, RankZephyr) are fine-tuned on MS-MARCO rather than ConTEB, whereas EBCAR is trained specifically on ConTEB, raising potential concerns about experimental fairness and comparability.
5. The terminology is inconsistent: document-aware masked attention (line 238) and dedicated masked attention (line 94) are used interchangeably and should be unified for clarity.

### Questions
1. Although the ablation study includes w/o Pos, w/o Hybrid, and w/o Both variants, the paper does not explain why the effects of these components differ in direction across certain datasets.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents EBCAR: a new reranker model that works directly on the retrieved embeddings of the passages.

EBCAR concatenates the query and retrieved passages into one sequence, then calculates refined contextual embeddings for these passages in order to score them.  The refined embeddings are calculated by a  modified transformer architecture: a new dedicated masked attention module is added, where the attention is computed within passages of the same documents only. This document-masked attention is added along with the standard full-attention before the add+norm layer in the transformer encoder.

The paper has a comprehensive ablation and evaluation sections with multiple datasets covering various domains, showing uplift against strong baselines, and nearly a 7x throughput increase over the fastest neural reranker baseline in the experiments.

### Strengths
- The paper is easy to read and the contribution is clear.

-  The dedicated masked attention module is a novel idea that seems to work well in capturing intra-document coherence and semantics.

- The EBCAR model shows strong performance despite a very small size (126M parameters), showing significant uplift compared to much larger models.

- The throughput of EBCAR is impressive, this might be in part due to its smaller size considered in the experiments, but also due to working on the embeddings directly.

- The ablation studies in the appendix are comprehensive and justify the choices and architecture changes.

### Weaknesses
- Having a document ID embedding makes adding new documents difficult since a new embedding will need to be learned for it. It would be interesting to see experiments where the document embedding is computed from the token or passage embeddings to make adding new documents easier.

- Furthermore, learning a specific embedding per document makes me concerned about potential overfitting to these documents. It would be interesting to further analyze the content of these embeddings to see what concepts they captured that isn't already included in the tokens.

- I think a scaling study is lacking for this paper and would make it a lot stronger. Since the model is performing well on smaller scale, it would be interesting to see how the performance changes as we scale the model from 126M to 8B and from one GPU to 8 GPUs or even more, it would also be useful to understand how much of the impressive throughput is due to the architecture and embedding caching vs. the small number of parameters.

- Even though the paper focuses mostly on LLM-based reranker, but including models like BERT in the baseline drove me to wonder about the performance of a multi-vector ranking and retrieval methods like ColBERT in the same setup. Having this datapoint would be excellent for a fuller picture of how this model performs, especially to understand the impact of the information-bottleneck mentioned in the paper, since the multi-vector methods do not suffer from such information-bottleneck.

### Questions
- Figure 2 is very helpful, but it does not clearly show how the two attention modules are combined. Is the add an norm applied to all three sets of embeddings (raw, with shared full attention, and masked attention)

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Embedding-based Context-Aware Reranker (EBCAR), a reranking model designed to address the challenges of efficiency and cross-passage inference in passage reranking tasks. To improve efficiency, EBCAR operates directly on the embeddings of the query and passages, rather than their raw text representations. To enhance cross-passage reasoning, it augments each passage embedding with two additional components: a document ID embedding and a passage position embedding, enabling the model to better capture contextual and structural information across passages. Furthermore, the paper introduces a hybrid attention mechanism to refine the passage embeddings, which are then used to produce the final ranking.

### Strengths
1. The paper proposes a novel embedding-based context-aware reranker that improves both efficiency and cross-passage inference. 
2. Experimental results on eight datasets show that the proposed approach achieves the best average score. 
3. The paper is well-written and easy to follow.

### Weaknesses
1. EBCAR requires learning a fixed-size document ID embedding matrix of size k × d, which limits scalability. Expanding to a larger document set would require retraining the model to accommodate new document IDs, making it less flexible in dynamic or large-scale retrieval settings.
2. As shown in Table 1, the proposed model achieves exceptionally high performance on the Football and Insur datasets (e.g., 80.19 vs. 11.63, and 40.74 vs. 4.76 compared to the best baseline), which contributes significantly to the overall average improvement. However, the paper lacks insight into why the model performs so well on these datasets. Moreover, the lack of detailed descriptions of each dataset makes it difficult to understand and analyze the differences in model performance.
3. The reranking experiments are conducted only on passages retrieved by Contriever, making it unclear whether the proposed method can generalize to results retrieved by stronger retrievers such as E5 or BGE. Evaluating on diverse retrieval backbones would strengthen the validity and robustness of the proposed approach.
4. The paper lacks discussion and comparison with document-level retrievers or rerankers, which are relevant baselines given the cross-passage inference objective.

### Questions
1. How scalable is EBCAR to larger or dynamic document collections, given that the document ID embedding matrix has a fixed size? 
2. Can the authors provide insights into why EBCAR achieves such large gains on the Football and Insur datasets? Are there characteristics of these datasets that particularly favor the proposed method?
3. Has the model been tested with retrieval results from stronger retrievers like E5 or BGE? If not, how confident are the authors that EBCAR would generalize to different retrieval backbones?

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
3

### Summary
EBCAR is a novel reranking method to solve the critical challenge of cross-passage inference, ignored by other methods. Specifically, it incorporates the structural information of passages via a hybrid attention mechanism to capture both high-level inter-document and low-level intra-document interactions. Combining evidences from multiple passages, the performance improvement on benchmark datasets shows its effectiveness and efficiency.

### Strengths
--Cross-passage inference, the core problem studied in this paper, is critical for complex reasoning tasks.

--NDCG@10 of the proposed method is consistently better than state-of-the-art baselines on benchmark datasets.

--It is more efficient compared with rerankers based on large and high-cost language models.

### Weaknesses
--Both the hybrid attention and the positional encoding are two key components for EBCAR according to the ablation study in Table 2. However, how the hybrid attention mechanism improves the cross-passage understanding is still a question to be explored and explained further. Maybe a detailed analysis of the hybrid attention should be provided and also an intuitive example is used to illustrate their relationship.

--Compared with the efficiency of Contriever, the advantage of the proposed method EBCAR is not obvious. Its time complexity needs to be analyzed for complementary. 

--The performance improvement of the proposed method in Table 1 is 8~10 times of state-of-the-art baseline methods on Football and Insure, while the performance improvement is not so large on other datasets. The underlying reason should be clarified.

### Questions
-The proposed embedding based rerank method is highly dependent on the embedding model, such as BERT, Mistral and Llama. It is unfair to compare reranking models with different embedding models listed in both Table 1 and 3.

-The performance comparison between EBCAR and ICR in Table 1 under the evaluation of NDCG@10 is inconsistent with their comparison result in Table 3 under the evaluation of MRR@10. The underlying reasons are to be explored.

-How the reranked results’ influence on the generation results should be further explored.

### Soundness
2

### Presentation
3

### Contribution
2
