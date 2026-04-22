# Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Most text retrievers generate $\textit{one}$ query vector to retrieve relevant documents. Yet, the conditional distribution of relevant documents for the query may be multimodal, e.g., representing different interpretations of the query. We first quantify the limitations of existing retrievers. All retrievers we evaluate struggle more as the distance between target document embeddings grows. To address this limitation, we develop a new retriever architecture, $\textbf{A}$utoregressive $\textbf{M}$ulti-$\textbf{E}$mbedding $\textbf{R}$etriever $(\textbf{AMER})$. Our model autoregressively generates multiple query vectors, and all the predicted query vectors are used to retrieve documents from the corpus. We show that on the synthetic vectorized data, the proposed method could capture multiple target distributions perfectly, showing 4x better performance than single embedding model. We also fine-tune our model on real-world multi-answer retrieval datasets and evaluate in-domain. \model{} presents 6 and 16\% relative gains over single-embedding baselines on two datasets we evaluate on. Furthermore, we consistently observe larger gains on the subset of dataset where the embeddings of the target documents are less similar to each other. We demonstrate the potential of using a multi-query vector retriever and open up a new direction for future work.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Autoregressive Multi-Embedding Retriever (AMER) to address the limitation of standard dense retrievers, which typically generate a single query embedding and struggle to retrieve diverse sets of relevant documents for queries with multiple valid answers. AMER generates multiple query embeddings per query autoregressively, enabling better retrieval of diverse targets. Experiments on both synthetic and real-world datasets show that AMER significantly outperforms single-embedding baselines, especially when target documents are less similar to each other, demonstrating the importance and effectiveness of modeling diversity in document retrieval.

### Strengths
+ This paper proposes a new retrieval method where the original text, processed through the proposed AMER framework, can generate multiple retrieval embeddings. These embeddings enable comprehensive retrieval of the original text from multiple dimensions, identifying multiple documents containing the correct answers and thereby improving the performance of multi-question retrieval. 
+ The effectiveness of the method is further validated through experiments.

### Weaknesses
+ Indeed, the idea of representing queries and documents using multiple vectors for retrieval, namely multi-vector retrieval, is not new, and the authors seem to ignore such relevant literature. One notable method that the authors need to mention and compare against is "Colbert: Efficient and effective passage search via contextualized late interaction over bert, SIGIR 20", which is the first paper on multi-vector retrieval. The authors need to cite and compare against this paper.
+ I understand that the authors may want to learn multiple query embeddings for retrieval, which is not explored by papers like ColBert. However, one recent paper, "POQD: Performance-Oriented Query Decomposer for Multi-vector retrieval, ICML 25" has proposed a relevant idea, i.e., dynamically obtaining these query embeddings through learning to decompose queries. The authors also need to analyze the difference from this work and ideally compare the proposed method against it in experiments.
+ Some technical details are also not clear to me, particularly, how the document embeddings are obtained. Based on my understanding, each document is also represented by multiple embeddings, which play a critical role in learning query embeddings. However, no description is provided on how these document embeddings are generated for each document.

### Questions
See above

### Soundness
2

### Presentation
2

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
The paper tackles the task of multi-query retrieval. It introduces AMER, a retrieval model that generates multiple query embeddings autoregressively. All the predicted query vectors are then used to retrieve documents. This approach directly addresses the limitations of
single-query vector retrievers and enables retrieving diverse outputs. Finally, the authors test the proposed approach on several benchmarks (one synthetic introduced in the paper and two other existing benchmarks).

### Strengths
The paper tackles an important task. It shows some limitations in current methods and proposes a method that addresses them. Overall, the idea seems interesting to me, but I feel like the improvement on real world tasks is a lot lower than I expected.

### Weaknesses
My main concern is related to the experimental part: while I agree that in principle one query can be mapped to diverse documents and the experiments show that there's a degradation in performance by choosing the most diverse set of documents, the synthetic data experiments feel tailored explicitly for this case. If the proposed method achieves 100% performance on the synthetic data, it feels like the experiment is either to simple or specifically tailored for this case. I think that overall it is true that you can in principle have a query matched to diverse documents, but there's a limit to that - if the set of documents is too diverse, then the query shouldn't retrieve all of them. So, I feel like there's no real insight gained from the synthetic data experiment. My next concern is related to the novelty which seems relatively limited since the idea of learning multiple embeddings for the same query has been explored for example in cross-modal retrieval

Yale Song, Mohammad Soleymani; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1979-1988

weak comparison with other methods (see below)

### Questions
Maybe I am missing something, but why isn't the sota comparison made against methods from Fig 1?

### Soundness
3

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
2

### Summary
This paper targets retrieval settings where a single query has multiple, diverse target documents and single-embedding retrievers fail to cover distant clusters. It proposes AMER, an autoregressive multi-embedding retriever that generates a sequence of query vectors from the query encoder while keeping the document encoder frozen. Training uses InfoNCE with Hungarian matching to align unordered positives to generated vectors and scheduled sampling to reduce exposure bias. At inference, each vector retrieves candidates that are merged via round-robin. Results show perfect coverage on synthetic data and consistent, sometimes large, gains on AmbigQA/QAMPARI, especially when targets are dissimilar.

### Strengths
This paper reframes retrieval for multi-target queries as autoregressive generation of multiple query embeddings, creatively combining sequence modeling with contrastive retrieval. It uses Hungarian matching to align unordered positives and scheduled sampling to reduce exposure bias, removing a core limitation of single-vector retrievers.

### Weaknesses
(1) Baselines are incomplete. The paper does not compare against strong multi-vector retrievers (e.g., the ColBERT[1] family) or competitive query-rewriting/expansion approaches (such as MMLF[2]). Although architectural limitations of late interaction are mentioned, the lack of head-to-head metrics under matched budgets makes it hard to gauge relative advantage. 

(2) Single-answer regimes are underexplored. It remains unclear how the method behaves when a query has a single narrow intent. 

(3) The work does not explore instruction-tuning that ask the model to produce multiple distinct embeddings , leaving open whether instruction control could better leverage LLM generative priors for diversity.

[1]. Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on
research and development in Information Retrieval, 2020.

[2]. Yuan-Ching Kuo, Yi Yu, Chih-Ming Chen, and Chuan-Ju Wang. 2025. MMLF: Multi-query Multi-passage Late Fusion Retrieval. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 6587–6598, Albuquerque, New Mexico. Association for Computational Linguistics.

### Questions
As above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles a common and practical problem: standard search retrievers are bad at handling ambiguous queries. The authors first demonstrate that existing retrievers all struggle as the distance between target document embeddings grows and then go ahead to present AMER as a solution. Instead of generating just one query vector, it auto-regressively generates multiple query vectors. To handle the fact that the multiple target documents are an unordered set, the authors also employ an elegant Hungarian matching algorithm to find the optimal pairing between generated query embeddings and target document embeddings.

### Strengths
- The paper's core idea is simple, intuitive, and rigorously tested. It is also well written.
- The authors tackle a core, fundamental limitation of the dominant bi-encoder retrieval paradigm. By providing a practical architecture to move "beyond single embeddings", this work opens a new and important direction for retrieval model design.
- The authors look at performance gain on synthetic benchmark that showcases it's superior performance but also show the moderate gain on real world data.

### Weaknesses
- The authors state they "assume a setting" with a frozen document encoder for "faster development". This is a major experimental concession. While this makes testing easier, it creates a potential disconnect between the latest documents.
- There might be a potential practical downside of higher inference cost due to *m* separate matches.
- The true number of distinct answers (or answer clusters) varies per query, from one to many. This fixed parameter might not perform well in certain cases or might be inefficient.
- To train this system, you need to find all the gold documents for all the different answers for each query. This can be potentially noisy and unstable.

### Questions
- Have the authors though about any practical limitations like increased cost/latency or fixed document encoder?

### Soundness
3

### Presentation
3

### Contribution
2
