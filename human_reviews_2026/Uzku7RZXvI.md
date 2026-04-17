# Summaries as Centroids for Interpretable and Scalable Text Clustering

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
We introduce k-NLPmeans and k-LLMmeans, text-clustering variants of k-means that periodically replace numeric centroids with textual summaries. The key idea—summary-as-centroid—retains k-means assignments in embedding space while producing human-readable, auditable cluster prototypes. The method is LLM-optional: k-NLPmeans uses lightweight, deterministic summarizers, enabling offline, low-cost, and stable operation; k-LLMmeans is a drop-in upgrade that uses an LLM for summaries under a fixed per-iteration budget whose cost does not grow with dataset size. We also present a mini-batch extension for real-time clustering of streaming text. Across diverse datasets, embedding models, and summarization strategies, our approach consistently outperforms classical baselines and approaches the accuracy of recent LLM-based clustering without extensive LLM calls. Finally, we provide a case study on sequential text streams and release a StackExchange-derived benchmark for evaluating streaming text clustering.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method that analyzes clustering centroids from a semantic perspective, aiming to enhance clustering performance by integrating large language model–based semantic understanding. The idea is interesting and relevant, particularly in its attempt to move beyond traditional embedding-based approaches. However, several aspects remain unclear, and the overall level of innovation is somewhat limited.

### Strengths
The paper addresses an important problem in semantic text clustering by exploring how cluster centroids can be derived and interpreted from a semantic perspective rather than purely from embedding similarity. This direction is conceptually appealing, as it attempts to bridge the gap between surface-level vector representations and high-level semantic understanding. The integration of large language model (LLM) semantics into the clustering process provides a meaningful enhancement over conventional embedding-based methods, offering improved interpretability and potentially richer cluster representations. The idea of treating summaries as centroids introduces an intuitive way to represent cluster meaning in natural language, which aligns well with current trends toward explainable and human-understandable machine learning. Empirically, the method demonstrates promising improvements on selected benchmarks, suggesting that semantic augmentation can indeed refine clustering boundaries.

### Weaknesses
A major weakness of the paper lies in the lack of clarity regarding its methodological distinction from existing embedding-based clustering approaches. Although the authors claim to perform clustering from a “semantic” perspective rather than through vector representations, it is not clearly explained how this process fundamentally differs from using pretrained models such as BERT or SimCSE to obtain semantic embeddings. Without a formal theoretical or algorithmic justification, the reader is left uncertain whether the proposed method truly captures semantics in a new way or simply reuses existing embedding mechanisms under a different formulation. In addition, while the paper presents interesting results, the novelty appears incremental, mainly in the use of summaries as cluster centroids without demonstrating a substantial methodological breakthrough.

### Questions
1. How does the proposed semantic clustering method fundamentally differ from using pretrained embedding models like BERT or SimCSE for vector-based clustering?
﻿
2. Can the authors provide ablation studies or comparative results to isolate the contribution of the summarization-as-centroid mechanism from other components?
﻿
3. How robust is the proposed approach when applied to diverse domains, noisy data, or varying semantic granularities? does it generalize beyond the tested datasets?

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
4

### Summary
The paper proposes k-NLPmeans and k-LLMmeans, two variants of k-means for text clustering in which textual summaries are used as centroids. At scheduled iterations, the numeric centroid of each cluster is replaced by a textual summary of its documents, which is then re-embedded to continue k-means updates. The authors show that this “summary-as-centroid” update improves interpretability, improves semantic faithfulness of cluster prototypes, and often boosts accuracy. They also provide a mini-batch extension for streaming text. Experiments across several benchmarks and embeddings show improvements over classical clustering baselines and competitive accuracy relative to LLM-based clustering, while using a fixed LLM budget.

### Strengths
- **Simple but novel idea:** The notion of introducing interpretable textual centroids inside the k-means loop is elegant, practical, and original. It creates a direct, auditable link between cluster means and human-interpretable summaries.
- **Interpretability without post-hoc processing:** Unlike topic models or LLM-based pipelines that only label clusters afterward, the prototype is the cluster, which is useful for debugging, transparency, and downstream analyst workflows.
- **Low-resource applicability:** The LLM-free version (k-NLPmeans) is already stronger than vanilla k-means and viable in offline or low-budget environments.
- **Streaming extension:** The mini-batch variant is an important practical contribution.
- **Reproducibility:** Anonymous code is supplied, and the method is easy to reimplement.

### Weaknesses
- **Summarization hints:** Performance depends on the summarizer, especially in heterogeneous clusters. The paper tests several summarizers but does not provide guidance on when one strategy is preferable (e.g., extractive vs. abstractive by dataset characteristics).
- **Missing comparison on interpretability:** Interpretability is a key selling point, but comparisons are mostly against centroid-based clustering. Topic-model-style baselines (e.g., BERTopic,) would give a fairer interpretability comparison.
- **No analysis of failure cases:** Interpretability examples are all positive. The paper does not provide examples of misleading or low-quality summaries, which are critical for understanding robustness and limitations.
- **Limited cluster sampling analysis:** The few-shot summarization step relies on sampling a subset of documents per cluster, but the effects of different sampling schemes are not studied. This leaves uncertainty regarding robustness under noisy clusters.

### Questions
1. **On summarizer choice:** Could the authors provide guidance or heuristics on when extractive vs. abstractive summaries are preferable? In particular, do you expect different summarizers to be more effective under heterogeneous vs. homogeneous clusters? Is there any metric that could be used to adaptively select the summarizer type?
2. **On interpretability baselines:** Since interpretability is a key motivation for the work, why are topic-model-style methods not included as baselines? These approaches also produce human-readable cluster prototypes, so they seem more directly comparable than centroid-only methods.
3. **On failure modes:** All interpretability examples provided in the paper illustrate successful summarization. Could the authors provide examples or qualitative analysis of failure cases (e.g., when summaries are too generic or misleading) to better understand the limitations of the technique?
4. **On sampling strategy in few-shot summarization:** The few-shot variant relies on sampling only a subset of cluster documents to feed the LLM summarizer. Have you evaluated whether different sampling strategies (e.g., diversity-based, entropy-based, or centroid vs. edge-document sampling) materially affect summarization quality and, in turn, clustering performance?

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
3

### Summary
In this manuscript, the authors aim to address the problems of poor interpretability of numeric centroids and high scalability costs in traditional text clustering methods. Specifically, the proposed k-NLPmeans uses lightweight and deterministic classical NLP summarizers to periodically replace numeric centroids with textual summaries. The proposed k-LLMmeans leverages LLMs for summaries under a fixed per-iteration budget. Experimental results across diverse datasets and embedding models show that the proposed method outperform baselines.

### Strengths
In this manuscript, the authors aim to address the problems of poor interpretability of numeric centroids and high scalability costs in traditional text clustering methods. Specifically, the proposed k-NLPmeans uses lightweight and deterministic classical NLP summarizers to periodically replace numeric centroids with textual summaries. The proposed k-LLMmeans leverages LLMs for summaries under a fixed per-iteration budget. Experimental results across diverse datasets and embedding models show that the proposed method outperform baselines.

### Weaknesses
There are some concerns for the manuscript as follows:

1.How to set k in the experiments? The influence of k in the k-means to the experimental results is not discussed.
2.In the example of Figure 1, it is based on the results of k-means. However, in the proposed method, the authors proposed new summarization to compute a textual prototype in place of the standard centroid update. Thus, how the proposed method guarantee that the instances in the same cluster can be used to generate promising summaries? In other words, if the texts in the same cluster express very different topics, how can it be ensured that the generated summary can cover all the topics?
3.The experimental results on Table 1 are not satisfied. The baseline including agglomerative of 1967 year can still achieve impressive performance in some case.
4.In the experimental results reported on Table 3, the proposed method does not seem to show advantages either. So, how can the effectiveness of the proposed method be proved?

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work proposes to modify standard k-means clustering such that instead of using the mean of the embeddings of each element in the cluster to compute the centroid, we generate a textual summary of cluster population and then embed the summary text to get the centroid. The summarization can be done for a subsample of the current cluster population using either LLMs or traditional summarization methods. This method of computing centroids is applied in certain iterations of the kmeans algorithm besides the usual arithmetic centroid calculation in other steps. This approach demonstrates better clustering performance as measured by accuracy and NMI scores with respect to ground truth labels over multiple datasets, and also when using diverse embedding models for embedding the texts. The authors apply this approach also to a streaming based setting where data comes in batches over time. For this, they create a benchmark based on StackExchange posts divided over multiple years. They demonstrate performance gains from using the summary as a centroid approach on this benchmark and also release this benchmark.

### Strengths
- The proposal of using summaries as centroids in k-means clustering looks like a very innovative and easy to implement approach.
- The authors included diverse ways of computing the summary centroid besides simply querying an LLM such as TextRank, SVD etc which can provide computationally cheaper alternatives.
- The approach shows good gains in performance over standard k-means clustering demonstrated consistently across many datasets and while using many different embedding models (Table1-4).
- The authors experiment with clustering in both full dataset and batched (streaming) modes, showing benefits in both settings.

### Weaknesses
Not much of a weakness but a suggestion : I would suggest elaborating on the NMI metric to give the readers a brief explanation of how it is calculated.

### Questions
Have you done analysis of what happens when and if the embedding of the summary lies outside the convex hull of cluster's datapoints? Intuitively it seems that the centroid must stay inside the hull and this is guaranteed to be true when the centroid is the arithmetic mean. However, since this is not guaranteed that the embedding of the summary would behave this way, I wonder if the stability of convergence would be affected negatively by this.

### Soundness
3

### Presentation
3

### Contribution
3
