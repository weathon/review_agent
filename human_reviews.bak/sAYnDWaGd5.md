# Quest: Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 3

## Abstract
Recent advancements in large language models (LLMs) have highlighted the importance of extending context lengths for handling complex tasks. While traditional methods for training on long contexts often use filtered long documents, these approaches lead to domain imbalances, limiting model performance. To address this, techniques like random document concatenation (Standard) and similarity-based methods (KNN, ICLM) have been developed. However, they either sacrifice semantic coherence or diversity. To balance both aspects, we introduce Quest, a query-centric data synthesis method aggregating semantically relevant yet diverse documents. Quest uses a generative model to predict potential queries for each document, grouping documents with similar queries and keywords. Extensive experiments demonstrate Quest's superior performance on long-context tasks, achieving remarkable results with context lengths of up to 1M tokens and confirming its scalability across various model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Quest, an approach for grouping documents to train long-context LLMs. Existing approaches for document grouping either use random grouping or grouping by similarity. However, the former leads to aggregated documents that contain unrelated pieces, while the latter yields aggregated documents with repeated information coming from similar documents. Quest then seeks to balance the coherence of the aggregated documents and the diversity of their content.

The Quest algorithm first associates each document to a query, which is used to extract a representative keyword associated to the document. Documents are then indexed based on their respective keyword, in the same spirit as an inverted index. During training, a keyword is sampled and an aggregated document is formed by randomly sampling documents associated to this keyword up to the desired context size. The keyword sampling procedure also takes into account the rarity of the keyword (i.e., the number of documents associated to a keyword) by oversampling rarer keywords, to have a more balanced domain representation. Quest is validated through extensive experiments on long-context tasks and is also shown to preserve the short-context capabilities of the original pre-trained model.

### Strengths
- The paper is overall well-written, clear and easy to follow.
- The experiments are comprehensive and convincing.

### Weaknesses
- Quest is defined as a simple heuristic that is not backed by any theoretical guarantees or motivations.
- Several design choices in the approach lack justifications or intuitive explanations, and thus feel arbitrary (see specific points in the "Questions" section).

### Questions
- The first step of the Quest algorithm is to associate documents with keywords. This is done by first formulating a query from the document, before extracting keywords from this query. It's not very intuitive why this document-to-query strategy is favored over applying any standard summarization technique to the document. A summary may retain more of the document's information than a single query, and thus might lead to higher-quality keywords. The paragraph lines 93-105 attempts to make some links with search engines, but the motivation could benefit from being further clarified and strengthened.
- At the end of the keyword extraction step, the keyword that is eventually retained for a document is randomly selected among the different keywords associated to the document. This strategy seems a bit unintuitive and risky, as it could lead to the selection of a keyword that is only marginally relevant to the document. Why isn't the Rake score used here to decide which keyword to retain? Or even simply retain the keyword with the largest Rake score?
- Was there any evaluation conducted to validate the quality of the keywords identified for the documents? This could for example be done by asking some human annotators to provide keywords for a few documents and compare these against the keywords obtained through Quest, or instead ask them to rate the relevance of the Quest keywords. This seems like an important sanity check to conduct given the importance of proper keyword-document matching in the proposed approach.
- How is the oversampling of keywords from the short-index set $I_s$ done in practice? How are the probability of sampling a keyword from $I_s$ and that of sampling from $I_l$ defined, exactly? Adding the mathematical formula for this would be helpful for the reader's understanding.
- What is the motivation for splitting $I$ into only two sets $I_s$ and $I_l$ instead of having a more fine-grained approach? For example, one could define an adaptive probability of sampling a keyword based on the number of tokens in its associated documents. Such an approach would enable a more tailored treatment of the different keywords to compensate for their rarity.
- Figure 3 contains several typos: "query genaration" -> "query generation", "random smaple" -> "random sample"
- In the results on Longbench in Table 1, it seems that the superior average performance of Quest over baselines is essentially due to its significantly better results on the Code Completion task. The good performance of Quest on this specific task is attributed to the use of Levenshtein distance as the metric. It is however not entirely clear why using Quest would specifically boost this metric, so more detailed explanations would be appreciated.
- It is surprising that Quest achieves 100% accuracy on the needle-in-a-haystack task for a context of 1M tokens, but gets 97% accuracy for a context of 128k tokens (which should intuitively be easier). Are there any intuitive explanations for this phenomenon?

### Soundness
3

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
3

### Summary
This paper introduces Quest, a query-centric data synthesis approach addressing data scarcity and domain imbalance issues in long-context modeling of LLMs. Quest achieves a balance between semantic correlation and context diversity by predicting potential queries for each document and aggregating semantically relevant but low-redundancy documents based on query similarity and keywords. Experiments demonstrate Quest's superior performance on long-context tasks ranging from 32k to 1M tokens.

### Strengths
1. The motivation is clear. The authors identifie two critical challenges in long-context training data: scarcity and domain distribution imbalance, then provide analysis of limitations in existing methods (Standard, KNN, ICLM).
2. The authors propose a query-centric data synthesis approach to alleviate the above problem, includes: query-based document aggregation moduls, keyword extraction and filtering modules, split ratio mechanism.
3. Extensive evaluation across various model scales (1.4B-12B) and context lengths (32k-1M) supports the method's effectiveness.

### Weaknesses
1. The author used a query generation model to generate different queries for different documents, which would make the quality of query generation affect the final performance of the model. The author should discuss in more detail the impact of query quality or different query generation methods on model performance.
2. The model approach is more like an integration of existing tools, without any contribution to methodological or theoretical innovation. For example, the query generation and selection process directly employ the off-the-shelf tools (doc2query, RAKE) without significant enhancement.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QUEST, a long-context scaling method for LLM pretraining. The proposed method focuses on data sythesis and introduces a simple algorithm QUEST, in which queries are first generated by documents, followed by key words generation of the queries. Then, the keywords are used as inverted index to group similar documents, where the similar documents are arranged within the same training batch to improve long-context modeling by cross-document reasoning. The authors perform experiment on continual pretraining and demonstrate the effectiveness of the proposed QUEST on both long- and short-context benchmarks. The authors also provide additional analysis on the training dynamics and scaling properties of long-context modeling using QUEST.

### Strengths
1. The authors propose a simple and scalable method to group documents for long-context pretraining
2. The proposed method seem to work and achieve performance gains on the selected short- and long-context benchmarks
3. The authors also provide many interesting observations and analysis of QUEST and long-context modeling, which may be helpful for LLM pretraining

### Weaknesses
1. Although the proposed method is simple, it largely follows previous approaches but with a different indexing and sampling method. In addition, no ablation is provided so we are not sure which part of the QUEST is the most helpful (query-based indexing or sampling etc.).
2. The performance of QUEST is quite similar to ICLM on long-context benchmarks. Interestingly, in appendix, it seems QUEST often underperforms compared to ICLM or even KNN. THis raises questions on the real performance of the proposed method.
3. The authors provide many interesting observations and analysis, yet these do not necessarily demonstrate the validity of QUEST. For instance, the examples in 5.2 show that most baseline methods also have good clustering quality.

### Questions
1. In fig3, Llama 3 8B 128k has accuracy of 0.97, so why even longer context of 1M as in fig 1 would result in 1 accuracy?
2. Why the performance in appendix from table 8 to table 10 are in consistent, I find it hard to inteprete these results and how do they differ from the tasks that are present in the main text?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper proposes a method to generate synthetic long documents for model training. It aims to solve the problem of standard method that may concatenate unrelated short document to form a long document, and the similarity-based methods that concatenate too similar documents leading to the lack of diversity. The proposed method - Quest - selects the documents to be concatenated by the common keywords they would have in the generated queries to which they may answer. This is intended to concatenate documents that may be on a common topic (keywords), without being too similar.
The experiments on several datasets show that the synthetic data generated by the proposed method can lead to some improvements in the test tasks (on average).

### Strengths
1. The problem observed in the previous methods to generate synthetic data is interesting: the documents to be concatenated should be similar to some extent, but not too similar.
2. The experiments show that the method can result in improvement on some tasks.

### Weaknesses
1. The paper proposes a quite complex algorithm using doc2query, keyword indexing, then sampling through keywords. The paper does not motivate why these steps are necessary. One could imagine that much simpler methods would be able to lead to synthetic data that are to some extent similar, as is required by the authors. For example, if it is observed that the method based on KNN lead to concatenating too similar documents, it would be possible to select documents to be concatenated that are similar to some extent, but not the most similar. Could this simpler method lead to similar results?
2. The paper tends to over-claim the advantage of the proposed method. For example, it is said "Table 1 compares the Longbench results, showing that Quest consistently outperforms other methods across model sizes, ...". Looking at Table 1, one can see that the proposed method outperforms the others on average, but underperform them on several datasets. The advantage of the method is not so clear. This over-claim appear in other observations as well (e.g. Table 3 shows that Pythia obtains the best performance on 4 datasets, while Quest only on 2). Globally, if one consider all the datasets, the demonstration that the proposed method is better is very weak.
3. Some of the figures are unclear. Fig. 1 is not well explained. It intends to show the perfect performance of Quest on a specific task. However, the figure does not provide very useful information in addition to say it is perfect. One would also like to see some other methods in comparison, and to understand better the task itself. In Fig. 2, the dotted lines are not explained. One can guess later that they correspond to the performance of the model in some task.
4. Fig. 5 compares the results using the documents of some length in the original dataset, and the synthetic documents. It shows that the latter perform slightly better. Do these data have equivalent sizes? If the synthetic data are much more than the real subset of data, the observation would not be surprising. However, it would be difficult to conclude that the synthetic data are better than the real data, only they are more.

### Questions
Have you tried simpler methods to select documents to be concatenated, e.g. based on controlled similarity measure, so that the documents are within some range of similarity? Would this achieve the same goal as the proposed method?

### Soundness
2

### Presentation
3

### Contribution
2
