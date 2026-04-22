# $\text{E}^2\text{Rank}$: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 2

## Abstract
Text embedding models serve as a fundamental component in real-world search applications. By mapping queries and documents into a shared embedding space, they deliver competitive retrieval performance with high efficiency. However, their ranking fidelity remains limited compared to dedicated rerankers, especially recent LLM-based listwise rerankers, which capture fine-grained query–document and document–document interactions. In this paper, we propose a simple yet effective unified framework $\text{E}^2\text{Rank}$, means \textbf{E}fficient \textbf{E}mbedding-based \textbf{Rank}ing (also means \textbf{Embedding-to-Rank}), which extends a single text embedding model to perform both high-quality retrieval and listwise reranking through continued training under a listwise ranking objective, thereby achieving strong effectiveness with remarkable efficiency. By applying cosine similarity between the query and document embeddings as a unified ranking function, the listwise ranking prompt, which is constructed from the original query and its candidate documents, serves as an enhanced query enriched with signals from the top-K documents, akin to pseudo-relevance feedback (PRF) in traditional retrieval models. This design preserves the efficiency and representational quality of the base embedding model while significantly improving its reranking performance. Empirically, $\text{E}^2\text{Rank}$ achieves state-of-the-art results on the BEIR reranking benchmark and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark, with very low reranking latency.  We also show that the ranking training process improves embedding performance on the MTEB benchmark. Our findings indicate that a single embedding model can effectively unify retrieval and reranking, offering both computational efficiency and competitive ranking accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a unified framework that enables a single embedding model to handle both retrieval and listwise reranking efficiently. The approach reinterprets listwise prompts as pseudo-relevance feedback (PRF) queries and introduces a two-stage training scheme combining contrastive and listwise ranking objectives, achieving improvements on BEIR and BRIGHT benchmarks with low latency.

### Strengths
1. This paper proposes a conceptually idea of using a single embedding model for both retrieval and reranking tasks. Specifically for reranking, the proposed method rely on  a listwise ranking prompt, which is constructed from the original query and its candidate documents, serving as an enhanced query.

2. The experiment demonstrates significantly fewer model size while maintaining competitive accuracy, making it attractive for real-world search applications.

3. This paper reports strong results on BEIR and reasonable performance on BRIGHT, showing that embedding-based reranking can approach LLM listwise ranking effectiveness.

### Weaknesses
1. The results of E2RANK-8B on DL19 appear inconsistent — reported as 72.65 in Table 2 but 72.95 in Table 1. Could the authors clarify where this discrepancy arises from?

2. The E2RANK results in Table 5 differ from those presented in earlier tables. Providing a consistent comparison or explicit explanation of these variations would enhance clarity and reproducibility.

3. Even though the proposed model is not trained with listwise prompts in its baseline form, one could still use the same listwise prompt formulation at inference time for reranking. Have the authors evaluated this variant? It would be valuable to understand how much performance improvement arises from the training on listwise prompts versus merely using them at inference, as this could help isolate the contribution of listwise training.

4. About the reference latency, this paper mainly report the model size as the latency. It would be better to include practical runtime cost when compared to the baselines that has the same model size, for example, when comparing E2RANK with RankQ3en3.

5. How does E2RANK scale when applied to substantially larger document collections or real-world web-scale retrieval settings? While the paper emphasizes efficiency, it would be useful to quantify latency under increasing corpus sizes to assess whether the claimed advantages persist at scale.

6. The proposed framework appears adaptable to various embedding backbones (e.g., E5, BGE, or GTR). How sensitive is the overall performance to this choice? Have the authors observed consistent improvements across different base models?

### Questions
Please see above weaknesses.

### Soundness
2

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
5

### Summary
This paper proposes E2Rank, a unified framework that enables a single text embedding model to perform both the first-stage retrieval and the second-stage listwise reranking. The core idea is to reinterpret the listwise reranking prompt, which contains the query and top-K candidate documents, as a Pseudo-Relevance Feedback (PRF) query. This "PRF query" is then encoded into an embedding, and reranking is performed efficiently via cosine similarity with pre-computed document embeddings.

### Strengths
1. This paper proposes E2Rank, a unified framework that enables a single text embedding model to perform both the first-stage retrieval and the second-stage listwise reranking. 
2. The paper provides strong, quantitative evidence of dramatically reduced inference latency compared to auto-regressive listwise baselines.

### Weaknesses
1. First of all, the motivation of this paper is somewhat peculiar. The use of a dual-encoder approach for reranking aims to reduce the cost associated with cross-encoder, yet it achieves even better performance than cross-encoder, which is rather peculiar. The effectiveness of pseudo-relevance feedback (PRF) in dense retrieval has already been demonstrated by many retrieval models [1, 2, 3, 4], and it is not the first time it has been used in this paper. A very naive approach to PRF involves concatenating top-retrieval results with the query to enrich the query representation. However, it is somewhat unexpected that using PRF for a dual encoder can directly outperform a cross-encoder (Even with both rank loss and contrastive learning loss). 
[1] Yuanhua Lv and ChengXiang Zhai. 2009. A comparative study of methods for estimating query language models with pseudo feedback. In Proceedings of the 18th ACM conference on Information and knowledge management. 1895–1898.
[2] Joseph Rocchio. 1971. Relevance feedback in information retrieval. The Smart retrieval system-experiments in automatic document processing (1971), 313–323.
[3] Guihong Cao, Jian-Yun Nie, Jianfeng Gao, and Stephen Robertson. 2008. Selecting good expansion terms for pseudorelevance feedback. In Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 243–250.
[4] Hang Li, Ahmed Mourad, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon. 2023. Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls. ACM Trans. Inf. Syst. 41, 3, Article 62 (July 2023), 40 pages. https://doi.org/10.1145/3570724

2. Secondly, the experimental comparisons do not seem reasonable. For example, in the comparison of MTEB, only last year's models were compared. The paper uses a backbone model from the Qwen3 series. Currently, Qwen3-embedding 8B MTEB eng v2 ranks second on the leaderboard. The Qwen3-embedding model is purely based on a dual-encoder architecture and does not use PRF. The model training utilizes both synthetic data and publicly available data. The Qwen3-embedding technical report also presents results using only publicly available data. However, the paper does not compare with Qwen3-embedding on MTEB. Additionally, the embedding model BGE-en-ICL [1], which shares a similar idea of enriching query expression as this paper, achieves a score of 66.08 on MTEB v1 (56), while the E2Rank in this paper scores 65.03.
 
Furthermore, in the comparisons on TREC DL and BEIR, the training datasets used for the fine-tuning listwise ranker models (the main baseline: RankQwen, in this work) being compared are relatively weak. For instance, RankQwen utilizes a dataset derived from RankZephyr. The original RankZephyr paper states that it was trained on a labeled dataset of 100,000 queries from MS MARCO. However, why does line 844 in the paper indicate that RankQwen used only 40K queries? In contrast, the E2Rank model employs 1.5M queries in the first stage and 87K queries in the second stage. It is unclear whether the performance improvement of E2Rank over RankQwen in re-ranking tasks stems from the method itself or the training dataset. Simply replacing the backbone without conducting comparisons on training sets of a similar scale raises concerns about the fairness of the experimental comparisons, which need to be addressed. 
[1] Li, C., Qin, M., Xiao, S., Chen, J., Luo, K., Shao, Y., ... & Liu, Z. (2024). Making text embedders few-shot learners. arXiv preprint. ICLR26.

3. The paper suffers from minor but noticeable grammatical errors and occasionally awkward phrasing (e.g., "remarkking," "documents as PRF contribute meaningfully").

### Questions
Please see the weaknesses.

### Soundness
2

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
5

### Summary
This paper introduces a new technique called e2rank where the query where the embedding can be used as a ranker. The key to the technique is that during reranking, the query is augmented with the top-20 documents and re-embed. This gives the query additional context, alleviating typical limitations with pointwise ranking techniques and gives a possible way for embeddings to compete with listwise reranking approaches. There are very strong results, although perhaps lacking on analysis to build intuition why the technique works well. Finally, e2rank seems very dependent on the correct training setup. Fortunately, there are many training details provided in the paper and appendix.

### Strengths
S1. The technique is a clever way to integrate more context into the query and improve quality when reranking with embeddings. This could be an appealing efficient alternative to listwise rerankers in some settings.

S2. The results are strong across many datasets, although the paper is lacking in key analysis.

S3. The paper includes results when training, and many details about the training setup.

### Weaknesses
W1. The figure 2 shows the technique does not benefit much beyond using the top-20 documents in the listwise prompt and can even get worse after the top-20. This is a similar pattern as shown in previous work on reranker weaknesses where reranker quality starts to degrade as more documents are reranked (Jacob et al, https://arxiv.org/abs/2411.11767), although in e2rank's case it is the size of the context rather than the number of documents. This could indicate e2rank is very sensitive to the training setup, and some other setup could result in better performance, although it could be a natural limitation of the technique.

W2a. More generally, the evaluation is very focused on extracting the top-20 documents when reranking the top-100. It could be that RankQwen3 and other baselines are stronger in other settings, such extracting the top-10 from the top-50, or even the top-1 from the top-10.

W2b. The latency benefits of e2rank are particularly constrained to the extract top-20 from top-100 setting, and will appear less (or more favorable) in other settings. Also, there is no latency comparison to cross-encoders, which would be much faster than RankQwen3. RankQwen3 is a sliding window model, which is very slow compared to other methods.

W3. From a latency perspective, there are key missing baselines. There is no comparison to Parry et al (https://arxiv.org/abs/2405.14589) which is almost a drop-in replacements for RankQwen3, but is much faster and uses a pivot-focused technique similar to e2rank (e2rank essentially uses multiple pivots but Parry et al uses a single pivot). Also, there are a range of other alternative techniques with quality-latency trade-offs such as pairwise rank prompting (PRP), see Oosterhuis et al (https://arxiv.org/abs/2504.12063) for an overview.

W4. The intuition behind e2rank is not sufficiently supported. There are no examples that show how e2rank is acting like PRF --- pseudo-relevance feedback (PRF) would typically refine the initial query using a small number of documents, pushing the query to emphasize topics shown in the top-N documents. If this is how e2rank works, then I would expect less training would be needed achieve similar performance + I would expect some simmilarity to PRF baselines (which none are compared to). Instead, my guess is that e2rank is doing well simply because the query has more context about the top-N retrieved documents can help to disambiguate the results. I am not sure if this really is the case, but there are no results or analysis to justify the intuition as far as I can tell.

W5. Similar to W4, it's hard to distinguish whether the training data or the e2rank algorithm is key to these good results.

### Questions
The BRIGHT and ReasonIR papers showed the embeddings are missing fundamental capabilties regarding query length. In the future, would you expect that new embedding models will still need the specialized training here, or it will be an innate property of new embedding models?

Did you consider alternative to choosing top-N for every query? For example, could use more or less depending on the query. Could also do random sampling from the top-100.

Do you notice whether e2rank is better at ranking documents inside the top-20, or beyond the top-20? Not clear whether Figure 3 answers this question.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a framework that enables a language model to perform both retrieval and efficient listwise reranking, addressing the high latency of reranking stage in multi-stage search systems. The core contribution is to augment the query with a few retrieved documents and use it to produce an enhance query embedding. Reranking is then performed through efficient cosine similarity between this enhanced query embedding and the document embeddings. The authors conduct experiments on multiple IR benchmarks such as TREC DL, BRIGHT, and MTEB and show competitive performance against existing rerankers while being efficient.

### Strengths
- The paper is easy to follow and mostly well written
- The results are reported for multiple standard IR benchmarks
- The paper addresses an important problem of efficiency in the reranking stage

### Weaknesses
### W1 - weak experimental setup
- (a) baselines are not training data controlled - RankQwen (the controlled baseline) is trained on a different and smaller dataset from 2 years ago while the proposed method is trained on a more extensive reranking data. The authors could've trained RankQwen on their stage 2 dataset as well?
- (b) The results of RankZephyr and other LLM based rerankers are reported in a sub-optimal setting, RankZephyr using splade as the retrieval model achieves ~80% on DL20 but the results in the paper are reported for BM25 retrieval where it achieves ~70% - this raises the question why did the authors choose the BM25 setting over the more optimal setting with splade retrieval
- (c) similarly, BRIGHT results are reported in the weaker setting without query expansion, BRIGHT paper itself shows much stronger results with reasoned query expansion and ReasonIR's primary results are reported with expanded queries so the choice of not evaluating with query expansion needs justification

### W2 - lack of novelty
The core of the paper is to learn better embeddings using a retrieval augmented query expansion, similar ideas have been proposed [1, 2] - the method adds an additional ranking objective on the embedding learning but that in itself is not a substantial update. Without a clear empirical justification it is hard to justify the contributions of the paper.

### References
[1] RARe: Retrieval Augmented Retrieval with In-context Examples, Tejaswi et al

[2] Contextual Document Embeddings, Morris et al

### Questions
1. In stage 2 (which as per my understanding is training on listwise ranking data), how exactly is the infonce loss implemented? 

Please see rest of questions in weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
