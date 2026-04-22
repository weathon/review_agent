# FineKB: Domain-Adaptive Issue Summarization and Cluster-Aware Retrieval for Support Knowledge Bases

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Retrieving relevant knowledge base (KB) articles for enterprise support cases is difficult due to the semantic mismatch between noisy, verbose case descriptions and concise KB content. We present FineKB, a domain-adaptive issue–summarization and cluster-aware retrieval framework that addresses this gap through (i) a finetuned LLM trained on teacher-generated pseudo-summaries to normalize heterogeneous case narratives, (ii) per-KB multi-centroid clustering that models the diverse sub-problems associated with each KB article, and (iii) a confidence-adaptive hybrid inference mechanism that augments high-confidence vector search with selective content lookup and LLM reasoning for ambiguous cases. At inference time, raw case text is embedded and matched against this summary-structured index, avoiding runtime summarization while improving alignment. Experiments on large-scale enterprise data show that FineKB achieves 65.39\% Recall@3, substantially outperforming KB-content dense retrieval (42.73\%). To support reproducible research on noisy-to-structured retrieval, we release FineKB-Vectors, a vectorized dataset containing case-summary and KB-article embeddings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes an LLM-based retrieval method for domain-specific knowledge bases. The retrieval method uses a fine-tuned LLM-based summarizer to generate summaries which are used for indexing and retrieval, and a cluster-aware indexing method is applied. The empirical evaluations examine the effeteness of the proposed retrieval method against baseline methods, as well as the performance of the fine-tuned summarizer.

### Strengths
1. The proposed retrieval method based on LLM-generated summaries is intuitive and shows better performance than the baselines.

2. The methodology description and the experimental design are clear-written.

### Weaknesses
1. The technical contribution of this work is limited, as the main contribution seems to be the introduction of the issue summaries for improving indexing effectiveness. More discussions should be provided regarding the significance of this innovation.

2. The empirical experiments are not comprehensive enough, which are based on only one dataset and limited to Llama-3.1-8B and Llama-3.3-70B. More datasets should be included in order to demonstrate the generalizable improvement of the proposed method. Similarly, more LLMs from different model families should be considered.

3. The characteristics of the used dataset are unclear, raising issues regarding the reproducibility. Will the dataset ("enterprise support cases") be released?

4. Comparisons with existing retrieval methods are lacking. Such as the classic BM25 method, or dense retrievers such as NV-Embed [1].

References

[1] Lee, Chankyu, et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." The Thirteenth International Conference on Learning Representations.
[1]

### Questions
Please see the Weaknesses section.

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
4

### Summary
The paper proposes a practical RAG-based enterprise KB pipeline with three components: (i) weakly supervised, one-sentence ticket summarization to normalize noisy cases; (ii) domain-adaptive summarization using SFT/QLoRA to improve faithfulness and task fit; and (iii) per-KB, cluster-aware multi-centroid indexing to capture sub-topics within each article. The authors report large retrieval gains (e.g., Recall@3 rising from under 24% with KB-only indexing to over 66% with summary-based cluster-aware indexing) and show that 8B adapter-tuned summarizers can approach the retrieval effectiveness of a much larger 70B model, suggesting a cost-efficient path to deployment.

### Strengths
* The paper addresses a clearly defined retrieval mismatch between noisy, user-authored tickets and curated KB prose, with a transparent offline/online system decomposition that practitioners can follow.
* The paper is not limited to large language models; it systematically evaluates small language model (SLM) adapters (e.g., 8B with QLoRA) for the summarization stage, demonstrating competitive retrieval effectiveness at materially lower compute and cost, which is critical for enterprise deployment.

### Weaknesses
* The paper does not report latency, throughput, or cost for the end-to-end pipeline, which are essential for real-world adoption and for weighing trade-offs versus simpler baselines.
* The core idea of the method, summarize then cluster/index, follows established enterprise RAG practice. Prior work explores similar idea of “generate proxy/summary then retrieve” or multi-signal retrieval such as HyDE [1],  Aragog [2], entity-aware summaries for retrieval alignment [3], and query-context summarized signals for retrieval in sponsored search [4], The paper’s comparisons and ablations might not be sufficient to demonstrate effectiveness over these families at matched budgets settings.
* The evaluation of the summarization component is limited to ROUGE/BERTScore, which may not capture faithfulness, usefulness for retrieval, or end-task impact. The paper should incorporate LLM-as-judge protocols [5] with explicit rubrics on faithfulness/utility, and extrinsic metrics that reflect downstream retrieval quality

[1] Gao, Luyu, Xueguang Ma, Jimmy Lin, and Jamie Callan. "Precise zero-shot dense retrieval without relevance labels." In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1762-1777. 2023.

[2] Eibich, Matouš, Shivay Nagpal, and Alexander Fred-Ojala. "Aragog: Advanced rag output grading." arXiv preprint arXiv:2404.01037 (2024).

[3] Liang, Xiao, Xinyu Hu, Simiao Zuo, Jimi He, Yu Wang, Victor Ye Dong, Yeyun Gong et al. "What You See Is What You Get: Entity-Aware Summarization for Reliable Sponsored Search." In Neurips Safe Generative AI Workshop 2024.

[4] Mohankumar, Akash Kumar, Gagan Madan, and Amit Singh. "Improving Retrieval in Sponsored Search by Leveraging Query Context Signals." arXiv preprint arXiv:2407.14346 (2024).

[5] Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, et al. 2023. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.” arXiv:2306.05685.

### Questions
* Could you provide sensitivity analyses for the clustering hyperparameters (e.g., distance threshold and minimum cluster size) and show how retrieval metrics vary across a reasonable range?
* Could you report end-to-end latency and cost for summarization, embedding, and retrieval?

### Soundness
3

### Presentation
2

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
This paper presents FineKB, a domain-adaptive framework for enterprise knowledge base retrieval ￼. FineKB builds pseudo-labeled issue summaries from historical support logs, fine-tunes compact LLMs with parameter-efficient methods (SFT, QLoRA), and introduces a cluster-aware retrieval index that represents each KB article with multiple centroids to capture sub-problems. On an ad-hoc dataset, FineKB is shown to be effective. Overall, the work has very limited value for larger ICLR community, due to its applied & narrow scope.

### Strengths
- Paper is well written and motivated. 
- Experimental code is included with submission, allowing reviewers to better verify claims and understand proposed approach.

### Weaknesses
- The proposed approach is largely an ad-hoc application without much contextualization in existing literature. For example, when it comes to generating compressed text representations to improve retrieval, there has been a lot of work since seminal PAQ work from [Lewis et al \(2021\)](https://arxiv.org/abs/2102.07033). 
- The method is evaluated on a proprietary, in-house dataset. As the dataset is not public, the amount of details included in the paper are too sparse to judge effectiveness of method, and no test of the proposed method on public datasets is included. 
- in the era of LLM, ROUGE/BERTscore has been shown to penalize better summarization model [(Goyal et al 2023)](https://arxiv.org/abs/2209.12356). Using techniques that measure recall in summaries, e.g. FactScore [\(Min et al 2023\)](https://arxiv.org/abs/2305.14251) would be more appropriate.
- Case summary in Table 2 barely improves over using text index. Difference might not be statistically significant.

### Questions
Minor typos: 
- use \citep instead of \cite -> "Faheem et al. (2025)" should be "(Faheem et al., 2025)"

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents FineKB, a system that 1) does peft fine tuning on LLMs on LLM generated summaries, and 2) uses clusterwise retrieval to improve case-to-KB matching accuracy.  Experiments on a private dataset show reasonable performance.

### Strengths
The paper addresses an understudied practical challenge in enterprise support, ablating some of their choices and showing that they can reach reasonable performance even with smaller models.

### Weaknesses
My main concern is that the paper does not present meaningful novelty, uses a simple combination of techniques, does not compare to any existing methods or baselines (apart from the ablations on their method), (Most existing RAG methods can be used/modified to support the use case), and experiments are done on a proprietary dataset that is unreleased so there is no way to compare the performance of the method to any future or previous methods. I would suggest to the authors to release a dataset (that can be constructed artificially to avoid any leakage proprietary data).

### Questions
How does the method compare to other common RAG methods (modified for the use case)?
Can the authors create and release a dataset that matches the attributes that the would want in a successful case to KB matching system?

### Soundness
1

### Presentation
2

### Contribution
1
