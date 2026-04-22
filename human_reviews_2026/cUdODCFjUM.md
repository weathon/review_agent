# A Dense Subset Index for Collective Query Coverage

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
In traditional information retrieval, corpus items compete with each other to occupy top ranks in response to a query.  In contrast, in many recent retrieval scenarios associated with complex, multi-hop question answering or text-to-SQL, items are not self-complete: they must instead collaborate, i.e., information from multiple items must be combined to respond to the query. In the context of modern dense retrieval, this need translates into finding a small collection of corpus items whose contextual word vectors collectively cover the contextual word vectors of the query. The central challenge is to retrieve a near-optimal collection of covering items in time that is sublinear in corpus size. By establishing coverage as a submodular objective, we enable successive dense index probes to quickly assemble an item collection that achieves near-optimal coverage.  Successive query vectors are iteratively `edited', and the dense index is built using random projections of a novel, lifted dense vector space. Beyond rigorous theoretical guarantees, we report on a scalable implementation of this new form of vector database. Extensive experiments establish the empirical success of DISCo, in terms of the best coverage vs. query latency tradeoffs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces DISCO, a dense retrieval method that selects a subset of documents optimised for query token coverage. Unlike typical top-K retrieval, DISCO selects documents whose embeddings collectively maximise coverage over the query’s token embeddings. The method formalises this as a monotone submodular objective, allowing for a greedy approximation. It then proposes a practical approximation using random projections and a lifted representation, making the approach compatible with ANN search. The authors implement a multi-vector IVF index and show strong empirical performance in terms of coverage and latency on multiple datasets.

### Strengths
1. Clear motivation: The paper addresses limitations of independent ranking in multi-hop and multi-evidence tasks.

2. Sound theoretical foundation: The coverage objective is well-defined, submodular, and optimisable via greedy selection.

3. Efficient approximation: The use of lifted representations and random projections to estimate marginal gains is novel and effective.

4. Strong empirical results: On benchmarks like HotpotQA and FEVER, DISCO demonstrates significant latency gains and higher coverage compared to both greedy and top-K baselines.

5. Implementation quality: The multi-stage index is well engineered and extensively ablated.

### Weaknesses
1. Lack of downstream evaluation: The paper does not assess end-to-end improvements in downstream tasks such as QA or claim verification. All results focus on coverage and latency.

2. Limited comparative breadth: The baseline comparisons could be broadened to include recent token-aware or set-aware retrieval methods.

3. Unvalidated approximation behaviour: While the greedy coverage algorithm has known guarantees, the practical approximation introduced by DISCO is not empirically analysed against true marginal gains.

### Questions
1. Have you tested DISCO in a full downstream setting, such as QA or claim verification, where it serves as the first-stage retriever? It would be useful to know whether the observed coverage improvements actually lead to gains in final task metrics like accuracy or F1.

2. The current baseline set is helpful, but it would be good to understand how DISCO compares to more recent retrieval methods that consider token-level signals or diversity—such as ColBERTv2, GRIP, or SPLADE-style models. Even a brief discussion of these alternatives would help situate the contribution.

3. The approximation strategy for estimating marginal gains is clearly explained, but it would be helpful to see some empirical check on how closely it matches the true greedy gain. For instance, do the approximated and true gains result in similar document rankings or coverage values?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper correctly recognizes that many ranking algorithms rank items independently of one another, so that important aspects of query coverage may be missed if only the top K items are retrieved. This is absolutely true and further study on this problem is definitely worthwhile, especially from the perspective of modern dense retrieval.

Unfortunately, this work appears to have been conducted in a bit of a vacuum, without any clear consideration of the huge volume of prior work in this area, especially related work on learning to rank for web search. The section on monotonicity, submodularity, and greedy maximization could have be lifted without change from a paper appearing 10 or more years ago. I included a bunch of examples below.

I also have concerns about the evaluation. For coverage, only HotpotQA has gold labels, which limits the evaluation. However, there are various methods for employing LLMs to generate pseudo-labels. While it's not exactly what you want Ragnarok (https://arxiv.org/abs/2406.16828) can probably be adapted to extract nuggets for coverage.

MAP is very strange measure to use on these collection. NDCG and MRR are more standard.

Rodrygo L.T. Santos, Craig Macdonald, and Iadh Ounis. 2010. Exploiting query reformulations for web search result diversification. In Proceedings of the 19th international conference on World wide web (WWW '10). Association for Computing Machinery, New York, NY, USA, 881–890. https://doi.org/10.1145/1772690.1772780

Cheng Xiang Zhai, William W. Cohen, and John Lafferty. 2003. Beyond independent relevance: methods and evaluation metrics for subtopic retrieval. In Proceedings of the 26th annual international ACM SIGIR conference on Research and development in informaion retrieval (SIGIR '03). Association for Computing Machinery, New York, NY, USA, 10–17. https://doi.org/10.1145/860435.860440

Jun Xu, Long Xia, Yanyan Lan, Jiafeng Guo, and Xueqi Cheng. 2017. Directly Optimize Diversity Evaluation Measures: A New Approach to Search Result Diversification. ACM Trans. Intell. Syst. Technol. 8, 3, Article 41 (May 2017), 26 pages. https://doi.org/10.1145/2983921

Learning for Search Result Diversification. Yadong Zhu Yanyan Lan Jiafeng Guo Xueqi Cheng Shuzi Niu

These are just random papers that popped to mind. I think there was some good work by Olivier Chapelle but all I can find is this workshop paper: Paul N. Bennett, Ben Carterette, Olivier Chapelle, and Thorsten Joachims. 2008. Beyond binary relevance: preferences, diversity, and set-level judgments. SIGIR Forum 42, 2 (December 2008), 53–58. https://doi.org/10.1145/1480506.1480516

*Note that the last one is 2008*

### Strengths
The topic is important, and the theory seems correct. I think there are novel aspects, and certainly a version of this paper should be published at some point. 

The focus on efficiency is good to see.

### Weaknesses
What I said in the summary: There's a huge thread of similar work that has been ignored. The evaluation is not appropriate to the problem.

### Questions
Can you clarify the connection to past work?

Can you defend the limitations in the evaluation raised in the summary?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel reformulation of the classical task of information retrieval. While traditional IR is optimized for selecting a single "best" document out of a corpus of choices, the authors note that these techniques are ill-suited for modern, reasoning intensive retrieval tasks such as multi-hop question answering. 

To address this challenge, the authors propose to reframe retrieval as a set coverage problem and cast the task of finding the item that maximizes the marginal gain of query coverage as a multi-vector retrieval task. The authors implement their proposed pipeline in a system called DISCo which achieves significantly improved query-latency tradeoff metrics over a host of established retrieval baselines.

### Strengths
The paper presents a novel retrieval algorithm that achieves improved performance on reasoning-intensive retrieval tasks by creatively combining submodular optimization with multi-vector retrieval. While none of the components of the authors' proposed architecture is entirely novel, the combination of these techniques is original and appears to achieve state-of-the-art performance across a number of standard retrieval baselines in this area. The authors' proposed technique is also principled and they also provide a number of theoretical guarantees. The paper is also very well-written and motivates the problem well.

### Weaknesses
1. While the baselines considered in the experiments is quite thorough, the claims in the paper might be better supported by considering additional retrieval benchmarks specifically tailored towards challenging retrieval-intensive tasks, such as the recently introduced [Bright Benchmark](https://arxiv.org/pdf/2407.12883). Evaluating on benchmarks specifically designed for reasoning-intensive tasks (as opposed to more generic retrieval like BEIR) might be a more natural fit for this paper and might help us better understanding the strengths and limitations of the DISCo method. 

2. The paper is well-written but I found it a bit hard to understand how all the pieces fit together on a first read through the paper. The authors might want to consider adding a schematic diagram or conceptual figure to aid in communicating their methodology. 

3. I think the main body of the paper could benefit from a dedicated section discussing the relevant related work (especially in introducing the baseline methods evaluated later in the experimental section). 

4. The authors could also make their theoretical contributions more clear by perhaps introducing informal statements of the theorems they prove instead of relegating all of this content to the appendix (which readers may not read and thus never find).

### Questions
1. Can you provide more clarification on the "gold set" of hotpotqa? Why is this different from the other benchmark datasets used in the experiments section?

2. Can you consider evaluating DISCo on the [Bright Benchmark](https://arxiv.org/pdf/2407.12883) and possibly other related reasoning-intensive retrieval benchmarks?

3. Do existing methods handle updates to the corpus efficiently? How would you assess this limitation of your method in relation to the existing literature?

### Soundness
3

### Presentation
3

### Contribution
3
