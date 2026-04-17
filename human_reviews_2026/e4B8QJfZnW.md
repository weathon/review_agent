# Clustering Improves Differentially Private Inference

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Differentially private (DP) language model inference is an approach for generating private synthetic text. A sensitive input example is used to prompt an off-the-shelf large language model (LLM) to produce a similar example. Multiple examples can be aggregated together to formally satisfy the DP guarantee.

Prior work creates inference batches by sampling sensitive inputs uniformly at random. We show that uniform sampling degrades the quality of privately generated text, especially when the sensitive examples concern heterogeneous topics.

We remedy this problem by clustering the input data before selecting inference batches. Next, we observe that clustering also leads to more similar next-token predictions across inferences. We use this insight to introduce a new algorithm that aggregates next token statistics by privately computing medians instead of averages. This approach leverages the fact that the median has decreased local sensitivity when next token predictions are similar, allowing us to state a data-dependent and ex-post DP guarantee about the privacy properties of this algorithm. Finally, we demonstrate improvements in terms of representativeness metrics (e.g., MAUVE) as well as downstream task performance. We show that our method produces high-quality synthetic data, at significantly lower privacy cost, than a previous state-of-the-art method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studied the problem of private synthetic data generation and proposed an optimized version of private prediction based solution. Specifically, the authors identify that uniformly sampling ICL examples can lead to degraded utility due to heterogeneous distribution within the sample batch. To encounter this problem, the author proposed a DP clustering method by first calculating k cluster centers using public data, and then DP select the top-k' centers upon private data to avoid skewed distribution. The authors also proposed using median aggregation rather than mean aggregation to lower the sensitivity for clustering-based batching strategy. The experiment results shows improved utility and privacy cost.

### Strengths
- The paper is well-written and easy to follow.
- Applying DP for private synthetic data generation is an important direction.
- Extensive experiments on clustering and empirical private estimates have been conducted.

### Weaknesses
- The contribution of this paper is somewhat limited. The main utility improvement comes from the clustering-based batching for DP sampling. In [1], the authors have already discussed the choice of non-uniform batching, and assign prompts with the same label to the same batch. This work instead proposed to cluster based on the embedding similarity. I wonder if the authors could better clarify the difference.
- According to the ablation studies, the utility improvement mainly comes from using PT models and add more ICL samples. The clustering contributes marginal utility improvement according to Table 3 and Table 4.
- The selection of the cluster number $k$ is not clearly stated. Is it data dependent? Besides, does the $\epsilon$ in the DP top-$k'$ selection has significant impacts? That is,  is the top-$k'$ clusters consistently being selected? According to Figure 2, the top-100 public centers almost includes all the samples.

[1]: Private prediction for large-scale synthetic text generation. https://arxiv.org/pdf/2407.12108

### Questions
- The authors discussed the comparison against PE methods. I wonder if there will be similar improvements if PE methods refer to PT models rather than IT models.
- Regarding the explanation of why median aggregation yields a smaller sensitivity, the authors wrote ' if Z contains at least 3 identical vectors, then the local sensitivity of median(Z) is zero.'. I have two questions : 1) in this case, add/remove a sample causes no impact to the algorithm, which should somehow degrade the utility; 2) this example does not make sense to me. Since the batch is sampled from a cluster, which implies a similar embedding distribution, there should be few outliers in the batch. In this case, the sensitivity for mean aggregation (theoretically 2C/B) should be similar to that of median aggregation (2C).
- The refined understanding of privacy risk is promising. Could the authors elaborate on the criteria for private/public tokens and how the data-dependent accounting is designed?

### Soundness
3

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
3

### Summary
This paper studies private synthetic data generation via LLM inference. Prior work follows a subsample-and-aggregate approach: split the private data uniformly at random into batches, query the LLM to get next-token logits for each record (non-privately), and then privately aggregate those logits to sample tokens. While this method can yield good downstream task accuracy, the synthetic text may be a poor representative of the original distribution because uniform batching mixes heterogeneous examples.

The authors propose batching similar examples together to improve representativeness and reduce privacy cost. They first compute public cluster centers using a non-private dataset, then privately rebalance by adding DP noise to cluster counts and keeping the top-$k$ centers to ensure that the clusters are not too small while spending a small privacy budget. Then, for each batch they compute logits for all seeds in the batch, apply clipping, and aggregate by the median rather than the mean. They analyze this with a data-dependent DP guarantee: when a batch’s logits are concentrated (small “median gaps”), the local sensitivity is smaller, which yields a lower $\epsilon$. Clustering makes batches more homogeneous, so the realized privacy loss can be lower than worst-case bounds.

Empirically, the method improves both representativeness metrics (e.g., MAUVE) and downstream accuracy compared to prior DP inference baselines. However, the overall privacy guarantee depends on the data and outputs.

### Strengths
- The paper is well written and easy to follow. 

- Their proposed methods is simple but effective. It does a small change to prior DP-inference pipelines (replacing uniform batching and mean aggregation with cluster-informed batching and median aggregation) that is easy to adopt. The proposed method produces more representative synthetic data (higher MAUVE) and strong downstream accuracy in comparison with previous methods.

### Weaknesses
- The privacy guarantee is ex-post and data-dependent, which is weaker. For example, releasing the realized $\epsilon$ may leak information about the private data’s “median gaps.”

- MAUVE is lower on Yelp, which is farther from DBPedia, suggesting that public–private distributional closeness matters; performance may degrade on niche private datasets.

### Questions
Would applying PTR/smooth sensitivity keeps the advantage of the method, while also give a data-independent privacy guarantee?
How does the number of clusters affect the results?

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
4

### Summary
Summary: This paper addresses the problem that random sampling leads to heterogeneous batches and degrades data quality when generating synthetic text via differentially private (DP) language model inference. It proposes an improved method combining clustering and median aggregation. Core contributions include: 1) Constructing homogeneous batches by pre-clustering sensitive data to solve the heterogeneity issue of random sampling; 2) Designing a median-based logit aggregation algorithm that leverages the similarity of prediction results under homogeneous data to reduce local sensitivity, achieving a data-dependent ex-post DP guarantee; 3) Validating on datasets such as AGNews and Yelp that the method significantly outperforms existing baselines in MAUVE (distributional similarity) and downstream task accuracy with lower privacy cost.

### Strengths
1.Novel use of clustering as a preprocessing step to improve DP inference quality; novel median-based aggregation with a tailored data-dependent ex-post privacy analysis.
2.The paper pairs formal analysis (Algorithm 1, Theorem 2 and proof sketch) with practical experiments and empirical privacy checks. Results are presented across multiple datasets and ε settings.
3.The paper has a well-structured hierarchy, with coherent logic from problem formulation, method design to experimental verification. Explanations of core concepts (such as local sensitivity, data-dependent ex-post DP) are concise and clear.
4.It solves the key problem of "imbalance between quantity and representativeness" in DP synthetic data, especially suitable for sensitive heterogeneous data scenarios such as medical records. It promotes the practicalization of DP inference technology.

### Weaknesses
1.It relies on public datasets (DBPedia) to construct cluster centers. If the distribution of public data differs significantly from that of sensitive data, cluster imbalance may still occur; DP clustering methods perform poorly on high-dimensional text embeddings.
2.There is a lack of further attack experiments (such as membership inference/extraction for a small number of anomalous samples).
3. It fails to conduct a fair comparison with the latest DP synthetic data methods (such as API-based private evolution technology) under the same prompt and model settings.
4. The sensitivity analysis of hyperparameters such as the number of clusters and embedding models is insufficient, and no general criteria for optimal parameter selection are clarified.
5. Although the data-dependent ex-post DP guarantee is more accurate, the privacy cost needs to be calculated after generation, making it impossible to determine the privacy budget in advance. This increases the complexity of practical deployment.

### Questions
1.How was the initial public k (e.g., 1000 → rebalanced to 60/80) chosen? Any sensitivity/selection rule? 
2. How should multiple per-output data-dependent ε values be combined into a final released ε? Which composition theorem is applied in practice?
3. This method is not compared with some of the latest baselines: Aug-PE[1], DP-fusion[2], DP-SynRAG[3], INVISIBLEINK[4].
4. The privacy cost of median aggregation depends on generation results. In practical deployment, how to balance the needs of "ex-post calculation" and "real-time generation"?

[1] Differentially private synthetic data via foundation model APIs 2: Text
[2] DP-fusion: Token-level differentially private inference for large language models
[3] Differentially private synthetic text generation for retrieval-augmented generation (RAG)
[4] InvisibleInk: High-utility and low-cost text generation with differential privacy

### Soundness
2

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
The paper investigates the quality of synthetic data generated via differentially private (DP) inference methods, where multiple LLM outputs seeded by different users’ data are aggregated to ensure privacy. The authors argue that heterogeneity among seed examples in each batch can harm the representativeness of the synthetic data. To address this, they propose a privacy-preserving pre-clustering approach that groups similar examples before aggregation and introduce a median-based aggregation (instead of averaging). Their experiments show improved MAUVE and accuracy scores compared to considered baseline.

### Strengths
- The paper addresses a practical limitation of existing DP inference methods by considering the effect of data heterogeneity.

- Proposes a simple and intuitive modification (clustering and median-based aggregation) that can be easily integrated into existing frameworks.

### Weaknesses
- I find the argument in line 52 to be flawed. I think it incorrectly conflates privacy guarantees with data representativeness or quality. Differential privacy doesn’t require the aggregated response to be “representative” of the data; it only ensures that the presence or absence of any individual record (seed example) doesn’t significantly affect the output distribution.

- Table 3 is difficult to interpret because the reported ε values are not directly comparable. As far as I understand, the methods are based on different privacy formulations i.e data-dependent ex-post differential privacy versus approximate differential privacy, which provide distinct guarantees and should not be contrasted using the same ε metric.

### Questions
- The process for selecting top-k cluster centers and reassigning seeds is unclear [line 244]

- Would using median aggregation alone (without clustering) yield similar improvements?

### Soundness
2

### Presentation
2

### Contribution
2
