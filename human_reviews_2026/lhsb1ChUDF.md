# DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Deep graph models have achieved great success in network representation learning. However, their focus on pairwise relationships restricts their ability to learn pervasive higher-order interactions in real-world systems, which can be naturally modeled as hypergraphs.
To tackle this issue, Hypergraph Neural Networks (HNNs) have
garnered substantial attention in recent years. 
Despite the proposal of numerous HNNs, the absence of consistent experimental protocols and multi-dimensional empirical analysis impedes deeper understanding and further development of HNN research. While several toolkits for deep hypergraph learning (DHGL) have been introduced to facilitate algorithm evaluation, they provide only limited quantitative evaluation results and insufficient coverage of advanced algorithms, datasets, and benchmark tasks.
To fill the gap, we introduce DHG-Bench, the first comprehensive benchmark for HNNs.
Specifically, DHG-Bench systematically investigates the characteristics of HNNs in terms of four dimensions: effectiveness, efficiency, robustness, and fairness. 
We comprehensively evaluate 17 state-of-the-art HNN algorithms on 22 diverse datasets spanning node-, edge-, and graph-level tasks, under unified experimental settings.
Extensive experiments reveal both the strengths and limitations of existing algorithms, offering valuable insights and directions for future research. 
Furthermore, to facilitate reproducible research, we have developed an easy-to-use library for training and evaluating different HNN methods.
The DHG-Bench library is available at: 
https://github.com/Coco-Hut/DHG-Bench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a large benchmark for hypergraph neural networks (HNNs), consisting of 17 HNN methods across 22 datasets. The benchmark standardizes training/evaluation spanning node-, hyperedge-, and graph-level tasks, then analyses four dimensions: effectiveness, efficiency, robustness, and fairness. Empirically, no model dominates and models resist structural noise but are sensitive to feature/label noise.

### Strengths
1. The work organizes experiments around RQ1–RQ4 (effectiveness, efficiency, robustness, fairness), with clear task coverage across node/edge/graph; complete leaderboards appear in the appendix with standardized splits and operators.
2. Findings such as no single HNN dominates, accuracy–efficiency trade-offs, and fairness brittleness of message passing are likely to steer algorithm development (e.g., toward robust/efficient or debiased HNNs).

### Weaknesses
1. Hyperedge prediction uses random splits with mixed heuristic negative sampling. This ignores temporal/inductive drift and can create unrealistic candidate sets. The paper can be strengthened by adding (a) temporal splits (train earlier hyperedges, predict future ones), (b) inductive splits with disjoint node/hyperedge partitions, and (c) open-world negatives drawn from realistic candidate generators (size- and motif-conditioned but respecting time).
2. All tasks are supervised. Many real deployments use self-supervised pretraining or contrastive objectives. The benchmark can be strengthened by adding self-supervised learning tracks (node/hyperedge masking, motif prediction, contrastive pretraining) and evaluate fine-tuning across tasks to reflect modern practice.
3. The paper successfully demonstrates that HNN performance varies significantly across datasets (a key insight for RQ1), but it falls short of explaining why. The analysis largely stops at reporting performance rankings.

### Questions
1. Is performance sensitive to datasets with a few very large hyperedges versus many small ones?
2. How do different models handle nodes with very high vs. low degrees?
3. Given the results, what is the recommended decision-making process for a practitioner when selecting an HNN model for a new task?
4. What are the key trade-offs, supported by the benchmark's findings, that one should consider between performance, scalability, and specific data characteristics (e.g., homophily)?

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
The authors propose a comprehensive hypergraph neural network (HNN) benchmark, which is called DHG-Bench.

The benchmark implementation incorporates most of the representative HNNs and datasets.

Moreover, the authors evaluate the HNNs under diverse scenarios (e.g., classification, hyperedge prediction, and noisy cases).

### Strengths
S1. The authors provided a timely benchmark for hypergraph neural networks.

S2. The benchmark is comprehensive, in terms of both HNNs and downstream tasks.

S3. Most of the benchmark hypergraph datasets have been covered.

### Weaknesses
I do not have major criticisms of this work, but I have several suggestions.

**W1. [pip installation]** For now, I think one needs to download the GitHub repo to run the code. I think authors can improve the code to be easier to use, as in PyG (https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html).

**W2. [Label split]** While many HNNs use a 50/25/25 split for node classification, I personally think this ratio contains too many training nodes, compared to the common graph evaluation settings. Can the authors analyze the HNNs in more label-scarce (i.e., less training nodes) scenarios?

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DHG-Bench, a large-scale benchmark for Hypergraph Neural Networks (HNNs). The authors implement 17 existing methods and evaluate them on 22 datasets across node-, hyperedge-, and graph-level classification tasks. The evaluation covers four axes: effectiveness (accuracy), efficiency (runtime and memory), robustness (to structural, feature, and supervision perturbations), and fairness ($\Delta$DP and $\Delta$EO metrics). The authors also release an open-source library to ensure reproducibility. Overall, this is an ambitious and useful engineering effort that aims to unify evaluation practices in hypergraph learning.

### Strengths
1. **Comprehensive benchmark coverage.** DHG-Bench implements 17 HNN models and evaluates them on 22 datasets spanning node-, hyperedge-, and graph-level tasks.
2. **Multi-dimensional evaluation.** The benchmark goes beyond accuracy to assess efficiency, robustness, and fairness, providing a more holistic view of model behavior.
3. **Reproducibility focus.** The open-source code and datasets enable other researchers to replicate results and extend the benchmark.
4. **Insightful findings.** The experiments reveal important phenomena such as scalability bottlenecks, fairness gaps, and underperformance of HNNs on heterophilic datasets.

### Weaknesses
1. **Limited conceptual novelty.** The benchmark aggregates existing models but does not introduce new methodologies or theoretical advances.
2. **Insufficient graph-based baselines.** Only two GNNs are included, and simple but strong baselines (e.g., direction-aware GNNs) are missing, making it difficult to quantify the advantage of hypergraphs.
3. **Directed hypergraphs ignored.** The benchmark only evaluates undirected hypergraphs, omitting variants that model asymmetric or causal relations, i.e. directed hypergraphs. 
4. **Superficial scalability analysis.** Many methods fail with OOM errors, but mitigation strategies (e.g., mini-batching, sparse operations) are not reported. 
5. **Shallow heterophily treatment.** HNNs underperform MLPs on heterophilic datasets, but causes such as oversmoothing or feature mixing are not analyzed in detail.

### Questions
1. **Baseline selection.** What criteria determined the included baselines? Why were simple but strong baselines (MLPs, direction-aware GNNs) not systematically included?
2. **Directed hypergraphs and variants.** Do you plan to extend DHG-Bench to support directed hypergraphs, heterogeneous hyperedges, or temporal hypergraphs? If not, please justify. 
3. **OOM diagnostics.** For models that ran out of memory, which mitigation strategies were attempted (e.g., mini-batching, sparse matrices, mixed precision)? Can you report peak memory usage?
4. **Heterophily analysis.** Can you quantify heterophily per dataset, test hypotheses (e.g., oversmoothing, feature collapse), and relate results to prior literature on heterophilic GNN behavior?
5. **Fairness metrics.** For datasets lacking explicit sensitive attributes, how were $\Delta$DP/ta$EO computed? Were proxy attributes used, and how were they validated?

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
The paper introduces a benchmark for hypergraph representation learning, by evaluating 17 popular hypergraph networks on 22 datasets containing node-level, edge-level and hypergraph-level tasks. The evaluation is performed in terms of accuracy, efficiency, robustness and fairness. The paper also provide the code for the benchmark, representing a useful resource for evaluating future models.

### Strengths
- Hypergraph models suffer from inadequate evaluation, which slows down the advancement of the field. Moreover, most existing setups contain only node-level classification tasks. This paper represents a significant step forward in understanding the limitations of current models and provides a consistent, uniform framework for evaluating new models in a fair way.
- The paper points out a couple of limitations exhibited by current models, which represent good areas for future research. In particular, it shows how little progress has been made in hyperedge-level prediction, highlighting the need for more focused efforts in this area.
- Exploring additional metrics such as robustness and fairness is an important represents an important direction for advancing the field.

### Weaknesses
- The node classification setup used in the paper appears similar to that of ED-HNN. However, the reported results are noticeably lower. Is there any major difference in the training setup?
- I agree that structural robustness is an important metric. However, for models that explicitly take connectivity into account, not observing a drop in performance at a 90% perturbation ratio seems more like a negative result than a positive one. The paper presents robust performance across different perturbation levels as a desirable outcome, but to me, this suggests either that the dataset does not require higher-order processing or that the model does not properly incorporate structural information. I suggest that once the new metric is introduced, the authors also include a discussion on how a good hypergraph model is expected to behave under such conditions. ****
- It would be interesting to see more statistics on the datasets used in the experiments to highlight the extent to which they are representative of higher-order interactions. For example, it is known that PubMed has around 80% of its nodes isolated. Reporting statistics such as the number of isolated nodes, the number of nodes involved only in pairwise interactions, and the number of nodes participating in higher-order interactions would be particularly useful for node-level classification tasks, where the performance on isolated nodes would depend solely on the MLP component, regardless of the model used.

### Questions
Please see the Weaknesses section

### Soundness
3

### Presentation
4

### Contribution
3
