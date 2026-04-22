# Fixed Aggregation Features Can Rival GNNs

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Graph neural networks (GNNs) are widely believed to excel at node representation learning through trainable neighborhood aggregations. We challenge this view by introducing Fixed Aggregation Features (FAFs), a training-free approach that transforms graph learning tasks into tabular problems. This simple shift enables the use of well-established tabular methods, offering strong interpretability and the flexibility to deploy diverse classifiers. Across 14 benchmarks, well-tuned multilayer perceptrons trained on FAFs rival or outperform state-of-the-art GNNs and graph transformers on 12 tasks -- often using only mean aggregation. The only exceptions are the Roman Empire and Minesweeper datasets, which typically require unusually deep GNNs. To explain the theoretical possibility of non-trainable aggregations, we connect our findings to Kolmogorov–Arnold representations and discuss when mean aggregation can be sufficient. In conclusion, our results call for (i) richer benchmarks benefiting from learning diverse neighborhood aggregations, (ii) strong tabular baselines as standard, and (iii) employing and advancing tabular models for graph data to gain new insights into related tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents arguments in favor of using feature-engineering based on multiple, fixed node feature aggregations, rather than having a GNN do the heavy lifting every time. It Is argued that working with tabular ML methods on pre-processed node features offers more interpretability and better training stability than using GNN and that this is supported by empirical experiments on 14 datasets. In addition, the authors introduce a theoretical contribution based on the KAT to motivate the particular feature extraction strategy.

### Strengths
Challenging the status quo is always a good idea. This paper does an extremely well job at summarizing prior work that has explored ways of making GNNs simpler, untrained, or heterophily-specific, and then it proposes a solution that seems inspired by multi-aggregation techniques like PNA and concatenation of features like jumping-knowledge networks (JK-Net, Xu et al.). I would like to see more papers devoting so much attention to prior work before proposing a contribution. In this sense, I believe that motivating the FAF scheme by revisiting the KAT under the lens of neighborhood aggregation is an extremely interesting and novel direction. The message-passing architecture is not easy to control and simpler but still effective solutions may even help us reason about different architectural schemes that are more effective than what we use today.

### Weaknesses
There are several aspects in the paper that in my opinion require improvements before being accepted at an A* conference. 

I think the introduction might benefit from stronger justifications when motivating the proposed technique. For instance, the second paragraph does not seem to necessarily raise the basic question in line 42, rather it could motivate any architectural study of GNNs as a whole. It would be better to find a stronger argumentation that is specific to the aggregation.

The introduction is also quite difficult to read due to some ambiguities. It is often unclear whether the authors mean “graph convolution” or simply the “aggregation operator” when referring to “learning the aggregation”. A statement like “we find that untrained aggregators yield useful features” is true for both GNNs and FAF if one refers to the permutation-invariant operator over neighbors, so I would like to suggest that the authors take more care in disambiguating these cases in the text. 

FAF reminds me of an untrained PNA followed by an MLP classifier. I would like to disagree, however, with the statement that FAF can bring more interpretability than current methods that provide an interpretation in the form of a subgraph. The example in Section 3.1 is very ad-hoc, and it does not transfer to a more convoluted scenario where features are continuous and their mixing with different aggregators could make no sense. The experiments in the paper in this direction are not convincing to me, unfortunately.

Section 4 presents interesting theoretical results, which I could not check in detail due to the reviewing workload amidst my other commitments. Apologies for that. I think it is a fresh perspective of neighborhood that I had not heard of. At the same time, I believe there is a big gap between the theoretical result and the empirical experiments, because of the use of non-injective functions invalidates all the theoretical arguments. Please let me know if I misunderstood something here. 

In this sense, the answer to the question of Section 4’s title seems to be a sound “yes”: we need to learn aggregators because we have no way of learning injective functions by repeated application of known fixed aggregators. The theoretical arguments, in my opinion, would make for a very nice paper if analyzed and expanded further, and the same could be said for the FAF method. I understand the reasoning behind the current paper’s structure, but in my opinion the paper does not do good justice to neither contribution, since FAF takes too big of a step compared to the theoretical arguments, which are missing some more qualitative analysis.

I have seen a very interesting reference to GESN and following works. I am aware of that research line: GESN variants have achieved very good results on at least subset of these datasets. Being correctly mentioned as related work, in the sense that the untrained reservoir of GESN acts as an “alternative FAF”, though not supported by theory, I am wondering why the authors did not compare empirically with this family of methods.

The most concerning issue on my end remains the empirical evaluation. When one introduces a new class of models, parametrized by a set of hyper-parameters, the usual way of evaluating the empirical risk is by running a model selection, selecting the best hyper-parameters on a validation set, and then evaluating the best configuration (assuming only one in a hold-out data split) on the test set. What I see in Table 1 is instead an instance of hyper-parameter tuning of the hyperparameter R (the set of aggregators to use) on the **test set**. There should be a single line for the class of FAF models in each of these tables, representing the fact that the set R was tuned separately on the validation set for each dataset together with the other hyper-parameters. This is a grave mistake in my understanding. Similarly, all ablation analyses should refer to the validation set performance without looking at the test set, but it does not seem to be the case.

Overall, I really like the perspective of the authors and I encourage them to revise the paper, possibly following some of the suggestions in this review. At the same time, I do not feel I can recommend acceptance of the paper in its current shape.

### Questions
Questions:
- I would like to ask the authors where in the paper they supported their statement of line 86 that learnability and numeric stability govern practical success in addition to expressiveness, and why the authors think that this was not clear in the past.
- Can you easily extent your theoretical arguments to F>1, or are they limited to F=1?
- The first two suggestions for future work (lines 358-9) look a bit underdeveloped and generic with respect to the authors’ contribution. Could you elaborate a bit better why they are relevant in this context?
- Potential typo: $\mathcal{X}$ may be undefined in the paper.

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
5

### Summary
The paper shows that a combination of non-learnable feature aggregation a an MLP can provide performance competitive with GNNs on many classic datasets for node classification. First, the paper proposes concatenating features obtained from neighborhood aggregations with different radius with the original node features to essentially convert a node classification task to a tabular classification task. Then, the paper investigates some theoretical aspects of such aggregations, showing that it is possible to design aggregations that preserve all neighborhood information (but, as the paper admits, such aggregations are not very practical). Then the paper conducts experiments with simpler aggregations and shows that MLPs on top of such aggregations can often rival GNNs.

### Strengths
- The main idea is simple but useful, its empirical performance can sometimes be quite impressive.
- Experiments are conducted on a vast range of datasets, strong baseline models with adequate hyperparameter search spaces are used.
- The paper raises timely questions regarding the adequacy of standard node classification benchmarks for the evaluation of complex models.

### Weaknesses
- The interpretability example with the minesweeper dataset (section 3.1) is wrong due to a misunderstanding of what the node features are. As described in [1], where the dataset was proposed, the node features use one-hot encoding for the number of neighboring mines, not binary encoding (see page 7 of [1], the Minesweeper paragraph). Due to this mistake, the explanation that the authors provide for how the model uses the features is entirely wrong. I do not consider this a serious issue, as it is just a minor example that does not affect the main points of the paper, but it needs to be fixed.

- The theoretical contributions seem not particularly interesting. First, Section 4.1 relies on the assumption of feature orthogonality, which is not very realistic. Even bag-of-words features are not orthogonal (and bag-of-words is an extremely outdated technique in 2025, but that is the problem of standard graph ML benchmarks, not of the current paper), and other feature types are typically even further from being orthogonal. Then, section 4.2 proposes a theoretical construction that, as the authors themselves admit (which is commendable), is not very practical. This leads to the theoretical sections being rather disconnected from the experimental sections. This is partially alleviated by some interesting discussions in Section 4.3, but they are not fleshed out enough in my opinion. I suggest shortening Section 4.1 and giving more space to the discussions in Section 4.3 (perhaps by providing more evidence for the hypotheses) to improve the paper.

- It is a strong point of the paper that it uses a lot of datasets for experiments and evaluates improved and well-tuned GNNs from [2] rather than weaker models that are often used as baselines in other works. However, this raises the question: if the authors use the codebase of [2] and also use almost the same hyperparameter search space, why are the reported results sometimes significantly different than those in [2]? For example, on the cora dataset, the reported result for GCN is 81.28, while [2] reports 85.10 (note that a similar results would make GCN rather than FAF4 the strongest model on cora in the current paper). There are similar discrepancies for some of the other datasets. What is the reason for them?

I am willing to raise my score if my concerns are addressed.



[1] A critical look at the evaluation of GNNs under heterophily: Are we really making progress? (ICLR 2023)

[2] Classic GNNs are strong baselines: Reassessing GNNs for node classification (NeurIPS 2024)

### Questions
See weaknesses.

Some other suggestions for paper improvement:

- The paper does not discuss efficiency at all, but it could potentially be another strong point of the FAF approach: precomputing aggregations once and then training MLPs on top of them is much faster than training GNNs (graph aggregation is typically the slowest operation in GNNs). I suggest discussing it and possibly even providing training times.

- The paper mentions a couple times that its results imply that the current benchmarks used in graph machine learning are inadequate for evaluating complex models. This could be discussed more and positioned within the recent line of works that also discuss and/or address this issue. Specifically, [3-5] discuss the problems with current benchmarks ([3] is briefly mentioned in the current work), and [5] proposes better datasets for node property prediction. Note that [5] additionally uses neighborhood feature aggregation (NFA) mechanism which is similar to a 1-hop version of FAF, and also shows that it can be a strong baseline. Even more recently, a concurrent work [6] also proposes better benchmarks for graph machine learning.

[3] Graph learning will lose relevance due to poor benchmarks (ICML 2025)

[4] No Metric to Rule Them All: Toward Principled Evaluations of Graph-Learning Datasets (ICML 2025)

[5] GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data (NeurIPS 2025)

[6] GraphBench: Next-generation graph learning benchmarking

### Soundness
3

### Presentation
2

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
This work presents Fixed Aggregation Features, a simple technique that combines neighborhood aggregations of different types (e.g., mean, max, std), applies them to neighbors at different hops, concatenates the obtained information with the original node features and trains an MLP on top of these representations. The experiments show that such a technique can provide results comparable to or even better than standard message passing networks when measured on classic graph ML datasets. Thus, the work introduces a previously overlooked class of strong graph ML baselines and questions the relevance of classic citation and co-authorship networks with bag-of-word node features for evaluating advanced graph ML models.

### Strengths
1. An introduction of non-learnable multi-hop feature aggregation as a preprocessing step is very simple, intuitive and practical approach to augment the original node features with graph-based information.

2. A discussion of the problems in optimization procedure of GNNs and theoretical properties of FAF, which also explains why a standard MLP on top of simple graph-based aggregations can perform on par with standard GNNs.

3. An extensive empirical study showing that FAF enables to achieve nearly the same results as GNNs, together with ablation study showing that even a single aggregation type in FAF can be sufficient for some datasets.

4. A number of additional experiments showing that theoretically lossless Kolmogorov-Arnold aggregations are not so effective in practice, while more simple aggregations like mean or max should be preferred for constructing FAF.

### Weaknesses
As the main weakness of current version, I see the choice of graph datasets for experiments. There is a recently introduced GraphLand benchmark [1] that provides both classification and regression tasks from industrial applications, includes both homophilous and heterophilous graph datasets, and contains rich heterogeneous tabular node features. Moreover, this work introduces Neighborhood Feature Aggregation (NFA) that seems to be a specific instance of FAF using mean, max and sum aggregations over 1-hop neighborhood. It might be very relevant for this particular study and thus should be discussed as related work.

It is very interesting to see whether the observations about the importance of the closest neighborhood hold and how the performance of simple MLP on top of FAF representations transfers to GraphLand datasets. I admit that this benchmark appeared very recently, but I would highly recommend to include the experiments using at least the RL (random low) data split, as it could help the authors not only investigate their hypothesis regarding the need for more relevant graph datasets, but also significantly strengthen their empirical study in general. If the authors manage to provide such additional results, I am ready to increase my score.

[1] GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data, NeurIPS 2025

### Questions
1. A couple of comments regarding the Theorem 1 and its proof in Appendix A.2.

- I am not sure that the term "unique function" in the formulation of this theorem is clear for me. As I understand, "uniqueness" means *preserving the information about elements of the original multiset and making it possible to know how many elements in a multiset have a particular feature value*. If this is true, I would ask the authors to expand the formulation and make it more explicit.

- I also feel that some explaining comments about decomposing any multiset function $f$ in the mentioned form are missing. As I understand, any multiset function $f$ depends on the counters of unique elements observed in it. Since we preserve the whole information about multiset under the transformation $\Phi$, we can restore these counters by using $\Phi^{-1}$ and then apply the desired $f$ to them. If the reasoning is correct, I would ask the authors to add such comments in the proof of this theorem.

- There seems to be a typo in the proof — it should be $\mathbb{R}^{n_f}$ instead of $\mathbb{R}^n_f$.

2. Can the authors explain why they obtain lower metrics for GNN baselines than those presented in [2], if they use the same hyperparameter search space?

[2] Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification, NeurIPS 2024

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper challenges the common view that neighborhood aggregation of GNNs must be learned, and introduces Fixed Aggregation Features (FAFs). FAFs are essentially non-trained feature aggregators (e.g. sum, mean) applied at fixed hops and transformed into tables.A Kolmogorov–Arnold analysis justifies this design, which appears to work empirically in many node classification datasets.

### Strengths
- challenges the established view of needing learned aggregations in GNNs with convincing arguments; in this respect, it can be considered innovative.
- the tabular representation is interpretable with standard tools (e.g. SHAP), which clearly adds value.
- the experiments are adequate in both in width (14 datasets) and depth (large set of hyperparameters), the design is fair and reproducible
- some insights (e.g. GNNs may overfit later aggregations) are definitely thought-provoking and might help designing better GNNs (or improve their benchmarks).
- the paper is clearly written and easy to follow.

### Weaknesses
- The way it is presented, the main finding of the paper appears to be confined to (transductive) node classification. A more broader characterization (e.g. an extension to graph classification or to inductive node-classification) would increase the significance of this work.

- The work (and especially its recommendation on using FAF baselines and reassess benchmarks) is connected with previous work on properly benchmarking GNNs for graph classification. In particular, [1] proposes simple baselines and similarly argues that current graph datasets are often inadequate. These connections should be acknowledged in the Related Works section. 

[1] Errica et al. A Fair Comparison of Graph Neural Networks for Graph Classification. ICLR 2020

### Questions
No questions, besides the relatively minor weaknesses detailed above. I believe this paper is a very solid contribution to the field already as-is.

### Soundness
3

### Presentation
4

### Contribution
3
