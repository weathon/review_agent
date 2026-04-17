# Breaking Rank Bottlenecks in Knowledge Graph Embeddings

- Decision: Reject
- Scores: 6, 0, 2, 6

## Abstract
Many knowledge graph embedding (KGE) models for link prediction use powerful encoders. However, they often rely on a simple hidden vector-matrix multiplication to score subject-relation queries against candidate object entities. When the number of entities is larger than the model's embedding dimension, which is often the case in practice by several orders of magnitude, we have a linear output layer with a rank bottleneck. Such bottlenecked layers limit model expressivity.
We investigate both theoretically and empirically how rank bottlenecks affect KGEs. We find that, by limiting the set of feasible predictions, rank bottlenecks hurt the ranking accuracy and distribution fidelity of scores. Inspired by the language modelling literature, we propose KGE-MoS, a mixture-based output layer to break rank bottlenecks in many KGEs. Our experiments show that KGE-MoS improves ranking performance of KGE models on large and dense datasets at a low parameter cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates rank bottlenecks in knowledge graph embedding (KGE) models caused by the common low‑rank linear output layer. The authors theoretically show that such bottlenecks limit expressivity for ranking, sign, and distributional reconstruction tasks. They introduce KGE‑MoS, a mixture‑of‑softmaxes output layer that nonlinearly combines multiple softmax components to break low‑rank constraints at low parameter cost. Experiments on multiple large‑scale datasets demonstrate that KGE‑MoS consistently improves ranking accuracy and probabilistic fit compared with standard KGEs.

### Strengths
- The authors analyze the rank bottleneck problem in knowledge graph embedding (KGE) models and provide comprehensive theoretical constraints and inexpressibility proofs under three common tasks (DR, SR, and RR).
- A lightweight and general method, KGE‑MOS, is proposed by replacing the standard linear output layer with a mixture‑of‑softmaxes.
- The proposed method is evaluated on four datasets (FB15k‑237, Hetionet, ogbl‑biokg, and openbiolink) and achieves performance improvements on large knowledge graphs.

### Weaknesses
- The proposed method may not scale effectively to extremely large graphs (e.g., tens of millions of entities), where increasing embedding dimensions remains a common approach to enhance expressivity.
- The evaluation currently involves a limited set of relation‑rich datasets; it would be valuable to include additional experiments on large open KGs such as Wikidata.
- The performance degradation on smaller datasets indicates potential instability, which requires a clearer explanation.
- The training time increases due to multiple softmax computations.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
Argues that factorization-based KGE models produce a low-rank scoring matrix so that its representational power is limited by a rank bottleneck. Then transforms entity-relation embeddings uses multiple different non-linear transformations and subsequently ensembles; this increases the rank of the scoring matrix at low additional cost. Reports on a small experimental study using multiple KGE models.

I recommend to reject this paper because it is oblivious to key related work (W1), it partially does not study the actually relevant problems (W2), and it's experimental study is not convincing (W3).

### Strengths
Hard to say, as key related work is not represented.

### Weaknesses
W1. Not novel and key related work missing. The rank bottleneck has been studied in more detail, using tighter bounds, and applied to more models in [A]. It also proposes an ensemble approach. This paper is neither cited nor discussed.

W2. Does not use the right problems. The paper asks whether a KGE model can express every ranking. That's not relevant, however, if a KGE model can express every ranking, but only whether it can rank positives higher than negatives: the relative ranking of, say, two positives does not matter.

W3. Experimental results not convincing. Improvements of FB15k-237 are tiny and results (in terms of MRR) fall behind what has been reported for the baseline models in other papers. This reduces trust in the other datasets, which haven't been used that much in related work as far as I know. Datasets such as Wikidata5M, which have been used, are missing.

Wang et al., On Multi-Relational Link Prediction with Bilinear Models, AAAI, 2018

### Questions
-

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
A drop-in output layer for select Knowledge Graph Embedding models to increase model expressivity and thus predictive power by addressing "rank bottlenecks". The output layer consists in a mixture of softmaxes.

### Strengths
- Problem novelty: rank bottleneck problem not studied yet in KGE literature, to the best of my knowledge.
- Paper is well written, and it includes comprehensive material.
- Contribution is an original adoption of methods from language modelling literature.
- Evaluation: good mixture of benchmark datasets.

### Weaknesses
- The rank bottleneck problem could use a more in-depth introduction, to broaden up the audience.
- Contribution limited to adopting a MoS layer to existing KGE architectures.
- KGE-MOS does not support translation-based KGE methods (e.g. RotatE).
- Evaluation: limited impact of \*-MoS on predictive power. results at par with baselines.
- Experimental results presented in the paper does not justify the adoption of KGE-MOS in practice due to computational overhead (e.g. 2.75 slower to train)

### Questions
- Figure 1: Why adding a relation type as target object? Could you please elaborate?

### Soundness
3

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
The paper looks into a limitation in tensor-multiplicative scoring functions used in KGE models, which the authors describe as a kind of "rank bottleneck" that prevents models from properly capturing complex relationships, especially in large graphs with high connectivity. To address this, they introduce KGE-MOS, which modifies the output layer by combining multiple softmax outputs. The idea is that this mixture can produce a more expressive scoring function and help models rank entities more accurately, particularly on large-scale knowledge graphs.

### Strengths
I think the paper’s main strength is its theoretical analysis of the bottleneck problem. The authors do a solid job of identifying and characterizing this limitation and even relate it to graph connectivity, which is interesting. The experiments are also convincing overall that KGE-MOS seems to improve results in the right settings (i.e., large graphs) without blowing up the parameter count. It’s a clear and well-motivated piece of work.

### Weaknesses
I do have a few concerns about the experimental part.

W1. The baselines are reasonable (DISTMULT, ConvE, etc.), but they’re a bit outdated. There are more recent KGE architectures — some Transformer- or GNN-based — that also use similar scoring layers. It would strengthen the argument a lot if the authors could show that their method helps even those stronger baselines, not just the classic ones.

W2. The ablation on the number of mixtures, $K$, is only done on DISTMULT and on a single dataset (ogbl-biokg). That’s informative, but it’s hard to know how general the trend is. Running at least one more model or dataset would make the case much stronger.

### Questions
Q1. The approach reminds me a bit of ensemble-based KGE methods (e.g., [1]). It might be worth clarifying how KGE-MOS is different from those, since both seem to combine multiple outputs to improve expressivity.

Q2. While the paper has a nice theoretical discussion of the rank bottleneck itself, the expressivity of the proposed fix is mostly justified through intuition and experiments. Have the authors considered analyzing the expressivity of KGE-MOS more formally? It would help complete the theoretical story.

### Soundness
3

### Presentation
4

### Contribution
3
