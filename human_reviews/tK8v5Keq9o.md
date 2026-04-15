# Edge-free but Structure-aware: Prototype-Guided Knowledge Distillation from GNNs to MLPs

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
Distilling high-accuracy Graph Neural Networks (GNNs) to low-latency multilayer perceptrons (MLPs) on graph tasks has become a hot research topic. However, conventional MLPs almost exclusively rely on the graph nodes and fail to effectively capture the graph structural information. Previous methods address this issue by processing graph edges into extra inputs for MLPs, but such graph structures may be unavailable for various scenarios. To this end, we propose Prototype-Guided Knowledge Distillation (PGKD), which does not require graph edges (edge-free setting) yet learns structure-aware MLPs. Our insight is to distill graph structural information from GNNs. Specifically, we first analyze the impact of graph structures on GNN teachers, and then design two losses based on prototypes to distill such information from GNNs to MLPs. Experimental results on popular graph benchmarks demonstrate the effectiveness and robustness of the proposed PGKD.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Prototype-Guided Knowledge Distillation (PGKD), a technique for distilling Graph Neural Networks (GNNs) into low-latency, structure-aware Multilayer Perceptrons (MLPs) without requiring graph edge information. Utilizing class prototypes, the authors develop both intra-class and inter-class losses to facilitate this distillation process.

### Strengths
1. The manuscript is articulately composed and presents its content in a clear and coherent manner, making it accessible and straightforward for readers to follow. The overall quality of presentation within the paper is commendable.
2. The authors have dedicated a substantial portion of the paper (Section 6) to empirical analysis and detailed discussions of the method, providing valuable insights and a deeper understanding of the approach and its implications.

### Weaknesses
1. concerns about the effectiveness of PGKD:
    1) A crucial piece of information that is missing from the paper is the comparative performance of GLNN and a standard MLP. This comparison is vital as it establishes a baseline to understand the necessity of using PGKD. For instance, Lim et al. have demonstrated that a vanilla MLP can achieve a performance of 60.92 on the twitch-gamers dataset [1], which is slightly higher than what PGKD manages to achieve. 

    2) The enhancement in performance that PGKD provides over GLNN is minimal, with the majority of improvements being less than 1%. This marginal gain puts into question the practical significance and impact of adopting the PGKD method.

2. Missing important related works:
    1) The authors claim that they are the first to study the impact of intra-class edges and inter class edges. However, there are a moderate number of existing studies in the literature, e.g,in [2].

    2) Missing a few related works for prototypes and graph neural networks, e.g, [3][4]

    3) The intra-class loss serves to attract a selected node towards its associated prototype, and while it is not identical, its objective aligns with that of the compatibility loss proposed in [3]. 


3. The authors emphasize latency reduction during inference as a key motivation for their work. Nevertheless, the paper lacks a crucial comparison of runtime between PKGD, GLNN, MLP, and GNN. Including such a comparison is vital to substantiate the authors' claims and showcase the practical benefits of PKGD in real-world applications.


References:

[1] Lim et al. "Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods." Advances in Neural Information Processing Systems 34 (2021): 20887-20902.

[2] Chen et al. "Measuring and relieving the over-smoothing problem for graph neural networks from the topological view." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.

[3] Dong et al. "ProtoGNN: Prototype-Assisted Message Passing Framework for Non-Homophilous Graphs." (2022).

[4] Zhang et al. "Protgnn: Towards self-explaining graph neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

### Questions
1. What is the runtime comparison between PKGD, GLNN, MLP, and GNN?
2. What is the performance comparision between PKGD and MLP?
3. Regarding the motivation for edge-free setting (section A.2), could the authors provide more concrete examples or real-world scenarios where the edge information is strictly confidential or proprietary, yet the GNN outputs and node embeddings can be freely shared?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of knowledge distillation from GNNs to MLPs, with the absence of edges.
The authors try to tackle this problem by introducing prototypes, which is a common strategy in KD and GNNs.
The authors conduct some experiments on some small datasets and compare with a few baselines, which can somehow prove the effectiveness of the proposed method.

### Strengths
* The problem of distillation from GNNs to MLP is interesting and useful.
* The paper is overall easy to follow.

### Weaknesses
* The motivation of introducing the prototype is not clear, it does not seems to tackle the problem of "structure-aware distillation". Although it can somehow satisfy the edge-free requirement, the advantage in distilling the structure information is not clear, which is more critical in GNN distillation.
* The so-called "the impact of graph structures (i.e. graph edges) on GNNs has been studied" is over-claimed. First, the conclusion based on the inter and intra edges is trival. Second, the analysis and conclusion are based on some small datasets, which is not sufficient.
* The overall model only introduce the prototypes for more information distillation, rather than overcoming the critical "structure-aware distillation" problem
* The evaluated dataset is too small, where the distillation is not necessary. 
* The baselines are weak and many important baselines are missed.

### Questions
* How can the model distill the structure information?
* What's the advantage of introducing prototypes?
* Please provide more results on large scale datasets and more baselines.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces two regularization terms that help to control the process of knowledge distillation from a GNN teacher to MLP. The authors separate the impact of topology structures into two types, inter-class effects and intra-class effects, and encourage MLP to imitate the class embedding correlations. The experimental results demonstrate the effectiveness of this approach.

### Strengths
S1: The paper provides a novel aspect of the analysis of the effect of edges from inter-class effect and intra-class effect.

S2: The paper is well-motivated and clear.

S3: The method is simple but effective.

### Weaknesses
W1: A more comprehensive analysis is needed to explore the inter-class effects. While inter-class edges may impact the inter-class distribution, it's unclear whether mimicking GNN embeddings consistently yields the best classification results. An alternative approach, the supervised InfoNCE loss, compresses intra-class distribution and promotes uniform distribution of class prototypes in the embedding space for better class separability. To better understand the merits of PGKD, a comparison and analysis with supervised InfoNCE loss is warranted.

W2: When conducting experiments related to #H, it would be beneficial to include larger datasets. As indicated by GLNN, #H notably influences inductive results in larger datasets. It is advisable to evaluate PGKD in this context and provide a comparison to demonstrate its optimal performance, rather than solely assessing its performance on small datasets.

W3: Could you clarify the rationale behind selecting SAGE and GCN for small and large datasets? Are there specific reasons for this choice?

W4: While the hyperparameters \lambda 1 and \lambda 2 play a pivotal role in obtaining optimal results, the current search space for these parameters is rather constrained. What might be the impact of expanding this search space to encompass a broader range of values?

W5: Why does the inductive test of GNN not need the graph structure according to Table 2?

### Questions
Refer to weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a method called Prototype-Guided Knowledge Distillation (PGKD) to distill high-accuracy Graph Neural Networks (GNNs) into low-latency Multilayer Perceptrons (MLPs) for graph tasks. Unlike conventional approaches, PGKD does not require graph edges but effectively captures graph structural information by distilling it from GNNs. The authors propose two prototype-based losses to facilitate this knowledge transfer. Experimental results on popular graph benchmarks demonstrate the effectiveness and robustness of PGKD.

### Strengths
The proposed Prototype-Guided Knowledge Distillation (PGKD) presents a novel approach to address the limitation of conventional MLPs in capturing graph structural information, especially in an edge-free setting. This problem formulation is interesting and brings forth a creative combination of knowledge distillation and prototype learning.

The paper has conducted experiments on popular graph benchmarks, which adds some credibility to the empirical evaluation of the proposed method. The integration of prototype learning with knowledge distillation to distill graph structural information from GNNs to MLPs shows a thoughtful attempt to improve model interpretability and performance.

### Weaknesses
1. I don't fully understand why the distilled model with prototypical label can be called "structure-aware". It looks like another supervision we got from GNN over original graph. Given this definition, the original GLNN can also be regarded structure-aware. Can the authors explain this part more clearly, besides the prototypical label, what's additional difference?

2. Seems the only compared baseline is GLNN. Can the authors compare with some other MLP baselines (e.g., Graph Random Network)

This paper discusses a few such baselines: https://openreview.net/forum?id=tiqI7w64JG2

### Questions
I'm still not fully convinced why a prototypical cluster can help distillation, is it only work for graph or can be generalized to other datasets? Can the authors provide some explanation?

How is this work differ from some prototypical contrastive learning: https://openreview.net/pdf?id=KmykpuSrjcq and https://arxiv.org/abs/2012.12533

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
