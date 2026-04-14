# On the Benefits of Attribute-Driven Graph Domain Adaptation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Graph Domain Adaptation (GDA) addresses a pressing challenge in cross-network learning, particularly pertinent due to the absence of labeled data in real-world graph datasets. Recent studies attempted to learn domain invariant representations by eliminating structural shifts between graphs. In this work, we show that existing methodologies have overlooked the significance of the graph node attribute, a pivotal factor for graph domain alignment. 
Specifically, we first reveal the impact of node attributes for GDA by theoretically proving that in addition to the graph structural divergence between the domains, the node attribute discrepancy also plays a critical role in GDA. Moreover, we also empirically show that the attribute shift is more substantial than the topology shift, which further underscore the importance of node attribute alignment in GDA. Inspired by this finding, a novel cross-channel module is developed to fuse and align both views between the source and target graphs for GDA. Experimental results on a variety of benchmark verify the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, we focus on the graph domain adaptation (GDA) problem and propose a novel cross-channel graph attribute-driven Alignment (GAA) algorithm, which aims to address the distribution differences between the source and target graphs, especially the differences in node attributes and topology.

### Strengths
1. Through theoretical and experimental analysis, the difference of eigenvalues distribution between graph topology and node attributes is analyzed, which reveals that the difference of node attributes is more significant than the difference of topology structure, and further emphasizes the importance of node attribute alignment.
2. The Cross-Channel Module is proposed to align the graph topology and node attributes simultaneously. This module effectively reduces the domain differences by fusing and aligning the two views of the source and target images.

### Weaknesses
1. The whole setting of this paper is to perform domain adaptation (GDA) under supervised conditions, that is, the annotation of the target domain is involved in the training process. However, for most practical problems, the target domain is extremely lack of data annotation or even no annotation, which lacks certain considerations for practical applications. If the target domain is annotated why not design the model for the target domain instead of introducing the source domain.
2. In the experiments section, we see that the latest method you currently cite is the result of 2023, and the comparison test of the relevant latest method in 2024 is missing. And on the types of downstream tasks, there is a lack of evaluation of other tasks, such as graph classification tasks.
3. This paper lacks the time complexity analysis of the model, and does not analyze the overall operating efficiency of the model. In particular, GAA algorithm introduces additional calculation steps (such as cross-view similarity matrix), which may lead to increased computational complexity. On large-scale graph data, the efficiency and scalability of the algorithm may become an issue.
4. In the model graph part of this paper, in the target domain part, the legend labels of the attribute graph and the topology graph are wrong. The text annotation at the bottom of Figure 1 is ambiguous from the data in the bar graph (the difference in the distribution of eigenvalues is greater for topological graphs than for attributed graphs?).

### Questions
1. This paper does not analyze the time complexity of the model, whether it is universal for larger scale node classification problems, and whether it can ensure the operation efficiency of the model on large-scale problems. For example, in the construction of attributed graph using KNN, is it too time-consuming to successively calculate the similarity score between nodes to connect edges?
2. The article is generally on the supervised domain adaptation problem. Personally, I think this application scenario is relatively limited in the general scenario.
3. The article does not give detailed hyperparameter Settings, for example, in the weight design of the loss function, although the experiment carries out the analysis of different weights, the final setting value is not given. In addition, for the weight setting of the loss function, will the weight of a loss function be too low to cause it to not work in the model training process?
4. Can it be compared with the latest GDA method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of Graph Domain Adaptation and shows the significance of the graph node attribute for graph domain alignment. Then, they propose a novel cross-channel module to fuse and align both views between the source and target graphs for GDA. Experimental results on a variety of benchmarks verify the effectiveness of the method.

### Strengths
- The studied problem is interesting and important.
- The paper is well-organized and clearly written.
- The experiments are extensive and helpful to validate the effectiveness of the model.

### Weaknesses
- Some citation styles are not correct, e.g., Line 37.
- The paper lacks some of the latest baselines such as "Information filtering and interpolating for semi-supervised graph domain adaptation (2024)" 
- Masking node attributes could have the risk of losing crucial information. How about alignment in the embedding space?
- I suggest that the authors should use some tSNE to show the effectiveness of the method.
- The parameter sensitivity needs more analysis. How to apply your method to a new dataset.

### Questions
See above.

### Soundness
3

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
3

### Summary
This paper aims to address the Graph Domain Adaptation problem by incorporating both structural information and node attribute values, the latter of which is often neglected by previous approaches. The paper first presents the significance of considering node attribute shifts in GDA. It then derives a PAC-Bayesian analysis, which provides an upper bound on the overall discrepancy in expected loss. This bound relies solely on the divergence between the topologies and node attributes of the source and target graphs. The proposed method combines the topological and node feature representation through an attention module and performs domain adaptive learning between the source and target representations. Experiments showed good performance of the proposed algorithm, compared with multiple baselines.

### Strengths
1. This paper makes a good observation that node attributes are important for domain adaptation tasks and then follows up with a theoretical analysis.
2. The authors conducted comprehensive experiments to validate the algorithm's performance with a sufficient number of baseline methods compared.
3. The ablation study shows the effectiveness of the proposed method and its key components.

### Weaknesses
1. **Writing**: The writing of the paper can be improved. In particular, citation errors appear on the very first page of the paper. The author should proofread the paper more carefully before submission. Some sentences contain grammatical errors.
2. **Theoretical novelty**: The theoretical analysis of the paper is a direct application of the PAC-Bayesian results derived from [1], including the problem formulation and Theorem 1. The discrepancy between subgroups in [1] can be naturally applied to graphs in the source/target domain. 
3. **Tightness of bounds**: The upper bound derived in Proposition 1 is novel, yet I am curious if the upper bound is tight enough to be useful. It would be great if the author could provide some numerical evidence showing the tightness of the bound or experiments that show how the improvement of performance relates to the divergence of both topological structure and data features over training. 
4. **Scalability**: The scalability of the proposed algorithm is a concern, especially when applied to large graphs. The GAA algorithm requires constructing and processing both topology and attribute views, with a cross-view similarity matrix that grows with the size of the graph. This can lead to high computational costs, particularly when both views are combined through an attention module. Running time analysis and comparison with existing baselines are needed.

**References**
[1] Ma, J et al. Subgroup generalization and fairness of graph neural networks.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the Graph Domain Adaptation (GDA) problem by investigating the critical role of node attributes in domain shift. This study reveals that node attributes contribute significantly to domain discrepancy. Through theoretical analysis using the PAC-Bayes framework, the authors prove that the domain shift is bounded by both topological structure differences and node attribute differences. Based on this insight, they propose Graph Attribute-driven Alignment (GAA), a novel method that aligns both structural and attribute information between domains. Experiments are conducted on two datasets: the Airpot network, Citation, Blog, and Social network.

### Strengths
- Unlike previous methods that primarily focused on graph structural information, this approach provides a novel perspective by emphasizing the importance of node attributes in GDA.
- A solid theoretical analysis proved that domain discrepancy comes from both graph structure and node attributes.
- Strong experimental results.

### Weaknesses
- The motivation for using an attention-like strategy is unclear.
- L_A contains both $||att^S - att^T||_2^2$ and $||att_f^S - att_f^T||_2^2$. There is no clear demonstration of which component is more important for performance. To highlight the effectiveness of aligning attributes, we need to check the effectiveness of $||att_f^S - att_f^T||_2^2$ (with/without Coss-view Matrix Refinement).
- The effect of $L_D$ is unclear. It is necessary to verify the contribution of $L_D$ compared to $L_A$, as $L_D$ is also a domain alignment strategy.

### Questions
- Please provide more information about the motivation behind the attention-like strategy (Eq. 8).
- How does the contribution of $||att_f^S - att_f^T||_2^2$ compare to $||att^S - att^T||_2^2$?
- Which is more important for the performance? $L_D$ or $L_A$?

### Soundness
3

### Presentation
4

### Contribution
3
