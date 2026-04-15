# A shot of Cognac to forget bad memories: Corrective Unlearning in GNNs

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Graph Neural Networks (GNNs) are increasingly being used for a variety of ML applications on graph data.  As graph data does not follow the independently and identically distributed (i.i.d) assumption, adversarial manipulations or incorrect data can propagate to other datapoints through message passing, deteriorating the model's performance. To allow model developers to remove the adverse effects of manipulated entities from a trained GNN, we study the recently formulated problem of _Corrective Unlearning_. We find that current graph unlearning methods fail to unlearn the effect of manipulations even when the whole manipulated set is known. We introduce a new graph unlearning method, ***Cognac***, which can unlearn the effect of the manipulation set even when only $5$\% of it is identified. It recovers most of the performance of a strong oracle with fully corrected training data, even beating retraining from scratch without the deletion set while being 8x more efficient. We hope our work guides GNN developers in fixing harmful effects due to issues in real-world data post-training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose **Cognac** model to tackle corrective unlearning in Graph Neural Networks, efficiently removing the impact of adversarial manipulations, even when only 5% of the manipulation set is identified. It recovers most of the performance of an oracle trained with fully corrected data and is 8x more efficient than retraining from scratch. This makes **Cognac** effective in real-world scenarios with limited knowledge of corrupted data, enhancing GNN robustness post-training.

### Strengths
(1) The proposed graph contrastive unlearning is a novel and interesting concept.

(2) The method presented in this paper is simple and effective, without introducing additional discriminative components.

### Weaknesses
(1) The experiments in the paper are seriously insufficient. In the graph domain, there are numerous heterophilic and homophilic graphs, such as Wisconsin, Cornell, and Citeseer. The authors only demonstrate results on three graph datasets, which is clearly inadequate.

(2) There are some conceptually similar works [1], and the authors may need to provide a clearer comparison with these related works.

[1] The Heterophilic Snowflake Hypothesis: Training and Empowering GNNs for Heterophilic Graphs

### Questions
please refer weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces *Cognac*, a corrective unlearning framework designed for Graph Neural Networks (GNNs) to mitigate the adverse effects of manipulated data, even when only a small subset of compromised nodes or edges is identified. Cognac combines two key components: (1) *Contrastive Unlearning on Graph Neighborhoods* (CoGN), which minimizes the influence of manipulated nodes on neighboring nodes via contrastive learning, and (2) *Ascent-Descent Decoupled* (AC DC) gradient optimization, which unlearns harmful labels by alternating between ascent and descent steps. This method effectively reduces adversarial impact while maintaining efficiency and scalability.

### Strengths
1. Cognac’s integration of CoGN and AC DC components offers an innovative approach to corrective unlearning in GNNs, utilizing contrastive learning and decoupled optimization to handle the unique challenges posed by graph-structured data.
2. The framework demonstrates strong performance, achieving effective unlearning even when only 5% of manipulated data is identified, thus substantially lowering the typical data requirement for effective unlearning.
3. The experiments are comprehensive, covering various manipulation types, model types, and datasets. The paper’s ablation studies and visualizations provide clear insights into Cognac’s efficacy and the mechanisms behind it.

### Weaknesses
1. The evaluation metrics (ACC_aff and ACC_rem) mentioned in Lines 175–184, Page 4 differ from traditional accuracy metrics, which directly assess model performance on the original data. Does this mean that the model’s performance on the original data was not evaluated? Or does it imply that naive unlearning performance metrics were excluded? For clarity and comprehensive understanding, I suggest including results from standard unlearning metrics for comparison, even if they may not be favorable for the proposed method. Additionally, if ACC_aff and ACC_rem are novel metrics introduced by the paper, please clarify their originality, or otherwise include citations clearly defined in Lines 175–184.
2. The paper could benefit from a more detailed discussion on why malicious node additions degrade GNN performance. While related studies are referenced, incorporating recent studies would provide a more robust context [1, 2, 3, 4].
3. Correct the typo “th” to “the” in Line 432, Page 9.

**References for Additional Context**:
- [1] *Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning*, NeurIPS 2021, Datasets and Benchmarks Track.
- [2] *Mitigating Emergent Robustness Degradation on Graphs while Scaling Up*, ICLR 2024.
- [3] *Chasing All-Round Graph Representation Robustness: Model, Training, and Optimization*, ICLR 2023.
- [4] *Characterizing the Influence of Graph Elements*, ICLR 2023.

### Questions
See Weaknesses Section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work study the corrective unlearning problem for GNNs, where model developers try to remove the adverse effects of manipulated training data from a trained GNN, with realistically only a fraction of it identified for deletion. The authors introduce a new graph unlearning method, Cognac, which can unlearn the effect of the manipulation set even when only 5% of it is identified.

### Strengths
1. The authors provide a coherent story.
2. Empirical studies on the benchmark dataset are comprehensive.

### Weaknesses
1. When identifying the affected nodes, the inverse features need to be computed. What is the inverse features? The authors then choose the top K% nodes with the largest logit change. How will the chosen nodes influence the results? I suggest the authors provide empirical analysis of the K%. 
2. When calculating the logits change, should the GNN be trained well? I suggest the authors provide more analysis of how the GNN's performance influence on the logits change.
3. The algorithm has two optimization steps. How about the convergence of the algorithm? How to set the parameters of the optimizer?
4. The proposed method lacks justification. I suggest the authors theoretically analyze why selecting the top K% nodes with largest logit change and using the contrastive unlearning objective work.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper tackles Corrective Unlearning in Graph Neural Networks (GNNs) to counter the spread of manipulated data effects. Existing methods fall short, even with complete knowledge of manipulated data. The proposed method, Cognac, effectively reduces manipulation impacts with only partial information (5% of the manipulated set), restoring most of the ideal model’s performance and surpassing retraining efficiency by 8 times.

### Strengths
1. The paper is well-written and easy to follow.
2. This paper studies an interesting problem in the graph domain.
3. The motivation to identify most affected nodes make sense.

### Weaknesses
1. The experiments use GCN or GAT as backbones. Suppose we have a two-layer GNN; in that case, a node will only impact its 2-hop neighbors through message passing. This makes it confusing why, in Sec 4.1, the authors propose using output logits change as an indicator to find the most affected node.
2. I am concerned about updating the model with Eq. (2). If we have a large number of nodes to unlearn, I believe gradient ascent on $S_f$ would significantly reduce model utility. Could the authors add an experiment to verify this?
3. The experiments are not comprehensive, as they only focus on small graph datasets.

### Questions
please refer to weakness

### Soundness
3

### Presentation
3

### Contribution
2
