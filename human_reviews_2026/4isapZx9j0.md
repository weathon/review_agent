# Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Graph Neural Networks (GNNs) have shown remarkable capabilities in learning from graph-structured data with various applications such as social analysis and bioinformatics. However, the presence of label noise in real scenarios poses a significant challenge in learning robust GNNs, and their effectiveness can be severely impacted when dealing with noisy labels on graphs, often stemming from annotation errors or inconsistencies. To address this, in this paper we propose a novel approach called ICGNN that harnesses the structure information of the graph to effectively alleviate the challenges posed by noisy labels. Specifically, we first design a novel noise indicator that measures the influence contradiction score (ICS) based on the graph diffusion matrix to quantify the credibility of nodes with clean labels, such that nodes with higher ICS values are more likely to be detected as having noisy labels. Then we leverage the Gaussian mixture model to precisely detect whether the label of a node is noisy or not. Additionally, we develop a soft strategy to combine the predictions from neighboring nodes on the graph to correct the detected noisy labels. At last, pseudo-labeling for abundant unlabeled nodes is incorporated to provide auxiliary supervision signals and guide the model optimization. Experiments on benchmark datasets show the superiority of our approach over competitive baselines in noisy label scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of alleviating noisy labels in semi-supervised node classification. It proposes a two-stage framework including noise detection and noise alleviation. In the first stage, two neighborhood structures are defined: one based on graph structure via PPR scores, and the other based on node embeddings using KNN. A Gaussian Mixture Model (GMM) is then used to assess whether a node’s prediction differs from its neighbors. In the second stage, the node’s neighborhood and its own label are combined (weighted by a trainable confidence score) to produce a refined prediction. Empirical studies show that the proposed method outperforms all baselines and remains stable across varying noise levels and label ratios.

### Strengths
1. The presentation of this paper is clear and easy to follow. Figure 1 and most equations are mathematically correct, which enhances clarity.  

2. The number of baselines included in the comparison experiments is comprehensive, and the main empirical results are supported by statistical significance tests, which strengthens practical soundness.  

3. The proposed method demonstrates robustness to varying label rates and noise levels, as evidenced by sensitivity studies.  

4. The description of hyperparameters and implementation details is sufficient, supporting the reproducibility of this work.

### Weaknesses
**Major issues**:  

1. The novelty of this paper is rather limited, as it combines several standard techniques that have been proposed over a decade ago (e.g., GMM, Personalized PageRank for influence estimation). The rationale for choosing these technical routes is not clearly explained, as there are no preliminary empirical studies or theoretical analyses.  

2. The time complexity described in lines 243-247 is inaccurate. Eq. (2) requires the computation of a matrix inverse, which is costly at $O(n^3)$. When approximating the exact $T$ with an approximate PPR, the performance of the proposed framework may vary, which warrants further sensitivity analysis.  

3. The empirical improvements on most datasets, as shown in Table 1, are incremental. While this alone does not warrant rejection, the paper places strong emphasis on empirical performance yet does not offer trustworthy insights into why prior methods fall short or what motivates the proposed approach. In this context, the marginal gains become more concerning.

4. A major concern is the paper’s heavy reliance on the homophily assumption, as stated in lines 137–145. It remains unclear whether the method would still perform well on heterophilous graphs, which are common in real-world scenarios.  

5. The experiments are conducted only on small-scale graphs, with no study of scalability or performance on large-scale datasets. It would be desirable to include at least one or two large graphs, such as ogbn-arxiv or ogbn-products.

**Minor issues and typos**:  

1. Eq. (1) presents the formulation of GCN, not the general message-passing mechanism. A more general formulation would be preferable.  

2. Clarity would be improved by explicitly writing out the mathematical formulation of matrix $R$, rather than describing it only in English (lines 164–173).

### Questions
1. As stated in lines 66–68, previous methods for noise detection are considered undesirable. Is there any direct empirical evidence comparing this paper’s noise detection module with prior approaches? Could you provide insights into why the proposed detection method (e.g., combining both feature and structure information, using GMM) is more effective?

2. This work uses confidence estimation for noise alleviation (e.g., Eq. (6)). Given the existence of many prior works on confidence estimation in GNNs, how could your method be integrated with or compared to these approaches?

3. How did you approximate the exact PPR score $T$ in practice? What is the performance-efficiency trade-off involved? This aspect requires further empirical evidence.

4. How could your method be extended to heterophilous graphs?

5. Could you provide additional results on large-scale graphs, such as ogbn-arxiv or ogbn-products?

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
3

### Summary
The paper proposes a framework called ICGNN for improving the robustness of Graph Neural Networks (GNNs) under label noise. The method introduces an Influence Contradiction Score (ICS) based on the graph diffusion matrix to quantify node reliability, where higher ICS values indicate higher likelihoods of label corruption. A Gaussian Mixture Model (GMM) is then applied to detect noisy nodes, followed by a soft correction strategy that integrates predictions from neighboring nodes to adjust mislabeled samples. Additionally, pseudo-labeling is employed to provide auxiliary supervision for unlabeled nodes. Experimental results on benchmark datasets demonstrate that ICGNN achieves superior robustness and accuracy compared to existing baselines under various label noise settings.

### Strengths
1. Originality:
The paper proposes a framework for improving the robustness of GNNs under label noise. Unlike prior works that mainly rely on confidence estimation or sample reweighting, this paper proposes the Influence Contradiction Score (ICS) provides a principled way to identify mislabeled samples by leveraging the inherent structural dependency in GNNs, rather than relying on purely probabilistic confidence signals.

2. Quality:
The technical pipeline is coherent and logically motivated. The proposed method consists of two main steps. The approach is conceptually clear and follows standard procedures for label noise modeling, including the use of a Gaussian Mixture Model for probabilistic estimation. Experiments are conducted on several benchmark datasets with both symmetric and asymmetric noise settings. Results indicate that the proposed method performs competitively against existing baselines. The analysis of complexity and ablation provides partial support for the method’s effectiveness.

3. Clarity:
The article is generally well-written and most of the content is easy to understand.

4. Significance:
This paper addresses the issue of label noise frequently present in large-scale or automatically collected graph data in graph neural networks. The authors propose using "influence contradictions" as a noise indicator, providing a new direction for influence-based graph model reasoning and contributing to the development of subsequent research. However, the overall method is relatively simple and lacks innovation, potentially limiting its contribution to the theoretical advancement of the graph neural network field.

### Weaknesses
1. The method's correction mechanism heavily relies on neighbor label consistency. In low-homogeneity graphs (such as Chameleon and Squirrel), neighbor labels may have significant semantic differences, potentially leading to incorrect corrections. It is recommended to conduct empirical analysis or ablation experiments on low-homogeneity datasets to verify the method's robustness.

2. Although the paper analyzes the method's computational time complexity, the computation of the influence contradiction score (ICS) may be costly when the number of nodes in the graph is large.

3. The method's motivation is not clearly defined, especially why graph structure information needs to be incorporated into noise detection. Further clarification of the intuitive reasons for the method's design is recommended.

4. The experimental section of the article mainly includes basic comparison and ablation experiments, but lacks in-depth analysis of the impact of the calculation of the Influence Contradiction Score (ICS) and label correction on the final performance improvement. The lack of such experiments makes it difficult to intuitively demonstrate the specific contribution of the proposed method to performance improvement.

5. The baseline comparisons on OGBN-Arxiv and the heterogeneous network dataset Cornell only include four methods (Table 5), while the main table (Table 1) contains nine baselines. This prevents a full demonstration of the method's effectiveness on larger-scale or heterogeneous graphs, and also fails to compare it with recent methods such as CR-GNN and ProCon.

### Questions
Please refer to Weaknesses. In addition, the paper highlights the urgent need for a noise detection method that can fully incorporate graph structure information and combine it with a more effective label correction strategy to enhance the robustness of graph neural networks in noisy label scenarios. Could you further explain the specific advantages of the proposed method compared to existing methods in terms of detection or correction strategies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The study aims to address the significant decline in the robustness of GNNs on semi-supervised node classification tasks caused by label noise and label sparsity in real-world scenarios. The paper proposes an ICS mechanism to identify potential mislabeled samples and incorporates a pseudo-labeling strategy to enhance GNN training. Experimental results demonstrate that ICGNN establishes new state-of-the-art performance in robust semi-supervised node classification across six benchmark datasets.

### Strengths
- The paper is well-structured and clearly written, making it easy to follow.

- The overall methodological design is reasonable: the ICS mechanism effectively quantifies the impact of noisy labels, while the use of soft labels mitigates confirmation bias.

- The experiments take into account large-scale GNN scenarios, which strengthens the practical relevance of the work.

### Weaknesses
When the proportion of labeled nodes is relatively high, the computational cost of calculating ICS becomes significant.

- The ICS mechanism is likely to struggle in distinguishing between hard samples and mislabeled samples—an aspect that is particularly crucial in robust graph learning.

- The proposed approach may be highly sensitive to the number of node classes, yet this factor is neither experimentally examined nor discussed.

- The study does not include experiments on real-world noisy datasets, such as NoisyGL (NoisyGL: A comprehensive benchmark for graph neural networks under label noise).

- The performance of ICGNN appears to be strongly influenced by the graph structure, and the balance coefficient $\alpha$ between ICS-T and ICS-R remains difficult to determine, still requiring manual tuning.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes ICGNN, a method designed to improve the robustness of Graph Neural Networks (GNNs) against label noise by leveraging graph structural information. It introduces an Influence Contradiction Score (ICS) to identify potentially noisy nodes, applies a Gaussian Mixture Model for noise detection, and uses soft correction and pseudo-labeling strategies to refine labels. Experiments on benchmark datasets show that ICGNN outperforms existing baselines under noisy label settings.

### Strengths
1. The structure of the paper is clear.

2. The proposed measurement appears meaningful.

### Weaknesses
1. Clarity of the Proposed Problem
The proposed research problem is not clearly defined. For example, the statement “nodes with higher ICS values are more likely to be detected as having noisy labels” is confusing — are nodes with higher ICS values considered correct or mislabeled? In addition, the phrase “the credibility of nodes with clean labels” is unclear. If a label is already clean, what does “credibility” mean in this context? Overall, the problem formulation and motivation for detecting noisy labels need to be clarified.

2. Lack of Novelty in Motivation
The motivation of the paper is not sufficiently novel. The idea of incorporating graph structural information to improve the robustness of GNNs has already been extensively studied in previous works.

3. Missing Discussion of Related Work
The paper claims to quantify the influence contradiction among diverse node classes based on graph structure. However, similar ideas have been explored in prior works such as TSS[1], which also proposes a related measure based on Personalized PageRank. These relevant studies are not discussed or compared in the paper.

4. Experimental Comparison
The experimental section lacks comparison with state-of-the-art baseline methods. Moreover, only uniform and pair noise settings are considered, which are not sufficient to demonstrate robustness.

5. Misalignment Between Motivation and Experimental Setup
The experimental design does not align with the problem statement. The paper mentions that noise is more likely to occur in nodes with higher ICS values, yet the experiments generate noisy labels using uniform and pair noise, which does not reflect the stated assumption.

[1] Wu, Y., Yao, J., Xia, X., Yu, J., Wang, R., Han, B., & Liu, T. (2024). Mitigating label noise on graph via topological sample selection. arXiv preprint arXiv:2403.01942.

### Questions
see on Weaknesses

### Soundness
1

### Presentation
1

### Contribution
1
