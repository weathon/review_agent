# MuseGNN: Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 6, 5, 6, 8

## Abstract
Among the many variants of graph neural network (GNN) architectures capable of modeling data with cross-instance relations, an important subclass involves layers designed such that the forward pass iteratively reduces a graph-regularized energy function of interest. In this way, node embeddings produced at the output layer dually serve as both predictive features for solving downstream tasks (e.g., node classification) and energy function minimizers that inherit transparent, exploitable inductive biases and interpretability. However, scaling GNN architectures constructed in this way remains challenging, in part because the convergence of the forward pass may involve models with considerable depth. To tackle this limitation, we propose a sampling-based energy function and scalable GNN layers that iteratively reduce it, guided by convergence guarantees in certain settings. We also instantiate a full GNN architecture based on these designs, and the model achieves competitive accuracy and scalability when applied to the largest publicly-available node classification benchmark exceeding 1TB in size. Our source code is available at https://github.com/haitian-jiang/MuseGNN.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a scalable graph neural network (GNN) architecture that minimizes an energy function through a sampling-based approach, allowing for interpretable node embeddings optimized for tasks like node classification.

### Strengths
1. The paper discusses a critical issue of scalability when adding energy regularization to GCN, where the cost of energy calculation becomes enormous for large-scale graphs and previous methods fail to work on large graphs.

2. The paper proposes a new method, MuseGNN, to enable energy-regularized GNN to efficiently handle large-scale graphs. The paper also comprehensively discusses the optimization procedure and convergence analysis of their method.


3. The paper validates the performance and efficiency of MuseGNN by experiments on multiple-datasets, including large graph datasets where previous methods need huge computational cost.

### Weaknesses
1. The annotations and formulation in 2.1 may be confusing for new readers. The authors could adopt similar annotations in Descent Steps of a Relation-Aware Energy Produce Heterogeneous Graph Neural Networks by Ahn et al, which distinguishes the node embedding by basic model and the embedding by energy optimization by different annotations y and y*, and specify the procedure of UGCN to make the problem formulation clearer to a broader audience.
2. The authors could compare the time and memory cost of MuseGNN with more baselines that also uses energy regularization without sampling to strengthen the superiority of MuseGNN.
3. In the introduction, the authors could provide more specified discussions about the advantages of applying energy regularization to GNN to show the importance of this work.

### Questions
1. The ablation study of hyperparameter gamma shows that MuseGNN preserves high accuracy when gamma=0. In this case, the embedding of the same node in different subgraphs are not aligned. Can the node embeddings in subgraphs capture the global information of the original graph? Are there any theoretical insights about why MuseGNN still shows comparative performance even when gamma = 0?


2. What is the sampling strategy when sampling subgraphs from the large original graph?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper developed one subgraph sampling strategy induced from the energy regularization process over the GNN learning process. The author also discussed the convergence and expressive power of their proposed model under some conditions (i.e., $\gamma$). The newly proposed model is advanced with a shared weight matrix over all subgraphs and by the consideration of replicated nodes in multiple subgraphs. Empirical studies show promising results via large-scale graph datasets.

### Strengths
1. The paper resolved the scalability problem on the GNNs induced by their energy optimization form. Empirical studies show the proposed model outperforms many baselines via different graph sizes up to 1TB. 

2. The paper provided theoretical analysis on the convergence of the proposed model under different values of $\gamma$, which controls the degree of dependency between subgraphs.

### Weaknesses
1. Some important analysis and empirical study results are missing and shall be presented via the main text, please see the question part. 

2. The organization of the paper needs to be carefully adjusted.

### Questions
At the current stage, my questions are as follows: 

1. The organization of the paper needs to be improved. For example, figure 1 on page 2 is referred to in both the introduction section and the experiment section (ablation study), which is rarely to be observed in other studies. Also, the ablation result for $\gamma$, located in Table 6, in Appendix B2, can be put into the current main page, which only contains 9.5 pages. 

2. In addition to this, I think the ablation on $\gamma$ is very important as this shows how the degree of dependence between subgraphs affects the final training. Therefore, it is recommended that a more detailed analysis of this aspect be input. 

3. Another important empirical verification is the performance of the model via different sampling strategies since this directly affects the power of the term $\|Y_s - \mu_s\|$. I found the related content in the Appendix, but it is recommended that the author put some convincing evidence to the main content.

4. Furthermore, the energy regularizer $\lambda \mathrm{Tr}(\mathbf Y_s \mathbf L_s \mathbf Y_x )$ controls the degree of smoothing via each subgraph, and the quantity of $\lambda$ is shared over all the subgraph. However, in real practice, this may not be ideal as subgraphs shall have different degrees of smoothing.

5. The expressive power of the GNN is usually discussed under the graph pooling tasks (e.g., graph level classification), it would be better and interesting to show some pooling results from the proposed model.

Minor modification. 

1. In row 183, table 4 is missing, and it is allocated in the appendix.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MuseGNN, a novel Graph Neural Network (GNN) architecture designed to address the challenges of scaling and convergence in GNNs. The core idea behind MuseGNN is to iteratively minimize a sampling-based energy function during the forward pass, which allows the node embeddings to serve dual purposes: as predictive features for downstream tasks and as minimizers of the energy function. The authors present a scalable GNN that is able to deal with large-scale node classification benchmarks, including datasets exceeding 1TB in size.

### Strengths
- The paper provides an analysis of the convergence properties of the proposed energy function and the iterative reduction process. 
- MuseGNN is designed to handle very large graphs, as evidenced by its performance on the node classification benchmark exceeding 1TB in size. 
- The experimental results show that MuseGNN achieves competitive accuracy compared to state-of-the-art GNN.

### Weaknesses
- The paper focuses on comparing MuseGNN with a few specific GNN architectures and frameworks. A more comprehensive comparison with a wider range of state-of-the-art GNNs. In particular, I want to see the comparison to SGC [1] and its variants in terms of efficiency and accuracy.
- The training speed of the proposed method seems to be slower than GAT as shown in Table 1, while GAT is a well-known slow GNN.

[1] Simplifying Graph Convolutional Networks

### Questions
- How is the method compared to SGC and its variants in terms of efficiency and accuracy?
- How does the method perform on heterophilic graphs? The energy-related loss seems to highly rely on the homophilic assumption.

### Soundness
3

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
The paper proposes a GNN model  that scales effectively to large datasets by incorporating sampled subgraphs into its energy function design. This approach allows the model to handle graphs with around 100 million nodes and high-dimensional features, achieving competitive accuracy on benchmarks exceeding 1TB in size. Additionally, it maintains desirable inductive biases and convergence guarantees.

### Strengths
1). The organization and writing of this paper are excellent.

2). The model enhances UGNN framework by incorporating sampling into its energy function design and demonstrates solid convergence properties.

3).  Empirical results suggest that MuseGNN maintains competitive accuracy and scalability across various task sizes, performing well on large node classification datasets exceeding 1TB.

### Weaknesses
1). How to determine \alpha and \lambda for different datasets?

2). Does the model's improvement differ for homogeneous and heterogeneous graphs? Can you provide some additional explanation?

### Questions
Refer to the content in the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
