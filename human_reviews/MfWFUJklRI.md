# GSINA: Improving Graph Invariant Learning via Graph Sinkhorn Attention

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3, 5

## Abstract
Graph invariant learning (GIL) has been extensively studied to discover the invariant relationships between graph data and labels for different graph learning tasks under various distribution shifts. 
Many recent endeavors of GIL focus on discovering invariant features to improve the generalization of graph learning. However, such methods often have limitations in obtaining invariant features that are expressive enough in the solution space. 
In this paper, we first discuss the limitations of previous works and summarize there design principles of the invariant feature extractor for GIL: 1) the sparsity, to filter out the variant features, 2) the softness, for a broader solution space, and 3) the differentiability, for a soundly end-to-end optimization. 
By leveraging the Optimal Transport (OT) theory, we propose  Graph Sinkhorn Attention (GSINA) to meet these requirements in one shot. GSINA is a framework for GIL of multiple task levels, which infers differentiable graph invariant features with controllable sparsity and softness. 
Experiments on both synthetic and real-world datasets validate the superiority of our GSINA, which outperforms the state-of-the-art GIL methods (GSAT, CIGA, EERM) by large margins on graph-level tasks and node-level tasks. The PyTorch source code is provided in supplementary materials and will be publicly available on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of graph invariant learning (GIL), aiming to find edges or nodes that are related to label information and invariant to environmental changes. This paper proposes three principles in GIL: Sparsity, Softness, and Differentiability, which cannot be fully covered by previous GIL methods. To address this issue, this paper designs a new regularization, namely Graph Sinkhorn Attention (GSINA), based on the optimal transport theory. GSINA can control the sparsity and softness of edge attention, and therefore improve the performance of GIL. Experiments on both real and synthetic datasets validate the effectiveness of the proposed method.

### Strengths
1. The idea of utilizing the GIL for optimal transport is somewhat interesting. It makes sense to move the coefficients to 1 for invariant edges and to 0 for spurious edges.

2. The proposed method GSINA consistently outperforms other GIL methods.

### Weaknesses
1. In addition to the three principles proposed, I think there is a fourth principle, which is the completeness of the subgraph. In practice, we expect important edges to form a complete subgraph. For example, in molecular property prediction, invariant information should be related to functional groups rather than individual chemical bonds or atoms. This is the advantage of subgraph selection methods over information bottleneck methods. I am concerned about whether enforcing sparsity guarantees this principle.

2. There seems to be no clear reason to replace information bottleneck with optimal transport. We can also apply the Gumbel trick to  Graph Stochastic Attention (GSAT) to ensure its sparsity. Is there any theoretical evidence to prove the effectiveness of optimal transport over information bottleneck? Additionally, we can observe from the ablation study (Table 6) that without the help of the Gumbel trick, GSINA performs similarly or even worse than GSAT. Therefore, I am concerned about the effectiveness of the optimal transport.

3. The experimental results appear to be copied from other papers. But I think it would be better if this paper could provide the results for some important baselines. For example, GSAT in the graph-level OOD tasks.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel graph attention mechanism called Graph Sinkhorn Attention (GSINA) for graph invariant learning (GIL). GSINA extracts sparse, soft, and differentiable invariant subgraphs from input graphs by leveraging the optimal transport theory and the Sinkhorn algorithm. The proposed method acts as a powerful regularization to improve generalization in GIL tasks. The key benefits of GSINA are that it meets the desired principles of sparsity, softness, and differentiability for invariant subgraph extraction. Experiments across synthetic and real-world benchmarks demonstrate that GSINA outperforms prior state-of-the-art GIL methods on both graph-level and node-level tasks.

### Strengths
1. Originality: The paper proposes a new graph attention mechanism using optimal transport to extract invariant subgraphs. This is a novel application of the Sinkhorn algorithm not explored before for invariant learning on graphs.

2. Quality: The theoretical analysis explains the design principles and formulations behind the proposed approach. The experiments compare against multiple baselines over several benchmarks to demonstrate the effectiveness of the method.

3. Clarity: The paper is well organized and clearly explains the background, proposed method, and experimental results. Visualizations provide some intuition about the sparse graph attention.

4. Significance: The work introduces a general framework for graph invariant learning that is applicable to both node and graph tasks. It provides improvements over state-of-the-art techniques, showing promise for this approach to invariant learning on graph data.

### Weaknesses
1. The paper lacks an in-depth theoretical analysis of why the proposed optimal transport approach and Sinkhorn algorithm can effectively extract invariant subgraphs for graph learning tasks. More analysis connecting GSINA to invariance principles or analyzing its inductive biases could strengthen the method.

2. The paper lacks computational complexity analysis of the proposed GSINA method. It is unclear how the time and space complexity scale as the size of the input graphs increases. Moreover, there is no comparison of the running time or memory usage of GSINA compared to the baseline methods. Analyzing the overhead imposed by using optimal transport and Sinkhorn could quantify the tradeoff between accuracy gains and computational costs.

3. The Introduction section lacks specificity in explaining the innovative contributions of the proposed GSINA model for graph invariant learning. Additionally, the Approach section lacks adequate transition and explanation before formulating invariant subgraph extraction as an optimal transport problem. Addressing these weaknesses would enhance the clarity and logical flow of the paper's core contributions.

### Questions
1. The complexity analysis of GSINA is limited in this paper. Can you provide a detailed analysis of the computational overhead of GSINA compared to standard GNNs and other graph invariant learning techniques? What are the asymptotic complexities of key components like Sinkhorn attention and Gumbel trick?

2. This paper argues that sparse, soft, and differentiable are design principles for graph invariant learning (GIL). However, connectivity is another important consideration for extracting meaningful and interpretable invariant subgraphs in GIL. The invariant regions should represent coherent structures and patterns rather than disjoint disconnected components. How do you think about the connectivity of invariant subgraphs? Does the proposed GSINA model take this into account?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies graph invariant learning to discover the invariant relationships between graph data and its labels for different graph learning tasks under various distribution shifts. It adopts the optimal transport theory and designs one graph attention mechanism as a powerful regularization method for graph invariant learning. The experiments show the effectiveness of the method.

### Strengths
The strengths of this paper are listed as follows:
- This paper focuses on one important research problem which is graph invariant learning. It is very interesting to me.
- The writing is good in general. The motivations are clearly present. And the technical details are easy to understand.
- The experiments show the improvements on the baselines.

### Weaknesses
The concerns are from the following aspects:
- The technical contributions are a little straightforward to me since the key design for graph and graph OOD problem are not very well explained. For example, the design in section 3.1 is similar to GSAT [1] and the edge attention in section 3.2 is very similar to [2]. These differences with existing works are not very clear, which raises my concerns about novelty.
- Some theories are not well formulated. For example, it should formally define the graph distribution by considering the non-Euclidian graph properties. But the theories have weak connections with the graph itself.
- The experiments are not convincing enough since the authors seem to ignore some baselines (such as the graph-level and node-level method in [3]). Some of the results are confusing. For example, why some in-distribution results are lower than the OOD results.

[1] Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism. ICML 2022.
[2] Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure. NeurIPS 2022.
[3] Out-Of-Distribution Generalization on Graphs: A Survey. ArXiv 2022.

### Questions
See weaknesses part above

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In the context of graph invariant learning, this paper presents the GSINA model, which leverages optimal transport and graph attention to identify invariant subgraphs while satisfying the principles of sparsity, softness, and differentiability. Graph-level and node-level experiments on both synthetic and real-world datasets are carried out to demonstrate the effectiveness of GSINA.

### Strengths
(1)	The paper is skillfully written and exhibits a well-structured organization. 
(2)	The experiments encompass commonly-used datasets for this task and provide comprehensive comparisons with many typical methods, including CIGA and GSAT. I believe the experiments are robust and information-rich.
(3)	The majority of models relying on top-k selection extract hard subgraphs, often resulting in a notable loss of information in node-level experiments. In contrast, GSINA retains all edges by assigning low attention weights to variant part, ensuring the completeness of the graph structure.

### Weaknesses
（1） Using optimal transport theory to obtain differentiable solution to top-k selection has been adopted by other existing works, so I think the method of this paper lacks novelty. 

（2） The superiority of GSINA is marginal and not consistent. For example, for Graph-level OOD generalization performances in Figure 4 and Table 5, GSINA expresses worse results.

（3） The compared methods of different datasets are not unified. Baselines in Table2-4 are  less than Table 5.  And they ignored to compare some recently proposed methods: MoleOOD[1],  GIL[2],  Disc[3] and  GREA[4].

（4） The interpretable performance is not enough to verify the effectiveness of GSINA since  the used attention mechanism is expected to have stronger interpretability than other models.

（5） GALA[5] presents GALA for learning invariant graph representations without environment partitions under the proposed minimal assumptions. It is suggested to cite this paper.


[1] Learning substructure invariance for out-of-distribution molecular representations.

[2] Learning invariant graph representations for out-of-distribution generalization.

[3] Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure.

[4] Graph Rationalization with Environment-based Augmentations.

[5] Rethinking Invariant Graph Representation Learning without Environment Partitions

### Questions
See Weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new method (GSINA) for learning invariant representations when performing node or graph classification. The method is designed to provide sparse invariant features (subgraphs) like simultaneously ensuring differentiability. They draw inspiration from attention based strategies, which are differentiable, and top-k strategies, which are typically sparse but not differentiable. GSINA combines these two perspectives and uses proposes an optimal transport problem to obtain edge importances. The found invariant subgraph is then feed into the predictor to learn more generalizable features. The authors perform comprehensive experiments to demonstrate the efficacy of their method.

### Strengths
- The authors perform a very comprehensive evaluation across many datasets and tasks. Indeed, the proposed method does substantially outperform its competitors. For example, on by almost 10% with respect to CIGA on Table 4.

- The proposed method appears to be well-grounded. Maximizing the mutual information between a subgraph and the label is a popular approach in both the invariant feature learning and GNN-explanation literature. Moreover, the use of Optimal Transport on top of attention makes sense given the shortcomings of existing approaches. 

- Their method does not seem to require extensive hyperparameter tuning (just r), which uses the validation performance. This is especially helpful for OOD settings, where one cannot assume access to OOD data.

### Weaknesses
- I think the novelty is a bit lacking in the approach. Individual pieces seem like combinations of existing pieces. Perhaps I missed something and the authors could clarify this? Could the authors also clarify if the softmasks learnt by GSINA do in fact coincide with the known invariant signals in the graph (as per the explanation literature)? 

- Runtime/ Computational Complexity: I'm concerned that this method will substantially increase the runtime relative to the vanilla model and also other invariant representation methods. Could the others please provide some runtime plots so that I can understand if this is in fact the case? 

- The writing of this paper needs to be polished. For example, "The invariance optimization methods are based on the principle of invariance, which assumes the invariant property inside data, i.e. the invariant features under distribution shifts."

### Questions
Please see the weakness above, and below. 

Out of curiosity, I was wondering if GSINA could be applied to a pretrained GNN as a way of post-hoc improving the representations for better OOD generalization? Perhaps by enforcing consistency between the pretrained model's predictions on the original graph and the extract subgraph or through end-to-end fine tuning? 

Also, can the authors also please clarify if GSINA + the GNN Predictor are trained "end to end?" They mention a two-stage framework, but I just wanted this clarified. Maybe adding a pytorch-style algorithm would be beneficial.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
