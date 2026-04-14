# Rethinking Structure Learning For Graph Neural Networks

- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
To improve the performance of Graph Neural Networks (GNNs), Graph Structure Learning (GSL) has been extensively applied to reconstruct or refine original graph structures. While GSL is generally thought to improve GNN performance, it often leads to longer training times and more hyperparameter tuning. Besides, the distinctions among current GSL methods remain ambiguous from the perspective of GNN training, and there is a lack of theoretical analysis to quantify their effectiveness. Recent studies further suggest that GSL does not consistently outperform baseline GNNs under the same hyperparameter tuning. This motivates us to ask a critical question: *Is GSL really useful for improving GNN performance?* To address this question, we first propose a new GSL framework, which includes three steps: GSL bases (i.e. node representations used to construct new graphs) construction, new structure construction, and view fusion, to better understand GSL. Then, our empirical studies and theoretical analysis show that the mutual information (MI) between node representations and labels does not increase after applying graph convolution on GSL graphs that are constructed by similarity, indicating GSL could be unnecessary in most cases. Our experiments fairly reassess the performance of GSL and reveal that adding GSL to GNN baselines or removing GSL in state-of-the-art models has negligible impact on node classification accuracy. We also report that pretrained GSL bases, parameter separation, and early fusion are effective designs within GSL. Our findings challenge the necessity of complex GSL methods and underscore the value of simplicity in GNN design.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This study examines critically on the role of Graph Structure Learning (GSL) in improving Graph Neural Networks (GNNs). 
Although GSL has been used to improve GNN performance by capturing semantically similar nodes, the authors claim that there is insufficient theoretical and empirical evidence to support its necessity. The authors also show GSL's training times and the need for extensive hyperparameter tuning.

the paper proposes a new framework for categorizing GSL methods into three components: GSL base generation, new structure construction, and view fusion, which provides a better understanding of GSL's elements.
According to the authors' empirical analysis, GSL does not consistently outperform standard GNN approaches. 
Their findings show that GSL fails to provide more mutual information gain than traditional methods. 
Finally, the authors conclude that non-optimization-based GSL methods are frequently unnecessary because the quality of the original GSL bases guarantees informative node representations. 
This calls into question the prevailing assumptions about GSL's role in effective GNN performance and suggests reassessing its importance in future GNN designs.

### Strengths
1. The authors present a GSL framework, which is a valuable contribution. It provides a common terminology and aligns the community on the same page, facilitating clearer communication. Additionally, a unified framework benefits the subsequent empirical comparisons and discussions.

2. The empirical findings are solid. The observations in Section 4.1, particularly Observations 1-3, offer clear takeaways for practitioners. Furthermore, the empirical comparisons in Section 5.1 are thorough and comprehensive.

### Weaknesses
1. The primary weakness of this work is its lack of novelty. The key conclusion (especially the second point in the contribution list) seems rather obvious, at least to those familiar with GSL methods. Even for those who haven't empirically tested GSL methods, the result isn't surprising. If one assumes that the GNN model is sufficiently strong, it logically follows that the GSL method would be redundant. GSL appears more like an intermediate tool, and a good GNN model should outperform or make GSL unnecessary in an end-to-end training/inference.

2. In the last paragraph of the Introduction, the mentioned "theoretical analysis" (Appendix B) feels more like a derivation than a true theoretical analysis. Moreover, the analysis is too coarse-grained and offers little insight. In line 83, the claim that "GSL bases serve as the upper bound" is poorly framed. The GSL bases in this context refer to features, while the upper bound is a scalar term derived from these bases. The way this upper bound argued is a non-professional theoretical claim.

### Questions
1. Novelty of the Main Contribution: Can the authors elaborate more on the novelty of their key conclusions, especially regarding the second point in the contribution list? The observation that a strong GNN model might diminish the need for GSL methods feels intuitive and not surprising. How does this work differentiate itself from previous studies in this area? Are there specific aspects of GSL not yet addressed by prior research that the authors are tackling?

2. Theoretical Analysis vs. Derivation: In the introduction, the authors reference a theoretical analysis presented in Appendix B. However, the analysis appears to be more of a mathematical derivation rather than a comprehensive theoretical insight. Could the authors clarify why they classify it as a theoretical analysis? How does this analysis provide deeper understanding beyond the mere derivation of terms?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper summarizes the designs of graph structure learning (GSL) methods in graph neural networks (GNNs) using the proposed GSL framework. The author proposes to theoretically assess the effectiveness of the GSL methods by using mutual information. Both the empirical experiment results and the theoretical analysis conducted in this paper suggest that GSL methods do not enhance GNN performance when evaluated under the same GSL bases, that is, the representation used for GSL and hyperparameter tuning.

### Strengths
1. The paper is clearly written and is generally easy to follow.

2. The paper conducts extensive empirical evaluations.

3. The paper conducts theoretical analysis, which leads to a similar conclusion as the empirical evaluations.

4. The conclusion obtained is interesting and can be helpful in designing efficient graph neural networks.

### Weaknesses
1. Only the node-level learning task, i.e., node classification, is studied in this work. However, GSL is also widely used for graph-level learning tasks, such as graph classification.

2. Only small-scale datasets are studied in this work. 

3. GSL methods are also widely used to improve the robustness of GNNs by purifying the perturbed graph structures. It would be interesting to see whether the observation and conclusion hold on adversarially perturbed graphs.

4. Existing works suggest GSL algorithms achieve the best results in scenarios with fewer labels available [1]. However, this work only studied the setting of splitting 50%/25%/25% of the nodes in train/validation/test sets. This data split setting differs from what is used in some of the GSL papers evaluated. I would suggest the author study the impact of label ratio on the effectiveness of the GSL methods.

[1] Li, Zhixun, et al. "GSLB: the graph structure learning benchmark." NeurIPS 2023.

### Questions
1. Does the conclusion hold for large-scale graphs, such as ogbn-arxiv?

2. Does the conclusion hold for GNNs with GSL for graph-level learning tasks, such as those in [1]? 

3. Does the conclusion hold for node classification when the graph structure, node features, or class labels are adversarially perturbed as the settings in [1, 2]? 

4. How does the label ratio impact the effectiveness of GSL methods? Can you make the same conclusion when fewer nodes are labeled?

[1] Li, Zhixun, et al. "GSLB: the graph structure learning benchmark." NeurIPS 2023.

[2] Zhiyao, Zhou, et al. "Opengsl: A comprehensive benchmark for graph structure learning." NeurIPS 2023.

### Soundness
2

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
4

### Summary
This paper analyzes the graph structure learning methods from the perspectives of mutual information. The analysis suggests that no matter which type of GSL methods are used, after feeding the same GSL bases to the newly constructed graph, there is no mutual information gain compared to the original GSL bases. The paper then re-evaluates the graph structure learning using the same GSL bases. The results verify the the analysis.

### Strengths
1. This paper approaches GSL from an interesting perspective of information gain.
2. This paper reevaluated GSL methods using the same GSL bases and show the GSL can not consistently improve the performance of GNNs.
3. The paper is overall well written.

### Weaknesses
1. While this paper presents an interesting observation of GSL. It is not clear how this observation could help move GSL forward. The key observation is that GSL may not be the major contributor of model performance improvment but rather some other components are. This is a huge claim that tries to deny the previous efforts in GSL. I wonder if there is any certain scenario where GSL is still helpful?
2. In Table 2, there is a difference between the result of the current paper and result from Zhiyao et al. (2024). Is there an explanation for this?

### Questions
See waeknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Paper summarizes the utility of Graph Structured Learning (GSL): Does inferring edges among nodes (when some edges or no edges are given) help classification tasks? For instance, given many examples, one can develop an MLP that can infer on all examples in parallel. Alternatively, one can induce edges among example pairs, and run a GNN on the features+induced graphs.

Authors show that: there are no cases where GSL is useful. In summary, either run an MLP or a GNN on the original graph. If edges are homophonous, then just use GNN on the real edges (no induced edges). If the edges are heterophilous, just ignore the edges altogether and just run an MLP.

### Strengths
* It is nice to know this summary [do not induce any edges!]
* The paper's arguments are easy to follow [though writing can be improved]

### Weaknesses
The mean weakness of the paper (reason for my rejection) is:

* The arguments are too general: GSL does not add information
* The construction is too-specific.

From reading the paper, I can only remove some generality from the main argument:

* GSL is not useful for **classification** settings where the graph construction function is set to non-learnable KNN.

Crucially, their edge function is the kNN graph. Their GNN is similar to GCN [Kipf&Welling], i.e., one that averages node features with their neighbors, at every layer. Using kNN edges and GCN should work well if neighbor nodes (i.e. k-nearest) have the same class as the center node. This assumption is not met for heterophilous graphs.

None of the findings seem to be surprising. I will make a few notes

* Why use GCN if you are connecting on-purpose heterophilous edges? Why not use something like MixHop which (promises to) handle heterophilous edges?

* Why use untrainable kNN?

Many methods show that graph structure is indeed useful for their application, including https://arxiv.org/abs/2007.12002 or https://arxiv.org/abs/1711.04043 -- the first has more sophisticated edge-creation function and the second is trainable

# Final note

I do like the paper, but I think it can be greatly simplified to one small argument: dont induce kNN graph use in a GCN, instead use MLP -- noting that kNN+GCN is one instantiation of GSL.

### Questions
* How did you calculate the mutual information? It seems that one must integrate over the input space.

### Soundness
3

### Presentation
2

### Contribution
3
