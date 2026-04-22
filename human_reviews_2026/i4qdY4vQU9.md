# PROPGCL: Unleashing the Power of Propagation in Graph Contrastive Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Graph contrastive learning (GCL) has recently gained substantial attention, leading to the development of various methodologies. In this work, we reveal that a simple training-free propagation operator PROP achieves competitive results over dedicatedly designed GCL methods across diverse node classification benchmarks. We elucidate PROP’s effectiveness by drawing connections with established graph learning algorithms. By decoupling the propagation and transformation phases of graph neural networks, we find that the transformation weights are inadequately learned in GCL and perform no better than random. When the contrastive and downstream objects are misaligned, the attendance of transformation causes the overfitting to the contrastive loss and harms downstream performance. In light of these insights, we remove the transformation entirely and introduce an efficient GCL method termed PROPGCL. We provide theoretical guarantees for PROPGCL and demonstrate its effectiveness through a comprehensive evaluation of node classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates simple propagation operators in graph contrastive learning. The authors introduce PROP, a training-free propagation method that achieves competitive performance across node classification benchmarks. Through decoupling analysis, they reveal that existing GCL methods struggle to learn meaningful transformation weights while showing potential in learning propagation coefficients. Based on these insights, they propose PROPGCL, which eliminates transformation layers and only learns graph-adaptive propagation coefficients. The method achieves strong results on heterophilic datasets with significant computational advantages.

### Strengths
1.The core finding that simple propagation matches complex GCL methods is surprising and well-demonstrated. PROP consistently performs well across benchmarks, particularly on heterophilic graphs where many GCL methods struggle.

2. Thm. 4.1 now formally connects propagation to Dirichlet energy minimization with proper mathematical rigor. Thm. 6.1 provides theoretical guarantees for PROPGCL's advantage when contrastive and downstream objectives are misaligned. These additions address major theoretical gaps from prior versions.

3. The ablation studies in sec. 5.2 are particularly convincing. Tab. 3 shows that (1) GCL struggles to learn effective transformation weights even with optimal propagation coefficients, and (2) GCL can learn informative propagation coefficients when paired with well-trained transformation weights. This provides clear evidence for the main claims.

3. Experiments compare against diverse architectures including spectral-based models etc. The addition of large-scale experiments on ogbn-products addresses scalability. The efficiency analysis in sec. 6.4 shows 99% memory reduction and substantial training time improvements.

### Weaknesses
1. The title and narrative discuss "graph contrastive learning" broadly, but the method focuses on node classification. App. B shows a 2.82% performance gap on graph classification tasks. This suggests fundamental limitations when global graph representations are needed. The paper should either narrow the title and claims to node-level tasks, or provide deeper analysis of why propagation works for node classification but not graph classification.

2. Thm. 6.1 assumes the downstream-relevant component corresponds to low-frequency signals with smoothness. However, heterophilic graphs violate this assumption since connected nodes have different labels. Yet PROPGCL shows its strongest improvements on exactly these datasets. The paper should discuss when these assumptions hold and provide empirical validation of the decomposition in def. 6.1.

3. The paper frames PROP as performing implicit alignment in contrastive learning (thm. 4.2). Yet PROP requires no negative samples and no learned parameters. This blurs what constitutes contrastive learning versus effective preprocessing. Is the finding that simple propagation works well a contribution to GCL methodology, or evidence that the GCL paradigm is unnecessary? The paper should address this more explicitly.

### Questions
1.Fig. 2 shows training loss rapidly approaching zero for GCL with transformation. Could you also show validation or downstream performance curves? This would directly visualize the negative transfer effect. Does early stopping based on downstream validation help?


2. Tab. 2 shows random weights achieve 73.4% vs 72.8% for GCL-learned weights. Could you provide statistical significance tests? Have you tried other random initialization schemes beyond Gaussian?

3. The paper uses Chebyshev basis functions. How sensitive is PROPGCL to this choice? Do the conclusions hold across different filter families?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper claimed that matrix transform in traditional GCL degrades model performance. So, the authors removed it, purely relied on raw feature propagation. Moreover, they learned the coefficients to combine different order of graph Laplacian, enhancing propagation.

### Strengths
1. This paper gave a comprehensive analysis of each component in GCL.
2. The authors gave enough experiments and theorems to support their claims.

### Weaknesses
1. The experimental results on PROP is very strange. In table 1, PROP, a pure $A^kX$, can perfrom extremely well on heterophilic graphs. The authors explained this that $A^kX$ can smooth all k-hop neighbors. That's ture, but we know Dirichlet energy $H^TLH=\sum A_{ij}||h_i-h_j||^2$. For $A^k$, it becomes to $\sum A^k_{ij}||h_i-h_j||^2$. Therefore, minimizing Dirichelet norm just equals to make all nodes within k-hops have similar embeddings. This is against the fact that we have to differentiate neighbor nodes embeddings to work on heterophilic graphs. From graph spectrum, $A^kX$ is a definitely a low-pass filter, while only high-pass filter can perform well on heterophilic graphs.

2. In table 1, GSSL methods performed severely worse than supervised methods. This is also against consensus that GSSL methods are better than vanilla supervised GNN, like GCN.

### Questions
In figure 2 GCLw/o transformation, the loss nearly did not decrease, or the the model was not trained. So, if GCL is without transformation, which paramters are learnt in it? Since the propation part is just H_PROP, also without paramters.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PROP, a training-free propagation operator that aggregates k-hop neighbor features, establishing it as a strong baseline for self-supervised node classification by showing its competitive performance across homophilic and heterophilic datasets, even outperforming many dedicated GCL methods on heterophilic benchmarks.

### Strengths
The paper challenges the prevailing paradigm in graph contrastive learning (GCL) that complex parameterized encoders are indispensable for high performance.
By decomposing contrastive loss into downstream-relevant and irrelevant components, it proves that PROPGCL outperforms both PROP and traditional GCL when objectives misalign.
The paper conducts multiple experimental evaluations to validate its claims.

### Weaknesses
PROP and PROPGCL depend on the propagation step K, but the paper lacks a systematic analysis of how K optimally adapts to diverse graph structures and provides no heuristic for K selection without brute-force tuning.
The paper omits comparisons with recent lightweight GCL methods (e.g., SimGCL, LiteGCL) that also prioritize efficiency.
PROPGCL does not integrate with graph attention mechanisms to prioritize important neighbors, limiting flexibility for graphs with unevenly relevant neighbors. 
The paper compares PROPGCL with established GCL methods but omits newer lightweight GCL approaches that also prioritize efficiency. 
The paper does not test PROPGCL on inductive benchmarks with heterogeneous node types or dynamic graph splits.

### Questions
For the training-free PROP operator, how does its performance scale with graph size and density?
The paper reveals that GCL-learned transformation weights perform no better than random weights, but only tests Gaussian, Uniform, Kaiming, and Xavier random initializations. Do other randomization strategies yield different results, and could they potentially outperform GCL-learned weights more significantly?
How does Fix-prop SL compare to dedicated few-shot GCL methods, and can further tuning of hyperparameters improve its performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper claims that the feature transformation component of GNNs is detrimental in Graph Contrastive Learning, performing no better than random weights. The authors propose PROPGCL, which removes the transformation layer entirely and only learns propagation coefficients, and can outperform complex GCL encoders. The method is shown to be both effective, particularly on heterophilic graphs, and computationally efficient.

### Strengths
1. The paper analyzes a problem in GCL where complex transformation layers are often poorly learned and perform no better than random weights.
2. The proposed method is evaluated on heterphily graph benchmarks.

### Weaknesses
1. The novelty is a bit limited - the propagation of graph node features serves as a good representation is well studied in the literature
2. The paper excels at showing that the transformation fails but provides little insight into why. Why is the GCL objective sufficient for learning propagation coefficients $\theta$ but not transformation weights $W$?

### Questions
1. Why is the GCL objective sufficient for learning propagation coefficients $\theta$ but not transformation weights $W$?

### Soundness
3

### Presentation
3

### Contribution
2
