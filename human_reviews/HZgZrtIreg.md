# Generalizing Weisfeiler-Lehman Kernels to Subgraphs

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Subgraph representation learning has been effective in solving various real-world problems. However, current graph neural networks (GNNs) produce suboptimal results for subgraph-level tasks due to their inability to capture complex interactions within and between subgraphs. To provide a more expressive and efficient alternative, we propose WLKS, a Weisfeiler-Lehman (WL) kernel generalized for subgraphs by applying the WL algorithm on induced $k$-hop neighborhoods. We combine kernels across different $k$-hop levels to capture richer structural information that is not fully encoded in existing models. Our approach can balance expressiveness and efficiency by eliminating the need for neighborhood sampling. In experiments on eight real-world and synthetic benchmarks, WLKS significantly outperforms leading approaches on five datasets while reducing training time, ranging from 0.01x to 0.25x compared to the state-of-the-art.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work focuses on addressing challenges related to capturing arbitrary interactions both within and between subgraph structures.  To provide a more expressive and efficient alternative, this work proposes WLKS, a Weisfeiler-Lehman (WL) kernel generalized for subgraphs by applying the WL algorithm on induced k-hop neighborhoods.

### Strengths
W1. The research significance of subgraph learning in graph representation learning is undoubtedly valuable.

W2. The research in this manuscript is interesting, and the proposed framework of both efficiency and effectiveness is exciting.

### Weaknesses
W1. The analysis of existing challenges is unconvincing, and this manuscript lacks an analysis of existing works.  The author identifies the main challenge as capturing arbitrary interactions between and within subgraph structures. However, the local-global interactive learning strategy has been well studied by GNN-AK[1]. 

W2. This manuscript lacks sufficient interpretability analysis of the proposed method. For instance, the right panel of Figure 1 only shows the final coloring result, leaving us without an understanding of the iterative process of color refinement.

W3. I argue this work lacks a fundamental understanding: the goal of using WL-based deep learning methods or traditional machine learning methods is to improve the distinguishability of isomorphisms that would otherwise be indistinguishable. However, I do not see this insight or explanation reflected in the work.

[1] From stars to subgraphs: Uplifting any GNN with local structure awareness, iclr 2022.

### Questions
Q1. What is the purpose of the discussion and analysis in Sec. 2.2? The main content of Sec. 2.2 is the proof of Proposition 2.2, which shows that the expressiveness of $WL{S^{k + 1}}$ is not strictly greater than $WL{S^{k }}$. However, in the subsequent analysis, the authors do not seem to propose a method for designing $k$ based on this insight.

Q2. I am puzzled as to whether the design of $ k=0, D $ actually addresses the goal of capturing arbitrary interactions between and within subgraph structures, which is the stated research motivation of this work.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the challenge of effectively representing subgraphs in learning tasks by proposing a new approach called the Weisfeiler-Lehman Kernel for Subgraphs (WLKS), which adapts the classic WL algorithm for graph-level tasks to handle subgraph-level tasks by utilizing induced k-hop neighborhoods. By combining kernel matrices from different k-hop levels, WLKS captures complex subgraph interactions without relying on neighborhood sampling. The model achieves similar or better classification performance while reducing training time significantly on various datasets.

### Strengths
1.	WLKS improves subgraph-level task performance by encoding both internal subgraph structures and their k-hop neighborhoods, offering a more expressive alternative to traditional GNNs.

2.	WLKS achieves superior performance with reduced training times, outperforming several state-of-the-art models by less training time, without requiring GPUs, pre-training, complex hyperparameter tuning, or extensive pre-computation.

3.	WLKS can be integrated into GNN frameworks (like S2N) as an adjacency matrix, allowing users to benefit from both the structural expressiveness of kernels and the feature-rich nature of GNNs.

4.	The paper is generally well-written and easy to follow.

### Weaknesses
1.	WLKS depends on the node coloring algorithm, making it less suitable for datasets where nodes or edges have continuous feature values, which are challenging to map to discrete color values. It would be beneficial for the authors to discuss potential adaptations of WLKS for handling continuous node and edge features, or to clarify if existing techniques could address this limitation.

2.	The selection of k in WLKS is somewhat limited, focusing solely on the original subgraph and the entire global graph, which may be insufficient for larger, more complex datasets where intermediate k values could capture essential substructures. This is especially relevant given that most datasets in the study are not large (as shown in Table 1), raising concerns about how well this approach will generalize to larger graphs. It would be beneficial for the authors to provide theoretical or empirical justification for why using only k=0 and k=D is sufficient. Additionally, the authors could include experiments on larger graphs to demonstrate the scalability of this approach.

3.	WLKS leverages a basic WL color histogram on subgraphs and a similar histogram within the global graph. This approach, while efficient, may result in a loss of structural information (e.g., datasets requiring detailed interactions between subgraphs).

4.	The paper lacks a comprehensive discussion comparing WLKS to baseline models. For instance, although WLKS outperforms baselines such as SubGNN and GLASS on several benchmarks (Table 2), the paper does not explain clearly why WLKS might be more expressive in specific contexts (e.g., powerful in terms of distinguishing isomorphic graphs). It would be beneficial for the authors to provide a detailed analysis of key examples where WLKS outperforms the baselines, highlighting the structural properties that WLKS captures more effectively in these cases.

### Questions
1.	Could you provide further discussion on the selection of k? Specifically, does the optimal k value depend on the size or sparsity of the graph/subgraph?

2.	The paper does not employ labeling tricks, such as those used in GLASS. Could you clarify the reasoning behind this choice, particularly since distance-based labeling could be effectively integrated with WL kernels?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors tackle the problem of subgraph representation learning, and propose an simple, yet effective method based on Weisfeiler-Leman Kernels. The ideas is quite straightforward: that is, to run the WL algorithm over k-hop neighbourhoods around subgraphs of interest, to collect the output colour histograms, and to define kernels based on those. The authors show that choosing different ks together is beneficial from a discriminative point of view, and propose to consider a specific choice that guarantees a good trade-off between complexity and performance. Experiments validate the effectiveness of such trade-off and discuss additional analyses on the choice of k values and additional hyper-parameters. The authors additionally discuss hybrid approaches that combine their kernel with Graph Neural Network architectures, showing these can be beneficial depending on the relative importance of feature information w.r.t. structure.

### Strengths
- The paper is well written.
- The method is simple: it is easy to understand while offering a compelling trade-off between complexity and performance.
- Different aspects of the method are empirically analysed, while a couple of theoretical insights are offered.
- The combination of the approach with GNN models feels slightly arbitrary, but nonetheless it can already provide initial insights on the relative importance of feature information, and shows how the approach is compatible with other methods.

### Weaknesses
- The manuscript would benefit from a preliminary section that would more formally describe the problem setting of subgraph representation learning, and the shortcomings of state-of-the-art approaches which the authors seek to overcome.
- The paper lacks a more detailed, *formal* comparison with other state-of-the-art approaches. From a methodological standpoint, beyond experimental results, how is the method different than other approaches such as SubGNN? Is it capturing some information that this last cannot? What are, reasonably, the methodological components and technical features that lead WLKS to outperform them? A better contextualisation in terms of technical components and whether they are captured or not by other methods would also help readers chart a landscape of subgraph representation methods. Expanding Section 5 would go in this direction.
- The manuscript lacks a more direct discussion on the kind of information WLKS is employing. Is the method completely structural? How can features be included? Is the approach in subsection 2.5 the only way to do so?
- The proof for Proposition 2.2 does not seem complete. Essentially, it builds upon assumptions that are, themselves, proving most of the claim already (see “Assume that we have […]”, line 181 and similarly in line 187).
    - I believe that part of the proof should consist in showing these assumption effectively hold. The examples exhibited in Figure 2 already make a step in this direction, but the authors should better and fully formalise this. For example, technically, the subgraphs should be in the same graph.
    - Related to the above, I believe it would also help if the authors were more explicit in how quantification is made over $k$. Is the claim for any possible $k$? In that case, for example, one should show the existence of counterexamples for any possible $k$, perhaps with an inductive construction. Incidentally, exemplary pairs beyond $k=0$ could be more informative for readers.
    - Like 189 requires checks: the equation seems to be specific for $k=0$? Is this wanted?

### Questions
- What does the “Total training time” refer to exactly for WLKS? what does this term include?
- What about inference run-time? Can the authors provide some figures on that and discuss this in comparison to other approaches?
- Line “418”: what does “[…], which are universal for subgraphs […]” mean exactly?

Also, see “Weaknesses”. I am willing to improve my score after insightful clarifications / additions by the authors.

### Soundness
3

### Presentation
3

### Contribution
3
