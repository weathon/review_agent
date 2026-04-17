# Node-wise Filtering in Graph Neural Networks: A Mixture of Experts Approach

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Graph Neural Networks (GNNs) have proven to be highly effective for node classification tasks across diverse graph structural patterns. Most GNNs employ a uniform global filter—typically a low-pass filter for homophilic graphs and a high-pass filter for heterophilic graphs. However, real-world graphs often exhibit a complex mix of homophilic and heterophilic patterns, rendering a single global filter approach suboptimal. While few methods have introduced multiple global filters, they often apply these filters uniformly across all nodes, which may not effectively capture the diverse structural patterns present in real-world graphs. In this work, we theoretically demonstrate that a global filter optimized for one pattern can adversely affect performance on nodes with differing patterns. To address this, we introduce a novel GNN framework Node-MoE that utilizes a mixture of experts to adaptively select the appropriate filters for different nodes. Extensive experiments demonstrate the effectiveness of Node-MoE on both homophilic and heterophilic graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes NODE-MoE, a mixture of experts approach for GNNs that aims to select node-specific filters to help improve learning over both homophilous and heterophilous nodes in a graph. To decide which filter to use, the gating mechanism uses a set of feature similarity-style terms across a node's neighborhood, routing to either a soft mixture of all experts or a top-k set of experts. On both homophilic and heterophilic datasets, NODE-MoE demonstrates moderate performance gains across a variety of baselines. Moreover, this is supported by theoretical analysis on a CSBM model with a linear classifier that demonstrates a global filter is insufficient to learn across nodes with different patterns. Overall, I think the paper does a good job of presenting a reasonable idea, but with some shortcomings in justifying some aspects of it.

### Strengths
1. I believe the presentation of the work is a strong point, it is relatively easy to follow and the ideas logically build on one another throughout the work. I like the general flow of the paper. 
2. The empirical results seem promising, and I appreciate the authors added larger datasets, like Pokec, to demonstrate the scalability of the method. While I did not spend a ton of time going through the appendix, I also see there are tons of additional ablations and analyses, which I think is beneficial for the work. 
3. I think the theory is a good starting point to establishing why we would want to use an MoE style framework with different expert designs.

### Weaknesses
1. I have a few concerns about the theoretical analysis and what it conveys about the work. 
    - First, the theory relies on a relatively simple set up, with a linear model and binary classification. I don't feel as though the authors necessarily justify this choice, and explain why it is okay to neglect the designs that typically go into GNNs. 
    - Second, theorem 1 and 2 seems to argue the failure modes of GNNs with a global filter, but not how to solve the issue. The authors include theorem 3 to address this, explaining why it is okay to use the features to separate classes. However, I don't feel as though theorem 3 actually justifies the design. Specifically, the authors set up the CSBM in such a way that the feature-neighbor similarity correlates with homophily, and the features and structure are aligned. Thus, the theorem is just re-stating the assumptions of the CSBM model, as opposed to justifying why using these features would be meaningful in real-world applications. Why I think this is problematic is the authors argue that previous works are designed around unjustified heuristics, yet, these are also heuristics that are only justified in a very particular setting. 

2. Building on the point around heuristics, the other core justification the authors make for their method is moving beyond post-fusion/fixed-filters used by previous methods. Specifically, the authors claim that the advantage of their method comes from their node-wise filtering. However, I feel like, again, this is driven by heuristics, rather than a fundamental difference to previous methods. Within Node-MoE, the authors still use soft-routing which leverages all filters, but later apply a top-K gating to decide which to keep. There is no clear justification for why this would work, and feels like a heuristic-based choice that is not sufficiently justified over previous methods. 

3. The performance results are decent, but mixed in some cases. I do not think this is a true negative, assuming the follow-up analysis can thoroughly explain why certain behaviors are happening. That said, I think there are some weaknesses in the post-hoc analysis that do not necessarily convince me that Node-MoE is doing exactly what the authors claim. 
    - First, in figure 5, the authors show that the average weights change across homophily level. Yet, expert 1 (the high-pass filter) receives the highest scores for all bins. If this is the case, what is the point of the other expert? I also feel the authors did not provide enough justification for figures 4 and 5. For instance, on line 391 the authors say "This pattern confirms our design that nodes with varying structural patterns require different filters, demonstrating the effectiveness of the proposed gating model", yet this is the premise -- we already know nodes require different filters. I would expect an actual takeaway to show that Node-MoE effectively exploits this information and can adequately route nodes to different models that are aligned with their needs. If we were to take my point above about the weights and the dominance of expert 1, it makes it seem like routing is not actually happening. 
    - For figure 6, what is the performance of the more difficult regions, like the heterophilous nodes of Cora and homophilous nodes of Chameleon, when using an MLP? One concern I have is that the GNN models are known to under-perform a basic MLP in these regions,  and it possible that simply using an MLP in these regions can outperform Node-MoE. Moreover, I would be interested to see Mowst for Figure 6 given it routes between GNN and MLP. The more fundamental question I am asking is whether Node-MoE actually is leveraging the graph structure in these other areas, or simply choosing to rely less on it.

### Questions
1. Can you provide more justification as to why the strong assumptions for theorem 1 and 2 are justified? If they aren't, is it possible to relax some of these? 
2. Less of a question, but I think Theorem 3 needs to be reconsidered given the points I made above. 
3. I think the paper uses very strong wording around why previous methods are inferior, but then tend to fall into the same paradigms. This can potentially get solved by bolstering the theory, but I also think the paper needs re-framing to not over claim. 
4. Can the authors further discuss, going beyond figures 4 and 5, what is happening with the routing and how we can be certain the MoE framework is behaving as expected? 
5. For figure 6, what happens if you use an MLP or Mowst and regenerate these plots.

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
This paper studies the node classification problem. It proposes to address the problem where different part of the graph has different homophily. Its core idea is to apply the mixture-of-experts (MOE) framework on it so that every node would be encoded by different filters.

### Strengths
S1. The whole framework is intuitive and easy to understand.

S2. The motivation of the proposed method makes sense.

S3. The experimental part compares various datasets and methods.

### Weaknesses
W1. The novelty of the proposed method is low. The idea of "applying different filters to different nodes" has been used widely years ago, for example in the model FAGCN [1] , ACM-GCN [2], ALT [3] where every node's filter is tailored. E.g., Figure 3 from this paper and Figure 2 from ALT [3] are quite similar.

W2. I think this paper does not report the best performance of baseline methods. E.g., for ACM-GCN, please check its table 2 from (https://arxiv.org/pdf/2210.07606), which achieves 89.75% on Cora. Also, other competitors from (https://arxiv.org/pdf/2210.07606) seems very capable, even better than the proposed method, in this paper. 

[1] Bo, Deyu, et al. "Beyond low-frequency information in graph convolutional networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.

[2] Luan, Sitao, et al. "Revisiting heterophily for graph neural networks." Advances in neural information processing systems 35 (2022): 1362-1375.

[3] Xu, Zhe, et al. "Node classification beyond homophily: Towards a general solution." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.

### Questions
Please check the weaknesses I mentioned.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a limitation in Graph Neural Networks (GNNs) by proposing a multi-filter approach. It argues that real-world graphs contain a heterogeneous mix of homophilic and heterophilic patterns, so a single filter is suboptimal. The authors provide theory showing that globally optimized filters can harm performance on mismatched substructures, while node-wise filtering can achieve linear separability under mild conditions.

The main technical contributions are:

- A theoretical analysis under a mixed Contextual Stochastic Block Model (CSBM) demonstrating:

  - Global low-pass filtering yields near-optimal separability on homophilic nodes but induces a lower bound on loss for heterophilic nodes.

  - Node-wise application of different filters achieves linear separability across nodes with high probability.

  - A simple, local structure-based criterion can separate homophilic vs. heterophilic nodes with high probability.

- NODE-MOE, a Mixture-of-Experts framework that assigns filters on a per-node basis via a gating network, incorporating local structural cues for structure-aware expert selection.

- Expert models implemented with learnable spectral filters (e.g., ChebNetII) diversified via distinct filter initializations to encourage specialization.

- A filter smoothing loss that reduces spectral oscillations of learned filters to improve trainability and interpretability.

- A Top-K gating variant that activates only the most relevant experts per node, improving efficiency while preserving accuracy.

- Extensive experiments across various datasets show consistent improvements over baselines and MoE alternatives.

### Strengths
- Originality: The paper reframes multi-filter GNNs as a node-wise filtering problem, grounding the gating in local structural signals derived from theory. This surpasses post-fusion methods that rely solely on node embeddings.

- Quality: Theoretical results (Theorems 1–3) are coherent, well-connected to the design, and address why global filters fail in mixed regimes. The broad empirical study covers diverse datasets, multiple baselines, and MoE variants. Ablations on gating backbones, Top-K gating, experts, smoothing loss, and noise add credibility.

- Clarity: The problem setting and motivation are well-laid out, with a toy example contrasting global vs. node-wise filtering. The method is explained in modular components with intuitive rationale. Visualizations of learned filters and gating weights aligned to homophily are helpful.

### Weaknesses
- Theoretical scope and assumptions:

  - The analysis relies on mixed CSBM and linear classifiers, these restrict direct applicability to modern deep GNNs with nonlinearities. 

  - The transition from Theorem 3’s distance criterion to the specific gating input ‎`[X, |AX − X|, |A^2X − X|]` is plausible but not formally tied. 

- Methodological clarity:

  - Inconsistency on the expert backbone: Section 4 positions ChebNetII as the expert; Appendix C.4 says “we adopt GCNII as the experts.”

  - Load balancing and Top-K gating details: The paper notes use of a Shazeer-style load-balancing loss but the exact form and its influence are not deeply analyzed. 

  - The explanation of the use of GIN as gating model “neighboring nodes are likely to receive similar expert selections” is not well-grounded. Also the details of the GIN model is not given.

  - The ability of ChebNetII to learn diverse filter pattern largely depends on the chosen polynomial order, also Chebyshev polynomial has limitations in approximating nonsmooth functions, e.g. band-pass filters. This limitation is not discussed. Ideally it should be compared with other learnable spectral filters such as ARMA.

- Complexity
  - The filter smoothing loss can be expensive to compute given it requires eigenvalues.

- Presentation
  - The text on figures are too small and hard to read

### Questions
- Theorem 3 motivates a local feature-neighborhood distance. Could you formalize why two-hop differences $|A^2X-X|$ add complementary signal beyond one-hop in the gating? Any empirical evidence that 3+ hops saturate or hurt?

- What is the precise form of your load-balancing loss in Top-K gating, and how sensitive are routing distributions to its weight? Please include routing statistics (utilization per expert, imbalance ratio) and its relation to accuracy.

- The smoothing loss curbs spectral oscillations. Why is filter smoothness favored? And does the loss requires explicitly computed eigenvalues?

- In mixed-community graphs, does the gating align with detected communities? If you run community detection and aggregate gating decisions per community, do you observe coherent expert specialization? This could strengthen the interpretability story.

- How does NODE-MOE compare to a “hard assignment” baseline where nodes are first classified as homophilic/heterophilic by a heuristic (e.g., $||X_i - \frac{1}{D_{ii}}\sum_{j\in N(i)}X_j||$ threshold) and then routed to fixed filters? This would test the value of end-to-end learned gating and learnable filters.

- Why and how do you use GIN as gating model? How much layers of GIN you used? If you used layers > 1, theoretically the GIN model can also learn the 1-hop and 2-hop differences in the input, so are the 1-hop and 2-hop differences still needed in the input?

### Soundness
3

### Presentation
3

### Contribution
2
