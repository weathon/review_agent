# Graph Transformer Neural Processes

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Neural Processes (NPs) are a powerful class of model for forming predictive distributions. Rather than use an assumed prior over functions to form uncertainties---as is done with Gaussian Processes---NPs can meta-learn uncertainties for unseen tasks; however, in practice meta-learning uncertainties may require a great deal of data. To address this, we propose representing the inputs to the model as a graph and labelling the edges of the graph with similarities or differences between points in the context and target sets, allowing for invariant representations. We propose an architecture that can operate over such a graph and experimentally show that it achieves strong performance even when there is limited data available. We then apply our model on three real world regression tasks to demonstrate the advantages of representing the data as a graph.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a graph-based Neural Process architecture that encodes pairwise similarities between context and target points to learn invariant representations, enabling more data-efficient uncertainty estimation and achieving strong performance on real-world regression tasks.

### Strengths
1.	The paper is well-written and easy to follow.
2.	GTNP achieved state-of-the-art performance on a number of real-world datasets.

### Weaknesses
1.	The authors claim to improve NPs by representing inputs as graphs, but the motivation for why graph structure is suitable remains underdeveloped. The paper feels more like a straightforward architectural variation on Transformer NPs.
2.	Although the paper proposes a GTNP framework, the core ideas are pretty incremental. The novelty is largely in adding edge embeddings, which is conceptually simple and lacks a clear new theoretical contribution. 
3.	The experiments lack ablation studies isolating the contribution of graph edge features vs. standard self-attention.

### Questions
1.	Have the authors conducted ablation studies to isolate the effect of the graph structure from that of the Transformer backbone or hyperparameter tuning?

### Soundness
2

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
The paper aims to solve regression problems with uncertainty modeling by representing the regression data as a graph whose edges are given by a predefined similarity measure and directly applying a graph neural process on this graph. Experiments were conducted on 3 scientific pmodeling tasks that show promising results compared with 4 baselines.

### Strengths
The experiments show better performance than the 4 chosen NP baselines. 
The writing is clear and easy to follow.

### Weaknesses
The paper suffers from several key weaknesses:

**Novelty**

There have been works on Graph NPs that represent data as graphs and apply NPs on these graphs [1, 2]. While I appreciate the authors pointing out the relevant literature in Section 4, I think a more thorough conceptual exploration of how GTNP positions against these methods is necessary. I have yet to see the conceptual novelty of GTNP as it simply just encodes the x's as nodes and leverages kernels to model the edges, which is a standard technique in graph neural networks. Moreover, directly applying graph transformers over a graph for regression is also a standard procedure in graph learning. The core technique of GTNP is identical to [1] with the only difference being the data preprocessing step to convert the regression data into a graph.

**Experiments**

- The baselines in the papers are limited to non-graph NPs. Since the proposed GTNP is part of the family of graph NPs, comparing GTNP with these baselines [1, 2] would strengthen the results of the paper.
- The graph construction step scales quadratically with the number of data points, as noted by the authors in appendix C8. This limits the scaling ability of the proposed method.

[1] Andrew Carr, David Wingate. Graph Neural Processes: Towards Bayesian Graph Neural Networks 

[2] Hu et al. Graph Neural Processes for Spatio-Temporal Extrapolation, SIGKDD 2023

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the Graph Transformer Neural Process (GTNP), a variant of Neural Processes that introduces invariance by representing context and target points as nodes in a fully connected graph, where edges are labeled using pairwise functions of their coordinates (typically norms). The model employs a graph transformer to propagate information, incorporating edge features into the attention mechanism. Experiments cover synthetic Gaussian process regression as well as real-world applications (fusion, weather, and lipophilicity). The authors claim improved sample efficiency and stronger uncertainty estimation compared to existing NP variants such as AttnLNP, ConvCNP, and TNP.

The key idea, from the perspective of this reviwer, semmes to be to induce invariances (e.g. translation/rotation) by encoding pairwise relationships via a graph-transformer attention mechanism. While representing data as a graph is not new in general, its application within the Neural Process framework, particularly using edge-aware attention rather than standard DeepSets or grid-based convolutions, is an interesting engineering variation. However, similar ideas of encoding relational inductive biases or using GNN/Transformer hybrids for meta-learning and irregular sampling have appeared before (e.g. Neural Processes with structured kernels, Function Neural Processes, Conditional Graph NPs, and variants of Graph-aware or Edge-enhanced Transformers). The authors acknowledge these works, but it is not clear if the benefit in for example labeling edges makes for a non trivial contribution. The conceptual novelty is in this view modest, mainly an architectural adaptation rather than a fundamentally new principle.

### Strengths
- Clear motivation to achieve invariance without requiring gridded data, extending ConvCNP ideas to arbitrary dimensions.
- The approach integrates graph-based pairwise embeddings into the attention mechanism effectively.
- Systematic experimental comparisons across synthetic and three real-world tasks.
- Demonstrates improved sample efficiency, particularly in low-data regimes.
- The paper is generally well-structured and includes meaningful baselines. A reasonably broad set of examples is offered; synthetic, physical system, environmental, molecular. The results suggest robustness and generalization of the architecture across data types.

### Weaknesses
- The proposed “graph transformer” is effectively a fully connected transformer with pairwise edge features; this scales as O(N²) and may be costly for large N.
- The use of only Euclidean-norm edge features is somewhat limiting; no ablation shows whether this adds information beyond what the transformer attention could learn implicitly.
- Reported metrics (especially the “standardized score” and log-likelihoods) are difficult to interpret; more intuitive visualizations of uncertainty and predictions would strengthen the presentation.
- The model’s relation to prior “graph-aware” or invariant NP variants (e.g. CGNP, FNP) could be discussed more thoroughly, to clarify the exact incremental contribution.
- While results are strong, the degree of novelty relative to existing Graph Neural Process or Edge-enhanced Transformer frameworks remains unclear.

### Questions
- How does the graph transformer differ functionally from an edge-enhanced attention layer as in standard edge transformers (e.g., Shi et al., 2020)?
- How does it compare against structurally closest graph-based NPs, such as Conditional Graph NPs (CGNP) (Nassar et al., 2018) and Function NPs (FNP) (Louizos et al., 2019)? I did not see this in the experiments.
- Did the authors benchmark other types of edge functions ϕ(xᵢ, xⱼ)? Would learned pairwise kernels yield further gains? 
- Only the Euclidean norm is used for edge encoding. It would be useful to include a test with other pairwise features or with the edges turned off to isolate the contribution of the graph formulation.
- How does performance and training time scale with N compared to TNP?
- Can sparsification or local graph construction be used to alleviate the quadratic cost?
- For the real-world tasks, can qualitative examples (e.g., spatial uncertainty maps for weather or plasma) be provided?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the "Graph Transformer Neural Process" and empirically evaluates it.  The GTNP borrows ideas from transformers, neural processes, and graph neural networks.  It defines a neural process that iteratively updates per-vertex representations of the data, with a sort GNN-inspired message passing algorithm that uses transformer-style attention weights.  Empirically, the algorithms performs well compared to other modern algorithms on three real-world test cases.

### Strengths
I generally liked this paper. I thought the formulation of the problem was pretty straightforward, and the architecture generally makes sense.

* A graph-based attention mechanism makes sense, and iteratively updating vertex representations is clearly a workable thing (as demonstrated by GNNs)

* The algorithm seems to perform well in practice, dominating other models in terms of log likelihood

* The paper isn't especially novel, in the sense that it combines a lot of ideas that have already been explored, but it appears to do so in a tidy and reasonable way.

* I liked the effort the authors made in terms of applying the model to real-world problems.

### Weaknesses
*.I thought the paper was generally well-written, but I think it could have been better organized.  I'm especially thinking about Section 2, which unnaturally mixes a lot of background information with the explanation of the actual GTNP. I think the authors could do much more to clean up their exposition.  I also felt that the final discussion section was weak.

* While the vertex updates were reasonably well explained in the equations in Section 2.1, I feel like an explanation of the way edge information was used was almost non-existent.  It seems that edge information is computed once, and then never updated again (in contrast to the vertices, which are updated at every iteration).  Is that right?

* I wish the authors had given us more compared parameter counts of the different algorithms, or mentioned parameter counts at all.  As it is, it's very difficult to know if the increased performance of GTNP vs. the competitors is due to the architecture, or is simply a result of having more parameters. This is compounded by the "compute time" discussion in the appendix: the GTNP takes 2-3 times longer to train. Why is that? And it's unclear to me why the inference is a blazing 6x faster than other similar algorithms.

* While I generally appreciated the significant effort the authors went to to apply this algorithm to real-world problems, I felt that the time could have been better spent exploring the contours of the algorithm itself.  I think a lot of the detail in the problem descriptions could have been relegated to an appendix, for example, and the authors could have then explored things like ablations (what happens if edge information is deleted? what happens if all vertices are initialized to 0? etc.) and scaling curves (what happens as the dimension of the embeddings increases?)

* I was surprised to see that all of the graphs were fully connected.  To me, one of the most interesting aspects of graph-based algorithms is the flexibility to define non-fully-connected topologies.  Why wasn't that considered?

* I generally didn't like the Appendix.  By introducing GEENA in the appendix, it's like you've written a whole second paper and hidden in there.  While I understand that people might have questions about what happens if you update edge information, it just seems like it should be a standalone follow-on publication.

### Questions
See weaknesses section

### Soundness
3

### Presentation
2

### Contribution
2
