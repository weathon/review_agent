# WILTing Trees: Interpreting the Distance Between MPNN Embeddings

- Decision: Reject
- Scores: 1, 6, 3, 5, 8

## Abstract
We investigate the distance function implicitly learned by message passing neural networks (MPNNs) on specific tasks. 
Our goal is to capture the functional distance that is implicitly learned by an MPNN for a given task. 
This contrasts previous work which relates MPNN distances on arbitrary tasks to structural distances that ignore the task at hand.
To this end, we distill the distance between MPNN embeddings into an interpretable graph distance.
Our distance is an optimal transport on the Weisfeiler Leman Labeling Tree (WILT), whose edge weights reveal subgraphs that strongly influence the distance between MPNN embeddings.
Moreover, it generalizes the metrics of two well-known graph kernels and is computable in linear time.
Through extensive experiments, we show that MPNNs define the relative position of embeddings by focusing on a small number of subgraphs known by domain experts to be functionally important.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper analyzes several graph metrics in order to evaluate model performance and metric preservation.

### Strengths
The paper analyzes several graph metrics and sees correspondence between metrics on graphs and metrics on datasets and on MPNNs.

### Weaknesses
The paper lacks novelty, as it presents neither a new analysis nor the introduction of a new network. Its contributions fall short of the expectations for an ICLR-style conference, where higher levels of innovation and original research are typically required.
Specifically, Definition 4 (Evaluation Criterion for Alignment Between dMPNN and dfunc) doesn't capture any alignment between MPNN and func. Usually, the ratio of MPNN(G) -MPNN(H) and struct(G,H) is measured, and in this case, some previous papers showed theoretically, this ratio converges to zero for a specific sequence of graphs. High/Low of your proposed measure doesn't intuately mean something.
Genreally I really don't see any novelty or something surprising in this paper.

### Questions
Why did you take Definition 4 (Evaluation Criterion for Alignment Between dMPNN and dfunc) as a measure? What does it mean?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper suggests a way of understanding how MPNNs model the graph functional distance. Specifically, the author distills MPNNs into their proposed Weisfeiler Leman Labeling Tree (WILT) without changing the graph distance. The proposed algorithm operates on linear time, which yields optimal transport distance between Weisfeiler Leman histograms. Empirical analysis shows that the relative position of the embedding follows the importance of specific subgraphs, showing that MPNNs can capture domain knowledge.

### Strengths
**S1.** This work provides theoretical contribution in understanding the relationships between the MPNNs and structural distance.

**S2.** Some illustrative examples, e.g., from Figure 1 to 3 improve the readability of the manuscript.

**S3.** Extensive experiments show the validity of the proposed insight.

### Weaknesses
**W1.** This paper aims to show how Message Passing Neural Networks (MPNNs) define the relative position of embeddings. Starting from Definition 5, this manuscript suggests the WILTing Distance, which modifies the distance metric in Optimal Transport (OT) as d_path and stems from the mechanism of the shortest path metric on a tree [2]. Additionally, the author employs [3] for efficient (linear) computation. However, most of their contributions overlap with prior work [1], which proved that MPNNs have the same expressive power as the 1-Weisfeiler-Lehman (1-WL). From my viewpoint, the contribution of this work seems to be marginal unless it is compared with [1] properly.
  * [1] Fine-grained Expressivity of Graph Neural Networks, NeurIPS '23
  * [2] Fast subtree kernels on graphs, NeurIPS '09
  * [3] Wasserstein weisfeiler-lehman graph kernels, NeurIPS '19

Q1) Could you please elaborate on the difference between [1] and your work?  


**W2.** Most of my concern lies on the above question since the writing of this paper is very clear and the experiments are also interesting.

Q2) Could you please add [1] to the experiments as well?


I'm willing to increase the score if the above concern is addressed clearly.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates the distance of MPNN embeddings. The authors empirically found that the Euclidean distance of MPNN embeddings after training is aligned with the Euclidean distance of the graph labels. The authors then proposed a new graph pseudometric --- WILTing Distance --- for distilling MPNN embedding distance. The authors showed experimentally that the proposed WILTing Distance approximates the MPNN distance well, while revealing the important subgraph structure for the molecule property prediction tasks.

### Strengths
1. The study of graph distance and connection with graph neural networks is of high interest to the community.

2. The figures are nicely rendered.

### Weaknesses
1. The organization of the paper is hard to follow. The paper seems to have two independent parts. The first half (Sec 3 and 4) investigates the MPNN distance by comparing it with graph structural distances (task-independent) versus graph label distances (task-dependent). The second half (Sec 5 and 6) aims to distill the MPNN distance into the proposed WILTing distance. 

2. Unclear motivation. The first half seems rather intuitive: the MPNN embeddings (and thus their distances) are optimized to predict the target graph labels (in both classification and regression) and thus align with the target distances; the authors should justify and discuss more thoroughly why Q2-Q5 worth investigation. The second half touches on a few interesting aspects (e.g., optimal transport, distance upper bounds, MPNN interpretability, etc), but the authors did not connect them in a coherent way, nor dive deep in any of them. 

3. Limited contribution: The property of the proposed WILTing distance, and its connections with other recently proposed distances are not thoroughly discussed. See more details in the Questions.

### Questions
1. The purpose of the WILTing distance is to identity the important (learned) WL colors that strongly influences the MPNN distance. This can in turn be used to identify important edges or subgraphs that matters for the downstream task, providing a tool for MPNN interpretability. Is MPNN interpretability the main practical motivation of WILTing distance?
(a) If so, why not compare the important subgraphs identified from WILTing distance with other GNN interpretability tools (e.g. [1],[2]). What are the additional insights or advantages from using WILTing distance over existing interpretability tools?
(b) If not, what are other motivations of WILTing distance? Can it be a drop-in replacement of MPNN?

2. Expressivity of WILTing distance (Appendix B.4): The authors define $d_{\text{WL}}$ using the binary notion of expressivity in terms of distinguishing non-isomorphic graphs. However, recent works in [3], [4] have proposed a fine-grained, continuous notion of WL distances based on optimal transport of the induced measures of the WL colors, and the relationship between the continuous WL distances with the MPNN distance. It seems more natural and stronger to investigate the expressivity of WILTing distance under the continuous WL distance. Can the authors justify their definition and comment on the expressivity of WILTing distance compared to the continuous WL distance?

3. Relationship between WILTing distance and Tree Mover Distance [5]: The authors discuss the connections between WILTing distance and the graph edit distance (Thm 1) as well as Weisfeiler Leman Optimal Assignment distance (Thm 2). Intuitively, WILTing distance seems very similar to the Tree Mover Distance [3]: Can the authors compare them?




References:

[1] Yuan, Hao, et al. "On explainability of graph neural networks via subgraph explorations." International conference on machine learning. PMLR, 2021.

[2] Ying, Zhitao, et al. "Gnnexplainer: Generating explanations for graph neural networks." Advances in neural information processing systems 32 (2019). 

[3] Chen, Samantha, et al. "Weisfeiler-lehman meets gromov-Wasserstein." International Conference on Machine Learning. PMLR, 2022.

[4] Böker, Jan, et al. "Fine-grained expressivity of graph neural networks." Advances in Neural Information Processing Systems 36 (2024).

[5] Ching-Yao Chuang and Stefanie Jegelka. Tree mover’s distance: Bridging graph metrics and stability of graph neural networks. Advances in Neural Information Processing Systems, 2022.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper explores the metric properties of the embedding space in message passing neural networks. The authors observe that the embedding distances of MPNNs align with the functional distances between graphs, contributing to the predictive power of these networks. The primary contribution is the proposal of a Weighted Weisfeiler Leman Labeling Tree (WILT), which distills MPNNs while preserving graph distances. This WILT framework enables interpretable optimal transport distances between Weisfeiler Leman histograms, improving the interpretability of MPNNs by identifying subgraphs that influence embedding distances.

### Strengths
- As far as I am aware, the introduction of WILT to interpret MPNN embedding spaces is unique. By distilling MPNNs into WILT, the method is able to understand the role of specific subgraphs in determining functional distances. This seems especially useful in settings where interpretability is important. 
- By showing that MPNN embeddings naturally align with functional graph distances, WILT provides insight into why MPNNs achieve high predictive accuracy in certain tasks. This contribution enhances the field’s understanding of how MPNNs implicitly capture task-relevant structures in the embedding space, perhaps even opening the way for 'transferring' this knowledge. Moreover,  by offering a framework that generalizes high-performance kernels, the paper opens doors for developing kernels tailored to specific graph applications.
- WILT generalizes existing Weisfeiler Leman approaches. As these approaches are used in a wide variety of tasks, e.g. molecular prediction, making WILT a versatile tool. I especially like the approach runs in linear time, making it also applicable for e.g. large molecules. 
- I really appreciate the figures in the paper.

### Weaknesses
- Even though I appreciate the theoretical contributions of this paper, I think it would benefit significantly from more high-level intuition of the approach. The introduction is very short, and the paper is very condensed, providing little guidance for the reader. I would really urge the authors to move part of the formalism to the appendix and dedicate more space in the paper to building intuition behind the approach, as this is to me the major weakness in the paper.
- The paper would benefit from a more thorough comparison to recent interpretability approaches in graph learning, such as methods that use attention mechanisms or explainable subgraph extraction. I think this could really highlight the differences and benefits of this approach. 
- The empirical validation is limited and its effectiveness on other types of graphs (e.g., social networks, knowledge graphs) is not thoroughly explored. In molecular prediction tasks, we know that the topological information of the graph is very indicative of the predicted properties, but how beneficial is this work in these more subtle settings? Some non-molecular exploration would be hugely beneficial to judge the applicability of the framework. 
- There is a lot of work on extending the WL test to higher-order (e.g. simplicial, cellular etc). As WILT inherits the typical limitations of the WL test, it could perhaps benefit from these higher-order topological spaces, as the authors mention. This is claimed to be straight-forward, but some deeper reflections on this would be beneficial.

### Questions
- Have the authors considered adapting WILT to higher-order Weisfeiler Leman test? Or maybe using alternative graph matching approaches? 
- Given the efficiency of WILT, did the authors consider testing its scalability on high-dimensional datasets, e.g. social networks or molecular interaction networks? This would help demonstrate the method’s robustness across diverse graph types.
- Could the authors expand on how WILT compares to other interpretability methods in terms of capturing functional subgraphs, e.g. those based on attention?
- Can WILT work with incomplete graphs at all? What about directed graphs?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper aims to shed light on the workings of GNNs by investigating in how far the distance given by the graph embedding of the GNN is reflected in other graph distances. The authors find that the MPNN distance is not correlated to static graph distances that are oblivious to the task. However, it is related to the "functional" distance, which encodes the class label. The authors propose a novel technique, the WILT. The WILT is a tree, whose nodes are the colors of WL and whose edges connect preceeding colors to their successors in the iterations of WL. The WILT can be tailored to a specific problem by learning weights on the edges. The authors find high correlation between the WILT and MPNN performance.

### Strengths
- The paper provides a solid theoretical basis for the proposed methods, including proofs and detailed explanations of pseudometrics.
- By identifying important subgraphs, the paper enhances the interpretability of MPNNs, making it easier to understand what drives their performance.

### Weaknesses
- The experiments are conducted on specific datasets; it would be beneficial to see more diverse real-world applications to assess generalizability.
- The answers to the questions asked are pretty obvious beforehand. The structural distances that stem from non-trainable graph kernels have nothing to do with the task, therefore it is unreasonable to assume that an MPNN (before or after training) would be highly correlated (Q2, Q3) . The same goes for Q4, Q5, where the functional distance encodes the target, and is therefore what the MPNN is optimized for. While it is not inherently bad to ask questions that one expects the answer to, these questions, though many, create little new insight.
- The algorithm for learning the WILT weight is only discussed in the appendix.

### Questions
- How expressive is WILT? It implies a hyperbolic distance between colors, so intuitively, it should be weaker than MPNNs?
- How long does learning the WILT weights take? 
- Famously WL is extremely sensitive to noise in the graph structure. Does WILT handle structural noise and/or feature noise well?

### Soundness
4

### Presentation
3

### Contribution
4
