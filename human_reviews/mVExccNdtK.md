# SAGMAN: Stability Analysis  of Graph Neural Networks  on the Manifolds

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 8, 3

## Abstract
Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability.  We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a framework for evaluating the GNNs' stability. Specifically, the metrics for defining the so-called DMD distance are selected as effective resistance distance. The initial graph is first embedded via its spectral characteristics, and a secondary graph is constructed by the KNN algorithm followed by the graph pruning (sparsification) approach. Hence, for any pair of nodes, e.g., p and q, their distance quotient on the embedded manifold is leveraged as the evaluation metric for measuring GNN (node level) stability. Experimental studies show promising results via graph data under adversarial attacks.

### Strengths
This paper well-organized the computational flow of inducing new DMD distance for evaluating GNN stability. 

The newly proposed algorithm has nearly linear time complexity and empirical study shows the GNNs under the guidance of the proposed model own greater stability compared to their original counterparts.

### Weaknesses
Some of the contents are unclear, making the paper difficult to follow. 

Some technical parts of the paper may need further justification.

### Questions
At the current stage, I have the following questions:

1. More content on how PGM acts on the graph data may be needed here. Also, if PGMs create a manifold for graphs, what is the metric of the manifold? Furthermore, does this (discrete) metric change over the GNN propagation? 

2. When presenting the notation of PGM and the reason for preserving the resistance distance is essential, the introduction of stable or unstable nodes needs to be enriched in Appendix A.6; I couldn't find any formal definition or threshold of identifying what exactly is stable/unstable nodes.

3. I think the stability induced by the algorithm is limited to the GNNs that smooth the node features. What will be the conclusion/influence for those GNNs that typically fit the heterophilic graphs or can induce a sharpening dynamic to the node features? As in this case, each row of X will tend to be dissimilar to each other. 

4. Perhaps a hyperparameter study for the algorithm with several components is necessary here.

5. Only to my view, as the proposed algorithm contains many components, they seems not be presented as a whole pipeline in the paper. Therefore, the author is suggested to refine the structure of the paper, e.g., adding some inter-contextual contents between each step. 

Some minor changes are suggested as follows: 
1. On row 234, the input embedding matrix $X = U_k$, I think $X$ should be the initial feature?

2. On row 759, the Appendix A.1 is self-refered.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper concerns the stability analysis of GNNs. The idea is inspired from the previously proposed Distance Mapping Distortion (DMD) metric. To match the requirement of having a near low-dimensional manifold for both input and output the authors have proposed to perform graph dimensionality reduction step for estimating the probabilistic graphical model. Additionally for scalability requirements the authors have also proposed spectral sparsification.

### Strengths
The paper studies an important problem of estimating the stability of GNNs
The framework is easy to understand and the computational complexity is linear w.r.t the graph dimensions.
Experimental evaluations validate the efficacy

### Weaknesses
The proposed framework although effective but largely uses a combination of many existing approaches.
The authors have clearly stated the distinction from the previous methods for analyzing GNN stability. However, the theoretical methods have guarantees. The current framework does not provide any strong theoretical guarantees. 
Theoretical results related to DMD could be extended however the authors need to show theoretical guarantees on their approach of estimating the low-dimensional manifolds.
The authors have mentioned that previous literature lack analysis on stability towards feature perturbations. However, the results in  [R1] could be seen as stability bounds of GNN w.r.t. Feature perturbations. The authors are suggested to include it in the discussion.
The framework is restricted only to evasion attacks and does not considers a practical perturbation setup where the input graph itself is poisoned

### Questions
In addition to the above weaknesses:
How the framework performs against more sophisticated graph structure and attribute attacks (Metattack, PGD). Could you perform analysis on adaptive attacks [R2].
How the model performs on OGBN datasets.
Empirical comparison with Lipschitz-based stability methods

### Soundness
3

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
The authors propose a new framework based on spectral theory, called SAGMAN, for stability analysis of GNNs. They analyze the distance distortions between the input graph manifold and output graph manifold as a measure of stability. To this end, the authors propose a distance preserving graph dimensionality reduction method to obtain a low dimensional input/output graph manifolds and use the DMD metric to characterize the stability. The authors show that this approach is scalable by showing that the time complexity is near-linear in the graph size. In the experimental section, the authors show that SAGMAN can be used in recommendation systems. Further, the authors show that this framework can be used to facilitate adversarial targeted attacks.

### Strengths
The paper is well-structured and well-written.

SAGMAN can be used to quantify stability at the node level, which I think is quite useful.

The paper has a significant theoretical component, backed up by some experimental data as well.

### Weaknesses
The paper evaluates SAGMAN mainly on node classification and recommendation tasks. Testing on other GNN tasks, like link prediction or community detection, could strengthen the generalizability of the findings​

SAGMAN needs to be compared with other robustness techniques to get a better idea of what specific advantages it offers versus what are the limitations

I think that ignoring d (which in theory can be O(|V|) for a complete graph) from the complexity analysis and claiming that the method is near-linear in time complexity in the size of the graph is slightly misleading.

### Questions
SAGMAN’s effectiveness depends on the input graph being well-represented in a low-dimensional space and having a significant eigengap. Is there any fix for this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a complicated framework called SAGMAN to measure the stability of trained GNN model. 
It has three phases. 1. Construct the graph embedding based on node features and spectral features.
2. Use probabilistic graphical models (PGMs) to build low-dimension input graphs in the manifold. 
	At the same time, the node embeddings of GNN outputs are also used to build low-dimension graphs in another manifold via PGM.
3. Use the  distance mapping distortion (DMD) calculations to measure the GNN stability.

### Strengths
It is interesting to find another metric to measure the GNN stability, and cooperating with manifold via PGMs is natural.

### Weaknesses
1. Graph robustness is a widely discussed problems.
However, a comprehensive discussion of related works is needed in this literature.
Moreover, more baselines and benchmarks are needed to be provided in the experimental part.
Survey paper [1] can help you gain more understanding about more recent works.
[1] A comprehensive survey on trustworthy graph neural networks: Privacy, robustness, fairness, and explainability.

2. For this metric of GNN stability, is there any conclusion from the experiments? 
Does various graph robustness algorithms have been applied with this new metric?
Any comparison between this metric to other graph stability metric? For example, the prediction drop.

3. From my opinion, applying DMD on graphs is a problem as the manifold on graph is hard to define.
Although the PGMs can provide a solution, how good this PGMs can form the graph manifold is still a considerable questions.
Will the change of graph topology influence the mapping between graph structures and labels? This is also need to be analyzed.

### Questions
1. In Figure 1, do the three graphs represent internal views of the same input graph within the framework, or are they independent examples with no connection to each other?

2. How about the efficiency of the proposed algorithm?



-----
After rebuttal:
Thanks for your effort for the reply, and I will keep my score. My main concern remains the insufficient discussion and comparsion of related works in graph robustness area [1]. Thanks for your understanding.

[1] A comprehensive survey on trustworthy graph neural networks: Privacy, robustness, fairness, and explainability.

### Soundness
2

### Presentation
2

### Contribution
2
