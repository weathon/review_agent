# LaCore: Laplacian Cohesive Subgraphs for Graph Representation Learning

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Dense, cohesive subgraphs are valuable anchors for pooling and interpretation in graph representation learning (GRL), yet exact cliques are too strict and average-density heuristics are hub-biased and unstable. We introduce \textsc{LaCore}, a fast two-phase \emph{Laplacian-smoothed reverse peeling} method that rebuilds the graph in a fixed importance order and scores each \emph{connected} component with a smooth ratio that penalizes within-component degree variation. A simple one-step growth test yields a natural \emph{first-peak} stopping rule, and a degree-concentration certificate links low Laplacian energy to near-uniform internal support, making the selected subgraphs cohesive and interpretable. \textsc{LaCore} preserves the scalability of greedy peeling, running in $O((|V|{+}|E|)\log|V| + |E|k)$, and is learned-parameter-free when used as a pooling operator. On synthetic planted-subgraph recovery and graph classification benchmarks, \textsc{LaCore} consistently improves downstream GRL metrics. The result is a practical, stable alternative to density-only heuristics that plugs directly into modern GRL pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors focus on the graph cohesive problem. They propose LACORE, which employs Laplacian-smoothed reverse peeling to balance size and degree uniformity in cohesive subgraphs. The authors provide extensive experiments to validate the proposed method.

### Strengths
- Investigating cohesive graph is valuable, as it can offer insights for various studies in graph-related domains.
- The authors have designed a variety of experiments to comprehensively validate the model.

### Weaknesses
- The improvements offered by the proposed method in Table 2 appear marginal.
- The writing in the paper could benefit from substantial revisions to enhance overall clarity.
- The motivation of the paper is not clear, and there is inconsistency between the issues summarized in the abstract and those in Section 1 regarding current challenges in the field.
- It would be helpful to include an overview diagram of the method for greater clarity.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

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
The paper proposes LACORE, a reverse-peeling heuristic that rebuilds a graph in degeneracy order and scores each connected component 
 by a Laplacian-smoothed ratio, which favors dense, degree-balanced (hub-robust) subgraphs. The implementation keeps an incremental ΔQ with a k-degeneracy orientation and DSU, yielding total time O((∣V∣+∣E∣)log∣V∣+∣E∣k). The method is used (i) as a pooling operator for graph classification and (ii) as a model-agnostic explainer; synthetic planted-cluster recovery and TUDatasets experiments show improvements over several baselines.

### Strengths
- Simple, well-motivated objective connecting degree smoothness to cohesion via a clean certificate; intuitive stopping rule.
- Versatility across tasks. The same LACORE primitive is used as a graph-level pooling step and as a model-agnostic subgraph selector for explanation, without retraining or gradient access.
- The paper’s motivation and positioning are underdeveloped. C

### Weaknesses
- The abstract says LACORE is parameter-free when used as a pooling operator, yet \epsilon is a key parameter to tune.
- On synthetic graphs, the ablation of \epsilon peaks near 10^6. However, \epsilon is only pick as 0.1 for graph classification experiments. Can you explain this difference? A sensitivity study of \epsilon on these graphs could be helpful.
- The paper’s motivation and positioning are underdeveloped, and this paper provides limited discussion of existing dense subgraph discovery methods.
- Fidelity–sparsity is informative, but BA-2Motifs has ground-truth rationales; Report precision/recall/AUROC of recovered motif nodes/edges, is also common in explainer benchmarks.

### Questions
- Are there any specific failure modes of prior methods that your approach solves, and how your objective differs in principle from existing density/Laplacian-style methods.

### Soundness
3

### Presentation
2

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
This paper proposes LACORE, a novel algorithm for discovering cohesive subgraphs by leveraging a Laplacian-smoothed scoring function within a reverse-peeling framework. The method rebuilds a graph in reverse degeneracy order, scores each connected component and selects components based on a natural "first-peak" stopping rule derived from a one-step growth test. Experimental results show that LACORE outperforms strong baselines.

### Strengths
1.It is novel to creatively combine reverse peeling with Laplacian-smoothed score for cohesive subgraphs discovery.

2.This paper has a good theoretical analysis. The algorithmic complexity is rigorously analyzed.

3.The paper is generally well-written. The figures are well presented.

4.Experiments are comprehensive and experimental results demonstrate the proposed method outperforms baselines.

### Weaknesses
1.While the paper acknowledges that $\varepsilon$ needs tuning and provides an ablation study, the fact that the optimal $\varepsilon$ is highly sensitive to graph scale and density remains a practical limitation. The claim of being "parameter-free" in the abstract and for pooling is somewhat nuanced, as $\varepsilon$  is a parameter that requires careful selection for optimal performance, even if not "learned". The method's performance can be sensitive to this choice.

2.The cohesion certificate (Equation 4) is a central theoretical claim that provides a mathematical justification for the algorithm's output. However, the paper does not provide a proof or derivation for this equation. Its inclusion without proof or a direct citation to a specific source where this exact bound is proven diminishes the paper's self-containment and accessibility.

3.The paper claims in Section 4.1 that the $S_{L}(C)$  scoring function offers the advantage of a "Smooth objective → stable search". However, this key claim lacks support from direct experimental validation or rigorous theoretical analysis. While Figure 1(a) shows a smooth trajectory for one synthetic example, this is insufficient to demonstrate the universality of this property across diverse graph structures. Is this smooth, unimodal trajectory consistently observed across diverse graph types?

### Questions
1.The "peak-then-drop" trajectory of $S_{L}(C)$ is a key feature used for selection. Is this trajectory guaranteed for any graph, or is it an empirical observation? Are there graph structures where $S_{L}(C)$ might have multiple significant peaks? 

2.See weaknesses.

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
4

### Summary
This paper introduces LACORE, a topology-based algorithm for identifying "cohesive" subgraphs. The method employs a two-phase reverse-peeling heuristic that optimizes a Laplacian-smoothed ratio score. Its key component is the Laplacian energy of the subgraph's internal-degree vector, which penalizes degree variance and favors "degree-uniform" components. LACORE is evaluated as a parameter-free pooling operator for GNNs and as a model-agnostic GNN explainer on several benchmark tasks.

### Strengths
- The proposed Laplacian-smoothed ratio score $S_L(C)$, is a new and intuitive way to define subgraph cohesion based on degree uniformity, which provides a principled alternative to average-density heuristics.
- The proposed method shows strong performance on the synthetic planted subgraph recovery task. 
- The authors provide a scalable implementation with a bounded complexity, making it practical for large graphs.

### Weaknesses
- The relevance of LACORE to GRL is weak, as it is purely structure-based and completely feature-agnostic. The authors do not provide a clear argument for why and how a feature-blind structural prior is a desirable component for GNN-included GRL methods.
- The motivation for why the cohesive" subgraphs are critical structural components for GNNs is missing. It is unclear why this specific definition of cohesion is superior to, for example, identifying functional motifs or other structural patterns known to be important in the chosen TUDatasets (e.g., PROTEINS, NCI1).
- The claim that LACORE is a superior GNN explainer is overstated. The evaluation is limited to two small, well-known motif-based datasets (BA-2Motifs, MUTAG). The method's success here may be an artifact of these specific datasets. These results do not support a general claim of superiority over gradient-based methods, especially on more complex, feature-driven tasks.
- The claim of the method being "parameter-free" seems misleading. The $\epsilon$ regularizer is a crucial hyperparameter that controls the trade-off between size and smoothness and is shown to be highly scale-dependent (varying from $0.1$ to $10^6$). This requires careful per-dataset tuning, contradicting the "plug-and-play" implication.

### Questions
1) Why should the GRL community prioritize these specific "cohesive" subgraphs over other structural properties like motifs or communities defined by feature-similarity? The proposed method appears more aligned with classical network science, and its strong connection to representation learning is not clearly established.
2) For the pooling application, LACORE collapses dense subgraphs based on a homogeneous structural assumption. How does this interact with the GNN's message-passing? 
3) Can the authors justify why a GNN should be constrained by a structural prior that is completely blind to node features?

### Soundness
2

### Presentation
2

### Contribution
2
