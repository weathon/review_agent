# Improving expressivity in Link Prediction with GNNs via the Shortest Path

- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Graph Neural Networks (GNNs) often fail to capture the link-specific structural patterns essential for accurate link prediction, since their node-centric message passing might overlook the subgraph structures connecting two nodes. Prior attempts to inject such structural context either suffer from high computational cost or rely on oversimplified heuristics (e.g., common neighbor counts) that cannot capture multi-hop dependencies. We propose SP4LP (Shortest Path for Link Prediction), a new framework that integrates GNN-based node encodings with sequence modelling over shortest paths. Specifically, SP4LP first computes node representations with a GNN, then extracts the shortest path between each candidate node pair and processes the sequence of node embeddings with a sequence model. This design allows SP4LP to efficiently capture expressive multi-hop relational patterns. Theoretically, we show that SP4LP is strictly more expressive than both standard message-passing GNNs and several leading structural feature methods, positioning it as a general and principled framework for link prediction in graphs. Empirically, SP4LP sets a new state of the art on many standard link prediction benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SP4LP computes node embeddings with a GNN, then encodes the shortest-path node sequence between endpoints to build a path-aware link representation that is provably more expressive and achieves SOTA on HeaRT benchmarks.

### Strengths
Sound theory;
Completed proofs;
The representation can include both feature and path information.

### Weaknesses
Is computing shortest paths too slow on large graphs?

 Path-based representations can be unstable: if any edge on a path is (re)predicted to exist, the chosen path may change and thus alter the representation. So the method is not isolated to downstream training. Even when using only the shortest path, the hop length is usually be capped to keep structure local and reduce variance and such unstability. 

 If only the shortest path is used, paths may frequently pass through high-degree “popular” nodes, making many representations highly correlated. Would multiple paths or de-hubbed routing help?

 A shortest path may not encode semantic signals; in an unsupervised setting this adds uncertainty. It would help to discuss when the method is appropriate and to design pre-application diagnostics to evaluate data suitability.

 Using the representation to inject even a small amount of structural context is reasonable, but please clarify what could be lost by focusing on shortest paths (e.g., robustness vs. information loss, local vs. global structure) and quantify the trade-offs.

### Questions
- Should we see the robustness of the method with possible longer path been invovled? Even they are the shortest path.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper tackles a well‑known issue in link prediction with message‑passing GNNs: when we form a link embedding only from the two endpoint node embeddings, structurally different links can collapse to the same representation—especially when endpoints are automorphic. The authors introduce SP4LP, a GNN‑then‑SF framework that first runs a standard GNN to compute node embeddings and then forms a link representation by encoding the shortest path between the two endpoints as an ordered sequence with an LSTM/Transformer (or an injective sum) before combining it with the endpoints’ embeddings. On the theory side, the paper claims SP4LP is strictly more expressive than several popular baselines (Pure GNNs, NCN, BUDDY, NBFNet, Neo‑GNN). Empirically, the authors adopt the HeaRT evaluation protocol and report results on Planetoid and OGB datasets, which shows advantage performance of SP4LP.

### Strengths
Strengths:

[S1] This paper is an intuitive and natural extension from NCN --> NCNC, which consider not only 1-hop neighborhood overlap, but also high-order overlap.

[S2] Competitive results under a hard protocol, the HeaRT setting. SP4LP achieves top or near‑top performance on several datasets and is consistently strong across metrics.

[S3] The efficiency of the model is as good as other baselines, while excelling at the performance.

### Weaknesses
Weakness:

[W1] Several claims made by the paper is not rigourus. 

[W1.1] For example, the paper said "NCNC depends on the presence of common neighbors between link endpoints". However, NCNC can complete (this is what the last "C" comes from NCNC) the link if that node is connected to one of the target endpoint. 

[W1.2] The paper claims that "... prove that SP4LP is strictly more expressive than existing SF-and-GNN approaches". However, MPLP[1], a SF-and-GNN method, actually is **not less** expressive than SP4LP and NCN. Consider two rook's graph but with different degrees. These two graphs are strongly regular graphs. Any non-adjacent nodes on rook's graph will have two common neighbors. Then for any two non adjacent nodes, the final link representation of SP4LP or NCN on one rook's graph will be the same as the other, because they have no distinct node features or distinct shortest paths pattern. However, MPLP is able to distinguish because its norm rescaling can include node degree information. BTW, MPLP also have a good performance on HeaRT setting, which the authors may overlook.

[1] Pure Message Passing Can Estimate Common Neighbor for Link Prediction. Neurips 24'

### Questions
1. For the sequential modeling over the node embedding on the shortest path, is the positional embedding necessary?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SP4LP (Shortest Path for Link Prediction) — a framework combining Graph Neural Network (GNN) node embeddings with sequence models (e.g., LSTM or Transformer) applied to the shortest path between node pairs.
The goal is to enhance link prediction expressivity beyond message-passing GNNs, which struggle to distinguish automorphic nodes and lack link-level structural awareness.

### Strengths
1. *Clear Theoretical Framing*: The formal expressivity proofs (Def. 1.10–1.11, Thm. 2.5) are rigorous and well-structured.

2. *Strong Experimental Validation*: The results include multiple datasets and ablation studies, showing consistent performance gains.

3. *Transparency and Reproducibility*: The authors provide code links, full dataset stats, and hyperparameter configurations.

### Weaknesses
1. **Lack of Standard Paper Structure**: The paper’s organization makes it difficult to follow. There is no clearly labeled Introduction section, and the Related Work appears after the SP4LP section. For clarity and alignment with academic standards, the authors should include:
- Clearly name the text before "Preliminaries" as "Introduction".
- A Related Work section before SP4LP to properly contextualize the method.

2. **Incomplete Discussion of Hybrid Models**: The paper does not reference prior hybrid function-based approaches such as PROXI (PROXI: Challenging the GNNs for Link Prediction, TMLR 2025; https://openreview.net/pdf?id=u9EHndbiVw), where indices were incorporated in GNNs to improve their result.

3. **Flowchart**: It would be a good idea to include a flowchart of the model, and give a clear explanation for the model in SP4LP section.

### Questions
1. For large-scale graphs, how is the shortest-path extraction performed efficiently? Is it computed on-the-fly during training, precomputed, or approximated using heuristics?
2. Could SP4LP be applied to link prediction in heterogeneous graphs?
3. The datasets considered are all homophilic, how would the model perform on heterophilic datasets (Texas, Cornell, Wisconsin, Roman-Empire etc.)?
4. Please answer to the points made in "Weaknesses"

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SP4LP, a "GNN-then-SF" framework for link prediction. The method first computes node embeddings for the entire graph using a GNN, then extracts the shortest path for a given node pair, and finally processes the sequence of node embeddings along this path using a sequence model to generate a link representation. The paper claims this method is more expressive than pure GNNs and common-neighbor-based methods. However, this paper's core methodology is a direct rediscovery of the "Horizontal Geodesic Representation" from Geodesic GNN (Kong et al., NeurIPS 2022), which is not cited. Furthermore, its claims of scalability are questionable and based on a misleading complexity analysis that overlooks the cost of shortest path computation during inference.

### Strengths
- The paper performs a comprehensive evaluation and empirically validates the method against the challenging HeaRT benchmark over common OGB datasets.
- The authors provide a clear, data-driven justification for moving beyond common-neighbor-based methods by showing that a large fraction of positive links in many graphs do not share any common neighbors.

### Weaknesses
- The paper's major weakness is a lack of methodological originality. The SP4LP framework is identical to the "Horizontal Geodesic Representation" for link prediction, which was proposed and published prior in "Geodesic GNN (GDGNN)" (Kong et al., NeurIPS 2022). The SP4LP method, which runs a GNN once, finds the shortest path, and pools the sequence of node embeddings, is precisely the "GDGNN-Hor" method (see Section 3.2.1 of Kong et al.).
- The paper's argument for efficiency is not well-supported and relies on a flawed complexity analysis (Table 9).
   - The analysis places the entire shortest path cost ($T_{sp}$) into the "precomputation" cost $B$. This implies a one-time all-pairs-shortest-path computation (APSP), which has a massive $O(nm)$ cost. The authors didn't report the wall-clock time for this preprocessing.
  - For any practical online or inductive link prediction (on new nodes/links), this APSP precomputation is useless. The shortest path must be computed on-the-fly (e.g., via BFS), at a cost of $O(n+m)$ per link. This cost belongs in the per-link cost $C$, and the true per-link cost $O(n+m + kd + T_{seq})$ is therefore much worse than the methods it criticizes.
  - Given this high (and unstated) computational cost (roughly as SEAL), the empirical improvements over simpler baselines appear marginal on several key benchmarks (e.g., ogbl-ppa, ogbl-citation2). This raises questions about the method's practical utility.


Kong, L., Chen, Y., & Zhang, M. (2022). Geodesic graph neural network for efficient graph representation learning. Advances in neural information processing systems, 35, 5896-5909.

### Questions
1) The proposed SP4LP method is methodologically identical to the "Horizontal Geodesic Representation" from "Geodesic GNN (GDGNN)" (Kong et al., NeurIPS 2022). Could the authors please clarify the distinction between these two works?
2) The complexity analysis in Table 9 places the shortest path cost ($T_{sp}$) entirely in the precomputation cost $B$. Could the authors justify this for an online/inductive inference setting where the link is unknown in advance?
3) For predicting a single, unseen link at inference time, what is the real per-link complexity $C$?
4) Could the authors provide the actual wall-clock time for the all-pairs shortest path preprocessing step for all datasets?

### Soundness
1

### Presentation
2

### Contribution
1
