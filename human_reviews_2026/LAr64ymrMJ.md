# FSD-CAP: Fractional Subgraph Diffusion with Class-Aware Propagation for Graph Feature Imputation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 2, 6

## Abstract
Imputing missing node features in graphs is challenging, particularly under high missing rates. Existing methods based on latent representations or global diffusion often fail to produce reliable estimates, and may propagate errors across the graph. We propose FSD-CAP, a two-stage framework designed to improve imputation quality under extreme sparsity. In the first stage, a graph-distance-guided subgraph expansion localizes the diffusion process. A fractional diffusion operator adjusts propagation sharpness based on local structure. In the second stage, imputed features are refined using class-aware propagation, which incorporates pseudo-labels and neighborhood entropy to promote consistency. We evaluated FSD-CAP on multiple datasets. With 99.5% of features missing across five benchmark datasets, FSD-CAP achieves average accuracies of 80.06% (structural) and 81.01% (uniform) in node classification, close to the 81.31% achieved by a standard GCN with full features. For link prediction under the same setting, it reaches AUC scores of 91.65% (structural) and 92.41% (uniform), compared to 95.06% for the fully observed case. Furthermore, FSD-CAP demonstrates superior performance on both large-scale and heterophily datasets when compared to other models. Codes are available at https://github.com/ssjcode/FSD-CAP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper addresses the challenge of graph feature imputation under extreme missing rates—a limitation of existing methods that fail to produce reliable estimates or propagate errors in sparse graphs. The authors propose FSD-CAP, a two-stage diffusion-based framework designed to enhance imputation robustness and adaptability:

### Strengths
1. FSD-CAP combines three components (fractional diffusion, progressive subgraphs, class-aware refinement) in a cohesive framework, avoiding the "black-box" nature of many deep learning-based imputers.
2. The method’s two-stage pipeline is visualized and described in plain terms, making it accessible to researchers new to graph diffusion. Appendices provide implementation code details, hyperparameter ranges, and extra experiments (e.g., accuracy vs. distance from observed nodes).

### Weaknesses
1.  The paper only addresses missing node attributes, ignoring missing or noisy edges—a common real-world scenario (e.g., incomplete social network connections). Extending FSD-CAP to joint feature-structure imputation is mentioned as future work but is a key limitation for broader applicability.
2.  Experiments focus on standard benchmarks (citation networks, product co-purchase graphs) but lack a real-domain application (e.g., imputing missing gene features in biological networks, user attributes in recommendation systems). A case study would strengthen the paper’s practical impact.

### Questions
1. How would FSD-CAP need to be modified to handle both missing node features and missing edges? For example, could the subgraph expansion stage incorporate edge confidence scores to avoid propagating information through uncertain edges?
2. Most experiments use static graphs. Would FSD-CAP’s progressive subgraph expansion adapt to dynamic graphs (e.g., evolving social networks with new nodes/edges), or would frequent subgraph recomputation be computationally prohibitive?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FSD-CAP, a two-stage framework for imputing missing node features, especially in the cases with high sparsity of node metadata. The framework performs an initial imputation based on "fractional diffusion operator" to adjust propagation sharpness based on local structure, with "progressive subgraph diffusion" mechanism that expands layer-by-layer from observed nodes, localizing diffusion and limiting error propagation. Then, Class-Aware Propagation (CAP) refines the imputed features by applying a standard GCN to the FSD-imputed features to generate synthetic class labels. CAP then creates one vector for each class, and this class-specific vectors are formed based on the nodes inside each class but not those in the boundaries (though the "credibility" weight calculated by the faction of neighbors sharing the same label with each focal node in a class.) By down-weighting the features from the boundary nodes and imputing these class-specific vectors into the nodes in its class, this class-specific vectors serve as counteractive mechanism against over-smoothing. The effectiveness is demonstrated using the node classification and link prediction benchmarks, confirming that the proposed method is robust against the missing features of nodes.

### Strengths
The authors introduced a practical solution to balance imputation and over-smoothing though the fractional diffusion operator and CAP as two counteracting mechanisms. The authors also demonstrated the effectiveness in node classification and link prediction problems, which demonstrate the generalizability of the proposed solution for the missing feature problem.

### Weaknesses
W1 
The paper would be clearer if it explicitly stated the key assumptions underlying the proposed solution. In essence, the methods succeed only when three conditions hold: (1) the node features are informative for the task at hand; (2) the features are homophilic (adjacent nodes tend to share similar values); and (3) the graph exhibits community structure in which each community is dominated by nodes with the same feature label. It is important to note that node metadata do not always reflect the network topology, and not every metadata type is useful for a given application (see [1]). If condition (2) fails (i.e., neighboring nodes are likely to have different features), the diffusion process will incorrectly impute missing values. Moreover, the CAP framework relies on the premise that node features and community structure are strongly coupled. The fractional diffusion operator is built from the normalized adjacency matrix, and its mixing behavior is largely dictated by the community structure (as captured by the Fiedler vector). When node features are not correlated with the community structure, the diffusion does not remain localized; consequently, many nodes have similar feature values, and CAP loses its ability to distinguish clusters within the network.

I'm overall positive about the ideas from the authors but suggest them to make it very explicit about the key premises the method is based on, and clarify when the proposed solution works and do not work. 

[1] Peel, Leto, Daniel B. Larremore, and Aaron Clauset. "The ground truth about metadata and community detection in networks." Science advances 3.5 (2017): e1602548.

W2
I appreciate the authors to include GCN as a baseline, with missing features imputed with zeros. The paper could be benefit from providing other common heuristics and the comparisons against more recent convolution based graph neural networks. For example, another common imputation method is to fill the missing feature with a global mean. Alongside is the k-nearest neighbor methods. These are pretty common imputation methods that do not require graph structure (and thus are less prone to the violation of the assumptions discussed above). Also GCN (Kipf & Welling, 2017) is already primitive, and the GNN field has advanced significantly since 2017. I suggest the authors to include advanced and commonly used models (not necessarily the state-of-the-art models, such as GAT) to demonstrate the broader applicability.

### Questions
Q1: What are the underlying premise for the proposed solution?

Q2: What are the performance gain of the proposed imputation methods against traditional but well-accepted generic imputation methods (i.e., global mean and k-nn based method)?

### Soundness
2

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
This paper proposes FSD-CAP, a two-stage framework for imputing missing node features in graphs under high missing rates. The first stage uses fractional subgraph diffusion (FSD). The second stage applies class-aware propagation (CAP) using pseudo-labels and neighborhood entropy for feature refinement.

### Strengths
The fractional diffusion operator provides principled control over propagation sharpness, interpolating between uniform averaging and dominant-neighbor selection.

Progressive subgraph expansion reduces error accumulation compared to global diffusion.

Achieves competitive results across multiple datasets and missing patterns.

### Weaknesses
Table 11 shows counterintuitive performance gains with increasing missing rates: CiteSeer achieves 71.94% accuracy at 99.5% structural missing versus 70.00% with full features. Similar anomalies appear across multiple datasets and missing types, violating basic expectations that more missing information should degrade performance.

Critical absence of recent self-supervised graph learning methods for feature reconstruction (e.g., GRACE, GraphCL, MVGRL).

Missing comparisons with masked graph modeling approaches that follow BERT-style pre-training paradigms for graph feature completion.

No evaluation against graph contrastive learning methods (SimGRACE, GraphMAE, MaskGAE) that have shown strong performance on incomplete graphs.

Lacks comparison with recent graph pre-training methods and foundation models that can handle missing features through representation learning.

Missing recent diffusion-based graph generation methods that could serve as feature completion baselines.

Progressive subgraph expansion assumes feature similarity decreases with graph distance, but this may not hold for heterophilous graphs or when class boundaries cross multiple hops.

Convergence analysis (Theorems 2-3) relies on strong connectivity assumptions that may be violated in practice when 99.5% of features are missing.

Code availability promises may be insufficient given the complexity of the two-stage framework.

### Questions
The authors should address the concerns raised in the weakness section above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FSD-CAP (Fractional Subgraph Diffusion with Class-Aware Propagation), a diffusion-based framework for imputing missing node features in graphs with partially observed features. The first stage, fractional subgraph diffusion, introduces a tunable fractional operator that adjusts propagation sharpness according to local graph structure, and progressively expands diffusion from observed to unobserved nodes to limit error accumulation. The second stage, class-aware propagation, refines the imputed features using pseudo-labels and neighborhood-entropy weighting to promote intra-class consistency and inter-class separation. Experiments demonstrate the effectiveness of FSD-CAP on semi-supervised node classification and link prediction at a high missing rate.

### Strengths
1. The authors present a new diffusion mechanism for imputation on graph-structured data.
2. The paper is clearly written and organized.

### Weaknesses
1. State-of-the-art methods [1, 2] are missing from the comparison. It appears that the state-of-the-art methods [1,2] achieve better performance than the proposed method, raising questions about the claimed performance advantages. Futhermore, [2] adopts a similar idea of leveraging label information during propagation.
2. Since the proposed method leverages label information, its performance is likely to depend on label availability; however, no experiments are provided to analyze sensitivity to different label rates.
3. In link prediction, it is generally assumed that only graph connectivity is available, but the paper additionally utilizes node labels.
4. The motivation or empirical observation behind the design of the Fractional Diffusion Operator is insufficiently explained.
5. It would be preferable to represent matrices and vectors in italicized bold.

[1] Um, Daeho, et al. "Propagate and Inject: Revisiting Propagation-Based Feature Imputation for Graphs with Partially Observed Features." ICML 2025.\
[2] Yun, Sukwon, et al. "Oldie but Goodie: Re-illuminating Label Propagation on Graphs with Partially Observed Features." KDD 2025.

### Questions
Q1. Could the authors provide performance comparisons with state-of-the-art methods [1, 2] for both node classification and link prediction tasks?

Q2. Since the proposed method leverages label information, could the authors show how performance changes when the label rate is lower than the current experimental setting?

Q3. The Fractional Diffusion Operator appears to down-weight the participation of nodes with higher degrees during diffusion. Could the authors clarify the motivation or empirical rationale (not resulting performance) behind this design choice?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present FSD-CAP, a novel method for missing feature imputation within graph representation learning by applying a 1) Factional diffusion operator, 2) Progressive subgraph diffusion, and 3) Class-level feature refinement. The authors introduce the mechanisms through a series of propositions and theorems that details the nuance between needing to extrapolate from coarser to finer features within subgraph diffusion. Notably, FSD-CAP outperforms the current SOTA method PCF1 by a fairly-significant margin and shows improvement over standard baselines on the CiteSeer dataset.

### Strengths
1) The authors provide concise yet clear details about the uniqueness of the operators and diffusion refinement stages for FSD-CAP. Further explanations detail clarifying insights about the practical considerations from moving between localized and distant updates.
2) The writing of the paper flows well, transitions between sections make sense and are not abrupt. This eases understanding of more technical concepts.
3) FSD-CAP significantly outperforms current SOTA methods like PCF1, indicating a potentially broad-level of promise for subgraph diffusion models in other applications of graph representation learning.

### Weaknesses
1) Although this is common within existing graph-diffusion feature imputation literature, FSD-CAP only tests on Cora, CiteSeer, PubMed, and Amazon-Photo, Amazon-Computers. Further tests on smaller synthetic datasets with defined local and global structures could indicate the power of Theorem 2 in potentially more-difficult scenarios
2) There are no algorithms within the paper to describe the process of FSD-CAP. Although this is partially-abated by the inclusion of code, the pseudo-code seems important for clarity when detailing how the equations within the paper function in-practice.

### Questions
* In regards to Remark 1, is a super-diffusion regime something that other diffusion models remain in since they do not have a class-refinement mechanism?
* Lines 184-187: Given that much of this is step-by-step instructions of FSD-CAP function, this seems better formatted in an algorithm? 
* Lines 214-215: equation 3, equation 4 ==> uppercase?
* Theorem 2: The idea that you need Equation 3 for updates and Equation 4 for maintaining stability makes sense. However, I am confused by the notation in Theorem 2. Does $x^{(x)}(t)$ apply both Equation 3 and Equation 4 per-layer in sequence (Equation 3 - top to Equation 4 - bottom), as expected with piece-wise functions (indicated by the brackets)? I can certainly see that it would be a 'layer-wise' function, but it still seems confusing that $x^{(x)}(t)$ is the same regardless of the right-hand side of the function. This could be a mistake on my part as a reviewer, but I want to clarify with the authors, is this intended? If so, wouldn't an algorithmic description provide more clarity on the order of operations?

### Soundness
3

### Presentation
3

### Contribution
3
