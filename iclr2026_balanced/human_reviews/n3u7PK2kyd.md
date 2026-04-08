## Human Reviewer 1

### Summary
This paper introduces a toolkit for comparing neural network representations using topological data analysis (TDA). 

The authors propose Symmetric Representation Topology Divergence (SRTD) and claim that the proposed SRTD is symmetric and more efficient compared to the original RTD framework.

Also, the authors introduce Normalized Topological Similarity (NTS) and claim that they are instances of a scale-invariant, normalized similarity measure. NTS compares the hierarchical clustering structure of two representations by correlating their single-linkage merge sequences. 

Through synthetic experiments (e.g. Gaussian cluster splits, UMAP embeddings) and some real-world evaluations (TinyCNN layers, large language model (LLM) representations), the authors show that their topological measures capture structural differences and similarities that geometry-based measures like CKA often overlook.

### Strengths
1) the authors proposed some modifications of RTD similarity measure and provided a theoretical analysis of how SRTD relates to the original RTD and Max-RTD divergences

2) also, the authors proposed a Normalized Topological Similarity which compares the hierarchical clustering “shape” of two representations by looking at the order in which points merge into clusters

3) the authors performed empirical evaluation of the proposed measures on a set of test problems, which is typical nowadays for such kind of papers. In this respect the empirical evaluation has sufficient experiments, each targeting a specific aspect or property.

### Weaknesses
1) In both conceptual scope and the breadth of empirical evaluation, the submission reads as a modest extension of RTD and RTD‑Lite rather than a substantive advance.

2) Evidence needed for the main claims. To convincingly support the paper’s assertions, include head‑to‑head comparisons with RTD/RTD‑Lite and at least one concrete example where those baselines miss a topological discrepancy between two point clouds that the proposed methods (SRTD/NTS) correctly identify.

On lines 342-373 and in several other places negative comments about RTD and RTDLite are not supported by any arguments, for some reason there is no comparison with RTD, RTDLite or RTDmax in fig 4 etc.

3) There is less interpretability compared to RTD due to the fact that SRTD does not compare the graph A itself with something, but rather some features of the graph are involved in comparison 

4) There is an RTD version of the divergence measure specially designed for functions which acronym is SFTD (see Scalar Function Topology Divergence, ECCV 2024). So the proposed acronym SRTD does not sufficiently differentiate the method from the submitted paper with the method from the ECCV paper.

5) Several statements in the abstract and body mischaracterize baselines RTD/RTD‑Lite. Symmetry: the original work evaluates the symmetrized measure—specifically, the average of RTD(A, B) and RTD(B, A)—for its principal experiments, so the effective metric is symmetric. Normalization: both papers explicitly instruct normalizing each point cloud by the 90th‑percentile of its distance distribution.

6) By design, NTS considers only the 0-dimensional homology (connected components merging) of the representation spaces, since it uses MSTs to capture the clustering hierarchy. This focus is well-justified (hierarchical clustering is arguably the most salient topological feature for many high-dimensional data distributions), but it does ignore higher-dimensional topological features like loops or cavities (1-D, 2-D homology, etc.). 

7) The paper does not provide detailed complexity analysis in the main text. It remains untested how the methods perform on truly large datasets (e.g., tens of thousands of points or more per representation). RTD-lite (Tulchinskii et al., 2025) was developed for large-scale use, but even it has practical limits. What is about the proposed measures?

8) Independent verification of results is not currently possible because the implementation has not been released and so there is a reproducibility gap.

### Questions
1) Are there any ideas how to deal with the assumption of a one-to-one correspondence between the two sets of representations being compared? The paper does not discuss this issue in depth, so it’s unclear how robust the approach is if the correspondence is noisy or only approximate.

2) Raw divergence scores can be hard to interpret across contexts, and can sometimes be dominated by a few “ultra-long” persistence barcodes. The paper proposes two measures - SRTD and NTS. Are there any recommendations when each should be used? It is not clear how to interpret a given SRTD value as “big or small” without a reference, or how to understand if a low NTS (e.g., 0.2) is significant or just noise – these interpretability issues are worth discussing.

3) For the LLM analysis, the authors recommend Z-score normalization of features before computing NTS to account for activation scale differences. This is a sensible step, but it introduces a preprocessing choice; one might wonder if NTS (being rank-based) is still needed after such normalization or how much it matters. 

4) NTS focuses on connected components (0-dimensional homology). Have you considered creating a normalized similarity measure for higher-dimensional topological features (e.g., loops)? For instance, could a similar “rank-order” approach be applied to 1-D persistence barcodes? 

5) Since NTS produces a correlation coefficient, it can, in principle, be negative (indicating very different clustering orders). In your experiments (especially on real data), did you ever observe significantly negative NTS values, and if so, how would you interpret them? For example, would NTS ≈ -0.5 indicate two representations have “inversely” structured cluster hierarchies? Clarifying this interpretation (even if it didn’t occur often in your cases) could be useful for users applying NTS to arbitrary datasets

6) Your largest experiments use around a few thousand points. How does the runtime scale with the number of data points n for SRTD-lite and NTS in practice?

7) You introduced two versions of NTS. Could you elaborate on when one should prefer NTS-E over NTS-M (or vice versa)?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
In this paper, the authors introduce Symmetric Representation Topology Divergence (SRTD) and SRTD-Lite as better and more efficient alternatives to previously existing RTD and RTD-Lite topological dissimilarity measures; they also porpose Normalized Topological Similarity as a robust similarity score. Authors prove their mathematical properties and present a diverse set of experiments to illustrate them practically.

### Strengths
- A solid and extensively developed mathematical background of the proposed methods.

### Weaknesses
- My main concern with this paper is its somewhat lack of novelty. On the level of ideas and the range of the performed evaluation experiments, this article feels incremental to the original publications of RTD and RTD-Lite.

- To properly support the main claims of the paper, a more detailed comparison of the proposed methods against RTD/RTD-Lite is needed: an example of the case when previous methods fail to capture the topological difference between two point clouds but the proposed methods (SRTD/NTS) don't would be much appreciated.

- There are some claims about RTD and RTD-Lite (in the abstract and further in the text) that are not exactly true:

1) Asymmetry. The original paper proposes to use the average of RTD(A,B) and RTD(B, A) and does so in the main experiments, thus making the metric symmetrc.

2) Normalization. In papers for both of those metrics, normalization of both point clouds (by 90%ーquantile of distances in each) is explicitly pointed out.

- Reproducibility of the results could not be assessed because code wasn't provided.

### Questions
- See Weaknesses.

- Could you provide for RTD-Lite heatmaps similar to those on Figure 6?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposes to enhance the existing RTD family of topology-based methods to compare representations by addressing the problems of symmetrization, theoretical relation and normalization. First, the authors propose Symmetric RTD (SRTD) and corresponding lightweight SRTD-lite which are based on the map from the graph with element-wise maximum of edge weights to the graph with element-wise minimum of edges and, hence, are symmetric by construction. The authors provide theoretical relations between RTD, Max-RTD and SRTD and relations between Max-RTD-lite, RTD-lite, SRTD-lite. Second, the authors propose Normalized Topological Similarity (NTS) which is normalized scale-invariant measure to compare cluster structures. Two variants of NTS are provided: NTS-M reflects correlation between the merging times of connected components, NTS-E measures correlation of the weights of graphs’ MST edges. The proposed methods are applied to compare representations of UMAP, convolutional networks, LLMs and to preserve the topological structure of the data during training the auto encoder.

### Strengths
- The proposed SRTD (SRTD-lite) is more computationally efficient than RTD (RTD-lite). Simultaneously, SRTD AutoEncoder (AE) (SRTD-lite AE) performs on par with the RTD AE (RTD-lite AE) but training is faster.
- The authors provide a relationship between different versions of RTD complementing the theoretical properties derived in previous works.

### Weaknesses
- In a more general setup, RTD(-lite) compares cluster structure of two weighted graphs with one-to-one correspondence between vertices. In certain cases, SRTD(-lite) may be less informative than RTD(-lite) by construction. For example, when the two input graphs have disjoint sets of edges, SRTD(-lite) will mostly reveal the topological properties of the graph with union of edges. RTD(-lite) will compare each of the input graphs with the graph of union of edges and reflect the distinguishing features.
- Lines 211-212: the motivation is not clear. Typically, the length of a bar in the barcode corresponds to importance of the feature. Longer bars correspond to more important features while smaller bars correspond to unimportant or noisy features. The common knowledge is to pay more attention to important features, otherwise, the measure can be misleading, for example, when optimized as additional loss term in a learning problem.
- Most experiments focus on comparison of the proposed topological methods with CKA, but drawbacks of CKA compared to RTD family were explored in the previous works. At the same time, a sufficient comparison with RTD(-lite) is missing. Some claims (lines 342-346) are not clear and lack supporting experimental results.

### Questions
- Figure 4, b: For the (Pool, Pool) entry, NTS-E is close to -0.4. Shouldn’t it be close to 1 as the maximal similarity is expected to be between the corresponding layers?
- Table 1, 2: While all the metrics can distinguish auto encoders trained with different types of additional loss term, NTS-E seems to be insensitive to subtle changes. How NTS-E should be interpreted in this case?
- Figure 2, caption: Probably, (a) and (b) should be replaced with (e) and (f) ?
- Figure 7: Shouldn’t the arrow between $R_\alpha(G^{max(w, \tilde{w})})$ and $G^w$ be in the opposite direction?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes a new topological framework for analyzing neural network representations that fixes key weaknesses of the existing RTD method, whose asymmetry and lack of normalization make results hard to interpret. It introduces the Symmetric Representation Topology Divergence (SRTD) and its efficient variant SRTD‑lite, which provide a consistent and scalable way to measure topological differences. To overcome scale dependence, the authors also develop Normalized Topological Similarity (NTS), a normalized, rank‑based similarity score that captures hierarchical clustering structures. Together, these tools offer a more interpretable and robust approach to comparing neural representations than prior geometric measures like CKA.

### Strengths
- Originality: The symmetric formulation of RTD and its lightweight version, SRTD‑lite, represent a novel theoretical expansion of topological divergence methods. Moreover, they introduce a new, scale‑invariant, rank‑based metric for comparing hierarchical clustering structures.
- Clarity: The mathematical problem is explained clearly and the formulation is accessible also for non-experts.
- Quality: The experiments explore a wide range of cases with increasing complexity.
- Significance: In the context of analysing topological differences in neural network representations this work seems to solve significant issues, generalizing the applicability of the similarity measure to broader contexts.

### Weaknesses
- The first sentence in the introduction seems way too generalized with respect to the context of this paper: there is a very vast literature in interpretability of neural network, many works on geometric analyses, and I would say that CKA methods are a small part of all these works, considering direct comparison of representations is not the only way of interpreting representations. I might have misunderstood what the authors want to convey here, but I would suggest rephrasing this part for a fairer framing of the context.
- Along these lines, even in the realm of using TDA for representation analysis, RTD is surely a key result, but the authors miss a long list of works and techniques. Again, I wonder whether the authors might try to specify they are talking about the specific set of studies that use representation comparison as a mean to analyse and interpret neural network representations.
- When motivating NTS, the authors argue that an undesirable effect of divergence-based analysis is the sensitivity to large scale differences that can mask finer structural details. This is a bit counter-intuitive to me, but I be mislead: if there exist large scale differences between two reps, these should be very important to track, since they might describe global characteristics of the representations. I do understand the benefit of hierarchical structures, but it would improve the manuscript if the authors clarified further this point.
- I wonder whether the benefit of having a hierarchical similarity score with respect to interpretability might be made more explicit: the clusters experiment shows a clear advantage and it is interpretable, but it is very limited in scope. On the other hand, when applied to more realistic settings (e.g. intra-model comparison, cfr figure 5) it is not clear to me where the hierarchical nature of the measure stands out. I think for instance similarity measures computed in other papers  using cosine similarity (Gromov et al. 2024) or inter-layer persistence (Gardinazzi et al. 2025) show similar patterns.

### Questions
- It would be good to use a uniform criterion in heatmaps: in some figures yellow is large values, in others it is low values. Might the authors correct this for more intuitive visualization?
- In line 429, the authors say they extract the last token rep from the 6th layer as it gives more discriminative results. Given such a specific choice, it might be good for presentation purposes to show a few details about this choice (in appendix). Wouldn’t it be possible to stack information from all layers?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper introduces a modification of RTD called "Max-RTD" and its scale-normalized variant (NTS).
Theoretical analysis is provided. Experiments show that Max-RTD and NTS outperform traditional CKA, however, its superiority over RTD and RTD-Lite is not sufficiently justified.

### Strengths
1. The paper introduces a modification of RTD, called SymRTD (SRTD). In opposite to RTD, it is symmetric by definition and doesn't required extra symmetrization.
2. Theoretical relationship between RTD, Max-RTD, and SRTD is established.
3. The language of the paper is fine, main ideas are easy to follow.

### Weaknesses
1. Some introduction for a reader not familiar with TDA might be necessary, like definitions of VR complex, persistent homology, etc.
2. Line 168-173: the proof of a long exact sequence is not provided.
3. In Section 5.1, the superiority of SRTD w.r.t. of RTD is not proved. Also, the illustration of RTD in the "UMAP Embeddings Experiment" is missing.
4.  In Section 4.3, the Algorithms 1, 2 are not clear. Please provide a formal definition of a MergeTime and how $V_{merge}$ lists are created.
5. The weakness of NTS is its non-differentiability.
6. In Theorem 3.3 it is not clear how integration is performed.
7. Section 5.3: the classification accuracy of NTS is lower than such of CKA. Heat maps of RTD and RTD-Lite are missing. 
As I understand, author claim that the monotony of NTS is its main advantage. But why? It is not clear a priori that the dissimilarity of representation in layers must change monotone. Please consider adding here more arguments, maybe visualization, etc.
8. In Section 5.4, a comparison with RTD and RTD-Lite is not provided.

### Questions
1. NTS try to address an issue with "ultra-long barcodes" (line 211). But the problem setting itself is questionable, because long barcodes correspond to prominent (persistent) topological features which must make significant contribution to a dissimilarity score.

Conclusion. I acknowledge the contribution of the work, establishing relationship between RTD, Max-RTD and Sym-RTD. But I consider the contribution is incremental w.r.t RTD and RTD-Lite.Moreover, the superiority of SRTD w.r.t. RTD is not sufficiently grounded.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
4