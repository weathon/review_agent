## Human Reviewer 1

### Summary
This paper considers the connection between the oversquashing problem and negatively curved edges (quantified using a variety of different notions of discrete curvature). They show, contrary to popular belief, that negative curvature is merely a sufficient, rather than a necessary, condition for oversquashing to occur and show empirically that many oversquashed edges (quantified via jacobian norms) are "missed" by checking the curvature. To remedy this, the authors introduce a novel version of discrete curvature, as well as a fast approximation algorithm.

### Strengths
Oversquashing is a well-known, highly studied problem in the training of GNNs. This paper lends a new perspective showing that the prevailing wisdom, oversquashing <==> negative curvature, is inadequate. They also show experimentally which of the common GNN architecture are prone to oversquashing.

Empiricial results are supported by theoretical analysis and novel definition which provide a new framework for thinking about oversquashing

### Weaknesses
Assumption 1 comes out of no where and is hard to understand. More discussion and motivation should be given.

The definition of $\mu_u^\alpha$, is unclear

### Questions
Are there any viable ways to understand oversquashing which are not curvature derived?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
In this paper, the authors question the connection of edge curvature to over-squashing. These edges are identified by a discrete notion of curvature (which is specifically adapted for graphs) that assigns a curvature value to each edge. They specifically claim that high negative curvature is a sufficient but not a necessary condition for over-squashing. To show that, they create counterexamples where some of the edges are squashed while the curvature remains positive. Since the Ollivier–Ricci curvature, one of the most commonly used discrete curvature measure, fails to detect a considerable percentage of over-squashed edges, the authors propose the Weighted Augmented Forman-3 Curvature (WAF3) to improve the detection of over-squashed edges. Finally, a new approximated WAF is presented that is able to handle very large graphs.

### Strengths
- studies the over-squashing phenomenon in GNNs
- shows that over-squashed edges may not always be detected by curvature
- newly introduced metric for measuring over-squashed edges ratio by curvature-based criteria

### Weaknesses
- Counterexamples may not be representative of real-world graphs (e.g., citation, molecule, social). The authors should provide synthetic constructions with empirical evidence that similar topology/feature interactions occur in practical datasets.
- Weighted Jaccard and MinHash are algorithmic conveniences; the transformation may sacrifice geometric interpretability
- Proposed WAF3 correlates with or mitigates real over-squashing beyond proxy metrics

### Questions
- How robust is MOSR under different operational definitions of over-squashing?
- Do analogous failure cases appear in real-world graphs (social, citation, molecular)?
- Does WAF3-based rewiring or edge weighting actually alleviate over-squashing and improve downstream accuracy?

Typos
- line 155: Let s the soruce node, -> source

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors address the problem of over-squashing in GNNs. In this context, they investigate the following point: high negative curvature is a sufficient but not a necessary condition for over-squashing. In addition, they develop an approximation algorithm for the method Weighted Augmented Forman-3 Curvature.

### Strengths
- The paper is well organized and written
- The authors address a key issue about GNNs: over-squashing.
- The proposed method Weighted Augmented Forman-3 Curvature is detailed and reproducible.
- Experiments are well conducted.

### Weaknesses
The authors should indicate more information between over-squashing and over-smoothing.
Missing references:
A. Arnaiz-Rodríguez, F. Errica, “Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning”, Preprint, May 2025.
Y. Liu, et al., "CurvDrop: A Ricci curvature based approach to prevent graph neural networks from over-smoothing and over-squashing",  ACM Web Conference 2023, pages 221-230, 2023.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper argues that having a negative discrete curvature (e.g., Ollivier–Ricci, Forman variants) is not a necessary condition for over-squashing in GNNs. It provides a counterexample family Gnm, showing edges can be heavily over-squashed while having positive curvature; introduces MOSR (Missed Over-Squashing Ratio) to quantify how many over-squashed edges are missed by curvature; proposes a new curvature WAF3 (Weighted Augmented Forman-3) with a degree-weighting; and develops a MinHash-based approximation to scale WAF3. Experiments report MOSR across 21 datasets and three architectures (GCN/GAT/GraphSAGE), plus efficiency and rank-correlation results for the approximation.

### Strengths
- This paper shows that negative curvature does not imply necessity, with a concrete counterexample and theorem.
- MOSR is a reasonable way to quantify “missed” oversquashed edges.
- WAF3 is intuitive, keeps low complexity, and is shown to remove the counterexample under mild conditions.
- Equivalent form via weighted Jaccard + MinHash is clever and gives large speedups with good rank fidelity.

### Weaknesses
- Eqn (1) is not the MPNN model; that’s the GCN model. MPNN is a generalization of GCN. This is problematic since most of the results in over-squashing in the previous literature are for MPNNs, while in this paper, they’re only true for GCNs. This reduces the scope and impact of the proposed work and diminishes the results in Table 4, since the theoretical understanding is only available for GCN.
- Similarly, multiple formal statements (Lemma 2, Theorem 5) say “Assume an L-layer MPNN as in (1).” But Eq. (1) is the symmetric-normalized GCN layer with ReLU; it is not a general MPNN. The proofs that rely on the exact matrix M therefore apply to GCN-style propagation; generalizations would require additional conditions on the aggregation kernels.
- The last part of Section 4 lacks clear intuition on how over-squashing relates to bottlenecks and how this differs from “bridges”. Prior theory (e.g., Topping et al.) does not imply that removing bottlenecks eliminates over-squashing, yet the paper does not provide a concrete narrative or example to reconcile this. In particular, the manuscript does not illustrate what an intra-cluster over-squashed edge looks like (low betweenness, still over-squashed) or explain the mechanisms that make it over-squashed despite no obvious bottleneck. This gap makes the conceptual contribution harder to follow and weakens the takeaways of Section 4.
- The choice of the Forman-3 curvature for the extension seems arbitrary. Indeed, the balance Forman w/o 4-cycle and the Jost-Liu Forman have the same complexity as the Augmented Forman-3 according to Table 1. However, those are not compared in Table 2, and we don’t know if these other two computationally efficient metrics are better or worse than the Forman-3 curvature.
- I understood the reasoning behind Figure 4, but I feel it’s also important to compare the efficient version of the proposed curvature in terms of MOSR, as in Table 4. This new version might be more efficient, but the main point of this paper is to avoid missing the over-squashed edges where the curvature is positive, and if this is not the case for the efficient formulation, why should one go with this alternative? If the important point is the ordering (I agree with the authors on this), why should one choose this approach over something that achieves the correct rank, while being perhaps more efficient?


**General comment**: This is a timely paper, but two pieces are missing to make the contribution fully convincing. (1) The manuscript does not clearly connect the proposed most efficient curvature variant to the central claim of the paper (reducing missed over-squashed edges). Please make the causal chain explicit: why this efficient metric preserves the ordering or detection properties that matter for oversquashing, and how that supports the paper’s main thesis. (2) The paper lacks a large-scale, real-world evaluation demonstrating impact. A straightforward way to address this is to plug the efficient curvature into an existing rewiring pipeline and report results on a sparse OGB node property benchmark—e.g., ogbn-products or ogbn-papers100M—including both accuracy and compute (runtime/memory). This would substantiate scalability claims and show that the efficiency gains translate into practical benefits without undermining effectiveness.


**Minor comments:**

- Line 58: “This theorem…”, which theorem? At this point in the paper, it’s not clear this is supported by a theorem.
- Please define acronyms the first time they’re used, e.g., MOSR.
- It seems like Table 3 has not been referenced in the paper.
- Figure 2 is violating the margins.
- Typo in Theorem 5? There dose not.
- Could you include the statistics of the datasets? I think this is important to put things in context.

### Questions
- What is \rho in Lemma 2? This also appears in Appendix A.1, but I’m not sure what this is.
- Assumption 1 is not clear. What does the paper mean by all paths in the computation graph? What’s the computation graph in this case?
- The notation in Definition 3 is not clear. N_1 and N_2 are the 1-hop and 2-hop neighborhoods of whom?
- s_q is the proportion of over-squashing edges that are not identified by curvature, but I don’t see this in the equation. Do the authors mean MOSR_q?
- What’s the number of layers L for the experiments in Table 2?
- Could the authors provide the same results as in Table 2 for the balance Forman w/o 4-cycle and the Jost-Liu Forman? These two curvature metrics are also computationally efficient.
- In Appendix A.1, Lemma 5 is used to get the bound, but I didn’t find this Lemma. Is it Lemma 7? Even if this is the case, I’m not quite sure how to get the bound in the first Equation on page 14 (please label the equations). Why is greater than or equal to? I’m confused at that point.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 5

### Summary
The paper challenges the common assumption that discrete curvature is a reliable proxy for detecting over-squashing in GNNs, showing via a family of counterexample graphs and formal results that many severely squashed edges can still have positive curvature—so high negative curvature is sufficient but not necessary for over-squashing. To quantify this gap, the authors introduce MOSR, a metric that counts the fraction of truly over-squashed edges missed by a curvature rule; across models and 21 datasets, Ollivier–Ricci misses roughly 30–40% of such edges, and a betweenness analysis indicates that curvature mostly finds “bridge” edges while often ignoring in-cluster bottlenecks. They then propose Weighted Augmented Forman-3 (WAF3), a degree-weighted refinement that theoretically avoids the counterexamples and empirically reduces MOSR relative to prior curvatures while retaining low complexity. Finally, they derive an equivalent weighted-Jaccard form and a MinHash-based approximation that lowers runtime to linear in the number of edges, enabling curvature computation on graphs with about 5 million edges

### Strengths
The paper is original in challenging a widely assumed link between discrete curvature and over-squashing and in formalizing how and why common curvatures can fail, complemented by a clear diagnostic (MOSR) that quantifies misses rather than relying on anecdotal examples. It shows solid technical quality through explicit counterexamples, theoretical guarantees for the proposed WAF3 variant, and careful complexity considerations, including an equivalent weighted-Jaccard view and a MinHash-based approximation that makes large-scale computation practical. The presentation is clear: definitions, constructions, and algorithms are spelled out with enough detail to reproduce both the counterexamples and the fast estimator. Significance is high for both researchers and practitioners who use curvature to guide model design or interventions for over-squashing, since the work both tempers over-reliance on existing curvatures and supplies a drop-in alternative that is faster and empirically more aligned with actual over-squashing.

### Weaknesses
Despite strong theoretical framing, the empirical story rests heavily on MOSR—a model- and label-dependent proxy for “true” over-squashing—so it remains unclear how much the proposed curvature actually improves downstream GNN behavior (e.g., training stability, accuracy, or robustness) beyond ranking edges; adding intervention studies (rewiring, weighting, or attention bias guided by WAF3 vs prior curvatures) and showing consistent end-task gains would strengthen the claim. The novelty of WAF3 also needs sharper positioning: prior work on curvature-based structural encodings (e.g., “Effective structural encodings via local curvature profiles”) should be cited in the introduction and connections to other works that use (weighted) augmented Forman–Ricci variants (e.g., “Augmentations of Forman’s Ricci curvature and their applications in community detection”) should be cited and discussed.

### Questions
Could you provide intervention-based validation showing that using WAF3 to guide rewiring, edge weighting, or attention bias leads to consistent gains in downstream GNN performance versus Ollivier–Ricci and prior Forman variants, with compute-matched baselines, multiple seeds, and analyses of where improvements concentrate (e.g., non-bridge bottlenecks)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4