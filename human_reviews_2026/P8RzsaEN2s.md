# Local-Curvature-Aware Knowledge Graph Embedding via Extended Ricci Flow

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Knowledge graph embedding (KGE) relies on the geometry of the embedding space to encode semantic and structural relations. Existing methods place all entities on one homogeneous manifold—Euclidean, spherical, hyperbolic, or their product/multi‑curvature variants, to model linear, symmetric, or hierarchical patterns. Yet a predefined, homogeneous manifold cannot accommodate the sharply varying curvature that real‑world graphs exhibit across local regions. Since this geometry is imposed a priori, any mismatch with the knowledge graph’s local curvatures will distort distances between entities and hurt the expressiveness of the resulting KGE. To rectify this, we propose RicciKGE to have the KGE loss gradient coupled with local curvatures in an extended Ricci flow such that entity embeddings co-evolve dynamically with the underlying manifold geometry towards mutual adaptation. Theoretically, when the coupling coefficient is bounded and properly selected, we rigorously prove that i) all the edge-wise curvatures decay exponentially, meaning that the manifold is driven toward the Euclidean flatness; and ii) the KGE distances strictly converge to a global optimum, which indicates that geometric flattening and embedding optimization are promoting each other. Experimental improvements on link prediction and node classification benchmarks demonstrate RicciKGE's effectiveness in adapting to heterogeneous knowledge graph structures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets the geometric mismatch that arises when knowledge graph embeddings are forced into a predefined, homogeneous manifold, which distorts distances under locally heterogeneous curvature. It proposes RicciKGE, coupling the KGE loss gradient with local discrete Ricci curvature via an extended Ricci flow so that the latent geometry and entity embeddings co-evolve. The method offers theoretical guarantees of curvature flattening and linear distance convergence and reports consistent, if sometimes modest, gains on standard link prediction and node classification benchmarks.

### Strengths
1. The motivation is precise and grounded in limitations of homogeneous manifolds for KGE: the paper clearly articulates how a static geometric prior misaligns with locally varying curvature in real KGs. This framing connects well to manifold-based KGE literature and sets a concrete failure mode that the method aims to fix. 

2. Theoretical analysis is not superficial: Theorem 1 proves exponential decay of edge-wise Ricci curvature under bounded coupling, and the corollary establishes linear convergence of distances under strong convexity. The interplay between curvature decay and optimization is discussed rather than just stated, which raises confidence in the mechanism’s stability.

3. The algorithmic presentation is serviceable: Algorithm 1 aligns the distance-flow step with embedding updates, and the complexity section isolates the main overhead (curvature estimation via Sinkhorn) with a parallelization argument. While brief, this helps practitioners estimate costs of adoption.

### Weaknesses
1. The contribution over prior curvature-flow work feels incremental and needs sharper differentiation. The paper extends discrete Ricci flow with a gradient coupling term, but related graph Ricci-flow models and geometric regularizers already exist; the delta versus prior discrete/extended flows (e.g., geometry-only flows or Ricci-guided graph methods) remain under-quantified. A stronger ablation isolating “pure Ricci flow”, “pure gradient”, and “coupled” variants across datasets is necessary to establish novelty in effect, not just form. 
2. The theoretical guarantees rely on strong and somewhat idealized assumptions that may not reflect KGE training practice. The curvature result assumes volume/diameter bounds, Sobolev inequalities, and a spectral gap on a closed manifold; the distance convergence assumes µ-strong convexity in distance, while typical KGE losses combine non-convex components, negative sampling, and relation-specific transforms. The paper does not empirically validate robustness when these assumptions fail, leaving a theory–practice gap. 
3. There is a conceptual tension between the motivation (preserving heterogeneous local curvature) and the main theorem (all edge-wise curvature decays to zero). The text argues that curvature “imprints” are absorbed into embeddings during transients, but the paper provides limited qualitative or quantitative evidence of what structural signals survive once the manifold is flat. Visualization or probing tasks before/after flattening would help reconcile this tension. 
4. The empirical gains, while consistent, are often small and sometimes not state-of-the-art, and the statistical significance is not established. Table 1 shows multiple deltas around +0.1 to +0.4 MRR on FB15K-237 and cases that do not surpass the strongest baseline; no confidence intervals or paired tests are reported. Given the added O(|E|(k^2+kd)) overhead per epoch from curvature estimation, time–accuracy tradeoffs (including wall-clock, GPU memory, and scaling on larger KGs) should be reported to justify practicality. 
5. Reproducibility and implementation specifics are thin at submission time. Code is promised only upon acceptance, and several details are under-specified: the exact Sinkhorn parameters for W1, normalization schemes for edge weights, β search ranges and schedules, negative sampling strategies, and seed/variance reporting across runs. The β sensitivity plot is helpful, but does not replace a thorough hyperparameter protocol and release of full configs for each backbone

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RicciKGE, a curvature-adaptive knowledge graph embedding method that couples the KGE loss gradient with local edge-wise curvatures via an extended Ricci flow, allowing embeddings and manifold geometry to co-evolve. It targets the mismatch introduced by homogeneous manifolds (Euclidean, spherical, hyperbolic) when real-world graphs have heterogeneous local curvature, which can distort distances and limit expressiveness. The authors provide theory showing that, with a properly bounded coupling coefficient, edge-wise curvatures decay exponentially toward Euclidean flatness and KGE distances strictly converge to a global optimum, indicating mutual reinforcement between geometric flattening and embedding optimization. Empirically, RicciKGE improves link prediction and node classification performance on standard benchmarks, supporting the effectiveness of curvature adaptation for heterogeneous KG structures.

### Strengths
The paper is overall easy to follow.

The proposed method demonstrates a superior performance on large-scale graph dataset as shown in table 2 in the experiments section.

### Weaknesses
I mainly have concerns about the experiments.

The experiments use different embedding dimensions across datasets without explaining the rationale, and statistical significance measures are not reported in Table 1. This makes it difficult to assess the robustness and fairness of the comparisons.

The baseline results reported in Table 1 differ from those in the original paper (e.g., GoldE). It is unclear whether a re-implementation was used, and if so, the experimental settings, hyperparameters, and code differences should be documented to enable verification.

Insufficient baseline selection and discussion. The choice of baselines in Table 1 is not well justified. Although the paper shows performance improvements when integrating RicciKGE into several existing KGE models, the baselines appear limited to relatively basic KGE methods (except the recent GoldE). A more comprehensive comparison—including stronger, contemporary baselines and a discussion of why each was selected—would better demonstrate the generality and competitiveness of the approach.

### Questions
it is not clear to me why line 258-260 

"To validate itseffectiveness, we must demonstrate that under this iterative mapping, the Ricci curvature on every edge converges pointwise to zero, i.e., the manifold becomes asymptotically flat, thereby ensuring that all intrinsic curvature heterogeneity is faithfully captured in the learned entity embeddings. "

can you explain more on this

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RicciKGE, a knowledge graph embedding method that leverages a Ricci flow on the input graph to gradually flatten a low-dimensional representation of the entities in that graph, leading to the usual flat Euclidean manifold, but benefitted from awareness of the local curvature at each neighbourhood in the graph and task-related gradient information. This approach is evaluated both in the link prediction and node classification tasks, consistently performing on par or beating the baseline methods.

### Strengths
The presentation is very good, with appropriate formulation, helpful illustrations, and an algorithm listing. Furthermore, the underlying theory is (mostly, see below) clearly presented.

As I understand the paper, the authors extend the embedding evolution from Ricci flow (as in GNRF) to use also task-specific gradient information. I appreciate that there are analyses on the convergence, as this gives more confidence on the properties of the proposed method.

Experimentally, the authors explore not only KGE-specific tasks, but also node classification, along with additional insights into how the total loss and curvature variance evolve over time.

### Weaknesses
Given it is also predominantly based on Ricci flow, I believe the paper should address more explicitly the differences between its contributions and GNRF.

Experimentally, while the authors thoroughly evaluate multiple scenarios with multiple methods (well done!), I had the impression that little space was left to properly analyse the results. While the results are convincing, I expected more analysis here (see Question 4 for details).

### Questions
1. Can you define "neighbourhood measures" in the main paper before or around line 143? Since $\kappa(i, j)$ is so central to the Ricci curvature used to define the method, I believe it is important that most concepts it depends on are contained in the main paper.
2. Can you define $\beta$ and $\text{Ric}_{ij}$ in Eq. (2)? For the latter, it was also no longer used in the remainder of the paper, and its definition is left vague. Is it supposed to be some form of $\kappa(i, j)$? Is that important? Nevertheless, you could also consider rewriting Eq. (2) in a way that reflects that.
3. Algorithm 1 suggests that embeddings are updated in sequence. Is that the case? If yes, since certain entities can appear in multiple triplets, this might lead to multiple updates happening, which seems problematic. If not, can you please make this clearer in the algorithm?
4. Previous empirical evidence, such as AttH (Chami et al., 2020), suggests that a good match between the geometry of embeddings and the graph structure would enable lower-dimensional embeddings to outperform methods that ignore that (e.g., TransE, DistMult, etc.). In this context, looking at Table 1 (emb. dim. 32) would suggest that the performance gap should be larger for those methods versus AttH/GoldE, while that gap is indeed small in higher embedding dimensions. Why is that not consistently the case for lower dimenions in your results? Is there a more elaborate phenomenom going on?

**Other comments:**

- l. 483: Our theoretically => theory?

**References**

Chami, I., Wolf, A., Juan, D. C., Sala, F., Ravi, S., & Ré, C. (2020). Low-dimensional hyperbolic knowledge graph embeddings. arXiv preprint arXiv:2005.00545.

### Soundness
4

### Presentation
4

### Contribution
3
