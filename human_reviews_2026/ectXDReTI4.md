# GIST: Gauge-Invariant Spectral Transformers for Scalable Graph Neural Operators

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Adapting transformers to meshes and graph-structured data presents significant computational challenges, particularly when leveraging spectral methods that require eigendecomposition of the graph Laplacian, a process incurring cubic complexity for dense matrices or quadratic complexity for sparse graphs, a cost further compounded by the quadratic complexity of standard self-attention mechanism.
Conventional approximate spectral methods compromise the gauge symmetry inherent in spectral basis selection, risking the introduction of spurious features tied to the gauge choice that could undermine generalization.
In this paper we propose a transformer architecture that is able to preserve gauge symmetry through distance-based operations on approximate randomly projected spectral embeddings, achieving linear complexity while maintaining gauge invariance.
By integrating this design within a linear transformer framework, we obtain end-to-end memory and computational costs that scale linearly with number of nodes in the graph.
Unlike approximate methods that sacrifice gauge symmetry for computational efficiency, our approach maintains both scalability and the principled inductive biases necessary for effective generalization to unseen graph structures in inductive graph learning tasks.
We demonstrate our method's flexibility by benchmarking on standard transductive and inductive node classification tasks, achieving results matching the state-of-the-art on multiple datasets. 
Furthermore, we demonstrate scalability by deploying our architecture as a discretization-free Neural Operator for large-scale computational fluid dynamics mesh regression, surpassing state-of-the-art performance on aerodynamic coefficient prediction reformulated as a graph node regression task.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Thanks for the submission. The paper suggests an efficient transformer architecture for graph-structured data. The core idea is to use JLT’d spectral features (Laplacian eigenvectors, whose dimensionality is reduced with a random projection) in place of the usual queries and keys, which gives a form of spectral self-attention which is approximately gauge-invariant. These maps can be computed more efficiently than the full diagonalisation using FastRP, a random projection-based truncation algorithm developed separately by Chen et al. (2019)

### Strengths
1. Efficient transformers for graph structured data is an important unsolved research problem, and I can see that pure message passing will underperform in the graph neural operator setting because of its finite receptive field.
2. The paper is broadly well written.

### Weaknesses
1. I appreciate the need for memetic titles, but my understanding is that the method isn’t actually gauge invariant for any draw of the random projection matrix R. The authors do acknowledge that the invariance is approximate (it depends on the expectation of $RR^T$ being the identity, with $R$ a random low rank matrix), but in places I think the phrasing is a bit misleading. E.g. see line 298 – isn’t this equation actually wrong without taking the expectation? I appreciate that the JLT preserves the norms and distances between a set of vectors with high probability; it would be even better (but probably difficult) to formulate the approximate invariance property mathematically rigorously.
2. A note: if you compute regular linear attention in one branch and gauge-invariant attention in another branch and add the results, isn’t this the same as just concatenating the regular queries/keys with your spectral features? (At least up to the different value projection matrix). I’m wondering whether the division into branches is strictly necessary, or whether the paper is really about a new efficient absolute position embedding with some nice approximate invariance properties. 
3. Missing experiments, that are (imo) crucial:  
a) No ablation of removing the spectral attention branch, including just regular linear attention + message passing. This is surely the most crucial acid test in order to assess gains from your algorithm!    
b) No ablation over different embedding dimensions $r$ – unless you can fix r independently of N and get consistent performance, it’s a bit of a stretch to claim the algorithm is linear in N.  
c) No demos on any toy tasks that strongly depend on graph structure. The authors do include a few benchmarks (I especially like the neural operator results), but I’m not sure whether Cora/Citeseer/Pubmed really distill out whether your addition helps the transformer better capture graph structure. Something explicitly topological like shortest path distance prediction might be more natural, and a good setting to ablate the spectral branch.  
d) No time complexity scaling wrt number of graph nodes – wall-clock time or FLOPs.

### Questions
1. Please could you clarify if/how you choose the ‘large enough iteration step $K$’? Is this a hyperparameter, or is it chosen in some principled (graph-dependent) way to ensure the estimator is accurate? Doesn’t this implicitly set some finite receptive field? Won’t finite $K$, independent of $N$, cause regressions on very big graphs/high mesh resolutions?

### Soundness
3

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
5

### Summary
This paper examines the positional encoding effectiveness of Laplacian matrix eigenvectors in graph transformers, with a particular focus on node classification tasks. It introduces an Energy Spectral Density metric derived from class labels and uses this metric to identify the top-_k_ eigenvectors for encoding. The proposed approach is integrated into several existing graph transformer architectures, leading to consistent improvements in node classification performance across multiple datasets.

### Strengths
1. The proposed ESD metric and corresponding BTS method are simple, intuitive, and easily adaptable to a wide range of graph transformer models.
2. The paper offers a theoretical analysis of the rationale behind BTS, elucidating its effectiveness in the context of node classification tasks.
3. The experimental evaluation is thorough, including extensive ablation studies that validate the efficacy of the proposed BTS method.

### Weaknesses
1. The contributions of this paper are somewhat limited. The main contribution  lies in the proposed Gauge-Invariant/Equivariant Spectral Self-Attention mechanisms, while the linear-time spectral embedding implementation is largely based on prior work, i.e., FastRP.
2. The analysis of GIST is insufficient, both theoretically and experimentally. In particular, the paper does not examine how the parameters $r$ and $k$  in Algorithm 1 affect performance and complexity. When the graph spectral radius is close to 1, large $r$ and $k$  may be required, which could significantly increase computational cost.
3. The experiments are conducted on relatively small graphs. The method should also be evaluated on large-scale graphs with tens of millions of nodes to demonstrate scalability.
4. The experimental setup is vaguely described, especially regarding the choices of $r$ and $k$ . A more detailed discussion of these parameters, as well as an analysis of the actual computational (time and space) cost, is necessary.
5. The selection of baselines is outdated. More recent and relevant Graph Transformer models such as Specformer [1] and PolyFormer[2] should be included for comparison.
6. The overall writing quality could be improved. See the following minor comments for specific issues.

[1] Ma J, He M, Wei Z. Polyformer: Scalable node-wise filters via polynomial graph transformer[C]//Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024: 2118-2129.  
[2] Bo D, Shi C, Wang L, et al. Specformer: Spectral graph neural networks meet transformers[J]. arXiv preprint arXiv:2303.01028, 2023.


**Minor comments:**  
(1) The Introduction section provides only a brief overview of the proposed method and contributions; these should be elaborated further.  
(2) The mathematical notation throughout the paper is inconsistent — for instance, matrices and vectors are bolded and not in main paper and appendix. Please ensure consistency and follow the ICLR formatting style.  
(3) Baseline citations should be placed immediately after each method name (e.g., it is unclear which paper GCNIII refers to).  
(4) In Tables 1 and 2, if standard deviations are unavailable, the “±” should be omitted rather than left blank.  
(5) In Algorithm 1, the line $P \leftarrow A D^{-1}$ should be corrected to $P \leftarrow  D^{-1}A$

### Questions
Please respond to the Weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GIST, a graph transformer that steers attention using (projected) spectral embeddings while enforcing gauge invariance (invariance to rotations/sign flips within eigenspaces). The model is a multi-branch block: (i) a local graph-conv/linear-attention branch, (ii) a feature linear-attention branch, and (iii) a gauge branch with gauge-invariant and gauge-equivariant spectral attention. Authors claim linear complexity overall and show competitive results on Planetoid/PPI, mixed performance on Elliptic, and strong results on a large mesh regression task (DrivAerNet).

### Strengths
Clear conceptual core... use inner products of (projected) Laplacian eigenmaps to steer attention while remaining invariant to eigenbasis gauge; neatly explained and supported.

Random-feature JL projections to approximate spectral geometry plus linear attention to avoid $O(N^2)$ attention, embedded in a multi-scale block.

Compelling large-graph result in that DrivAerNet shows practical gains on real meshes without regridding.

### Weaknesses
The text claims linear end-to-end via the Katharopoulos et al. linear transformer, but the presented gauge-invariant/equivariant algorithms use softmax attention (Alg. 2/3). It’s unclear whether the actual gauge blocks use linear attention kernels (and if so, which feature map) and how gauge invariance is preserved under that kernelization), or only the “feature branch” is linear while the gauge path remains softmax (thus quadratic). Unless I'm missing something this point seems to critically affect the complexity claim.

For DrivAerNet the authors append Euclidean coordinates and normals to the spectral embeddings. This seems to re-introduce a coordinate-system dependence into the very vectors whose inner products are supposed to be gauge-invariant. Please clarify: are coords/normals entering only the values (feature branch) or also the Q/K used for gauge-invariant attention? 

The Planetoid/inductive tables omit several post-2022 graph-Transformer baselines that are now standard, e.g. GPS-style models with LapPE/RWPE encodings, sparse-attention Exphormer variants, and/or tokenized/CLS readouts such as NAGphormer/TokenGT. Including these would strengthen empirical positioning.

The text asserts that, in the refinement limit, similarities “recover the continuum Green’s-function kernel” and that self-attention therefore realizes a nonlocal kernel integral. This is interesting, but there is no theorem or convergence experiment (e.g., accuracy vs mesh refinement with fixed parameters) to substantiate discretization invariance beyond heuristic argument.

With a 3-branch block and added geometric channels, it’s unclear which component yields the improvement. An ablation (remove gauge path / remove local path / use softmax vs linear in each) is needed to credit the proposed gauge mechanism vs generic multi-branch modeling and extra features. 

Authors should provide wall-clock, peak memory, FLOPs, and parameter counts per branch on Planetoid, PPI, and DrivAerNet; confirm O(N) scaling end-to-end.

### Questions
Do the gauge blocks use linear attention or softmax?

If you sample a fresh R' at test time (or per graph), does accuracy remain stable?

Do coordinates/normals ever enter Q/K in the gauge-invariant path? If so, how is invariance preserved?

Operator discretization study: Fix parameters and vary mesh resolution; does error remain flat (up to sampling noise)? 

Any empirical support for Green’s-function limit behavior?

### Soundness
3

### Presentation
3

### Contribution
2
