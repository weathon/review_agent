# A Spectral Framework for Evaluating Geodesic Distances Between Graphs

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 0, 2

## Abstract
This paper presents a spectral framework for quantifying the differentiation between graph data samples by introducing a novel metric named Graph Geodesic Distance (GGD). For two different graphs with the same number of nodes, our framework leverages a spectral graph matching procedure to find node correspondence so that the geodesic distance between them can be subsequently computed by solving a generalized eigenvalue problem associated with their Laplacian matrices. For graphs of different sizes, a resistance-based spectral graph coarsening scheme is introduced to reduce the size of the larger graph while preserving the original spectral properties. We show that the proposed GGD metric can effectively quantify dissimilarities between two graphs by encapsulating their differences in key structural (spectral) properties, such as effective resistances between nodes, cuts, and the mixing time of random walks. Through extensive experiments comparing with state-of-the-art metrics, such as the latest Tree-Mover's Distance (TMD), the proposed GGD metric demonstrates significantly improved performance for graph classification, particularly when only partial node features are available. Furthermore, we extend the application of GGD beyond graph classification to stability analysis of GNNs and the quantification of distances between datasets, highlighting its versatility in broader machine learning contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The core of this research is the Graph Geodesic Distance (GGD), a spectral metric that quantifies graph differences. It works by using spectral graph matching to align nodes, followed by calculating the geodesic distance between their Modified Laplacian Matrices. For graphs of unequal size, it employs a resistance-based coarsening scheme. The method require point alignment which is done prior to the metric calculation.

### Strengths
The main strength of this approach is its complexity O(N^3) versus TMD.  On top of that the switch to spectrum comparison provides also superior results on graph classification as it handles structural dissimilarities (spectrum) better.

### Weaknesses
The main weakness is the need of point to point matching. This dependance puts limit on the size of the data that is feasible for comparison and puts a shade on the advantages of the spectrum comparison because it relies on point-to-point in pre-calc.

### Questions
Regarding Scalability and Approximation:  For large-scale graph comparisons, the paper suggests approximating GGD using a small fraction of extreme eigenvalues to maintain a low runtime. Any results or insights on the limitations and success of sub-graphs?

Regarding Generalized Graph Structures: Since the current framework is limited to simple, undirected graphs, what specific changes would be required in Phase 1 (Spectral Graph Matching) and Phase 2 (GGD Calculation) to successfully extend the GGD metric to handle directed graphs or graphs that incorporate higher-order structural information?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Graph Geodesic Distance (GGD), a spectral-geometric framework for measuring distances between graphs. Each graph is represented by a modified Laplacian (made SPD by adding a small diagonal value), and the distance between two graphs is defined as the affine-invariant Riemannian geodesic between their Laplacians. For graphs of different sizes, a resistance-based spectral coarsening is introduced to reduce the larger graph while preserving spectral structure. Empirical results show that GGD outperforms Tree Mover’s Distance (TMD), Weisfeiler–Lehman kernels, and several GNN baselines on graph classification, especially when node features are missing. GGD also correlates better with GNN embeddings for stability analysis and runs roughly 6–9× faster than TMD. This is a carefully executed and well-written paper that applies SPD manifold geometry to graph comparison in a principled way. The work is incremental but high-quality and will be of interest to researchers in geometric and spectral graph learning.

### Strengths
* Strong mathematical grounding: GGD is rigorously defined on the SPD manifold with formal proofs that it satisfies the metric axioms (identity, symmetry, positivity, triangle inequality). The connection between generalized eigenvalues and “cut mismatches” provides a meaningful spectral interpretation.
* Empirical performance: Demonstrates steady improvements (5–10 pp accuracy gains) over TMD and GNN baselines on benchmark datasets, and robustness under partial or missing node features.

### Weaknesses
* Expensive and small-scale: The method still requires multiple cubic-time steps (eigendecomposition, assignment, generalized eigenvalues). Approximate variants are described but not fully benchmarked on large graphs.
* Limited exploration of SPD metrics: Only AIRM is fully tested; Log-Euclidean (LERM) appears briefly in the appendix and performs similarly. Other SPD metrics or linear approximations could be compared.

### Questions
* Have you considered simpler or faster SPD metrics (e.g., Power–Euclidean) to trade some invariance for scalability?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces a graph distance measure: Graph Geodesic Distance (GGD).

To compute GGD, the paper uses previous work by Fan et al. (2020) to perform spectral graph matching between the two input graphs. It then calculates GGD based on the eigenvalues of the Laplacian matrices of the matched graphs. This approach is not novel, and the authors list previous works that measure the distance between two graphs using the eigenvalues of their Laplacian matrices.

Side note: There is a typo on line 482: "LLM USEAGE" should be "LLM USAGE".

### Strengths
Measuring distances on graphs is an important problem that has been studied for centuries. I suggest the authors review "netrd: A library for network reconstruction and graph distances" (https://joss.theoj.org/papers/10.21105/joss.02990) and the paper "Network comparison and the within-ensemble graph distance" (https://royalsocietypublishing.org/doi/10.1098/rspa.2019.0744).

### Weaknesses
-The Graph Geodesic Distance (GGD) is not a metric because it does not satisfy both directions of the identity of indiscernibles axiom: the distance between two points is zero if and only if they are the same point. This failure arises from the co-spectrality problem, where two graphs can have the same spectra but be different. Thus, GGD is a pseudo-metric.

-GGD is not scalable to large graphs with millions of nodes because it requires spectral matching and eigenvalue computation.

- Neither phase of GGD is new, as the authors point out in their manuscript.

### Questions
- If you disagree with GGD being a pseudo-metric, please provide a proof for both directions of the identity of indiscernibles axiom.

- Please show that the manifold assumptions hold across all graphs (e.g., social networks, biological networks, transportation networks, etc). You may use graph ensembles if you wish, where you know the processes that generated the graphs.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new spectral distance function to quantify the difference between two graphs of equal order.
The core idea is based on spectral graph matching (or alignment) process, drawing from a single previous work, GRAMPA.
A coarsening method is also proposed to reduce one graph's size and render it identical to that of another input graph.
An experimental study compares the proposed method to selected measures, such as the Tree-Mover’s Distance (TMD).

### Strengths
S1. Addresses the problem of graph distance measurement in a novel manner.

S2. Creatively exploits previous work in the area.

S3. Conducts experiments vs. a single previous metric.

### Weaknesses
W1. It is unclear why the new spectral distance function is necessary, and how it compares to established distance function such as the Frobenius norm between adjacency matrices and related graph alignment methods.

W2. It is unclear how the proposed spectral comparison methodology differs from already existing spectral graph alignment methods such as GRASP [TKDD 17(4)].

W3. It is unclear how the proposes spectral quantification of graph difference related to existing spectral signatures, such as NetLSD [KDD 2018].

### Questions
Address W1-W3.

### Soundness
2

### Presentation
3

### Contribution
2
