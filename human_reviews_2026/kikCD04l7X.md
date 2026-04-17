# SuperHype: Hypergraph Generation via Graph-Superposition Decomposition

- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
Hypergraphs are graph generalizations with key applications in domains such as healthcare, where strict data privacy requirements apply, or bioinformatics, where testing new compounds is costly. However, research into hypergraph synthesis is limited, and state-of-the-art approaches yield limited generation quality in terms of overall structural patterns and graph-level validity. This is caused by the hypergraph's combinatorial structure, which is composed of a number of possible hyperedges that is factorial in the number of nodes. In fact, current solutions rely on diffusion models denoising graph projections, which are exact but inefficient, or lightweight but approximate. To address such shortcomings, we introduce SuperHype, the first hypergraph diffusion model with tractable and exact modeling. To tackle the complexity of hypergraph representation, we introduce graph superposition, a novel representation that embeds a hypergraph into a multilayer graph. Superposition enables a tractable representation that maintains exactness. To generate new samples from such representations, we introduce a Graph-Superposition Transformer that treats the superposition as an interconnected sequence of layers. We optimize the model architecture to learn low-level patterns within individual graphs in the superposition and high-level patterns between the different graphs of the same superposition. Moreover, we enhance the model's performance with hypergraph-specific auxiliary features and triplet aggregation of indirect node interactions. Our evaluation on five datasets shows that \algo generally reproduces local and global connectivity patterns with superior fidelity to state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SuperHype, a diffusion-based framework for hypergraph generation. The core idea is to represent a hypergraph as a graph superposition — a set of layered graphs that collectively preserve hyperedge information while keeping the representation tractable. The authors design a graph-superposition transformer to perform diffusion across and within these layers, augmented with auxiliary clique features and triplet aggregation to capture higher-order dependencies. Conceptually, the method reframes hypergraph synthesis as denoising over these layered graph projections rather than directly modeling hyperedges.

### Strengths
- The paper explores a relatively underexplored research direction in hypergraph generation.
- Overall, the manuscript is clearly written, and the design choices are well motivated.
- The empirical results are strong and support the proposed approach.

### Weaknesses
- The description of the Transformer (l. 306–316) is vague, relying on imprecise terms such as “mix” or “model interaction.” Figure 3 also fails to provide a clear structural overview of the model.
- In Table 1, it would be more informative to report the proportions of valid, unique, and novel graphs rather than only the valid ones, which would also resolve the asterisk issue for HyperPLR.
- The evaluation is limited to synthetic datasets, which does not strongly support the practical contribution of the work.
- Minor notation issues: $\mathcal{P}_2(e)$ is undefined, and the notation $\mathcal{V}$ and $\mathcal{E}$ seems to be used for both the vertex/edge sets and their cardinalities, causing ambiguity.
- In the definition of the loss, should $\mathcal{E}_l^t$ actually be $\mathcal{E}_l^0$?

### Questions
- The restriction to 3- and 4-cliques in the experiments is reasonable from a computational standpoint. However, the claim that this choice “covers most of the hyperedges (over more than two nodes)” across datasets should be supported by quantitative evidence for each dataset.
- Please report inference times to allow for a fair assessment of the method’s computational efficiency.
- Regarding the discussion on evaluation relative to the training or test dataset: have you considered simply generating more data? In practice, this would not necessarily increase training time, as a larger dataset can lead to faster convergence and reduced variability between training and test sets.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SuperHype, a new diffusion-based model for generating hypergraphs. The key novelty is to represent a hypergraph as a graph superposition, which is a set of layered graphs whose maximal cliques together reconstruct all hyperedges. The authors design a Graph-Superposition Transformer for denoising the diffusion, and enhance it with clique-based auxiliary features and triplet aggregation. Experiments on five datasets show that SuperHype outperforms existing methods like HYGENE and HyperPLR on most metrics.

### Strengths
1. Generating realistic hypergraphs is an important topic to research.
2. The idea of breaking down distribute the hyperedges into the maximal cliques of multiple layered graphs, i.e. "graph superposition" is interesting and novel, although much further justification is needed -- see weaknesses below
3. Experiment design is overall reasonable; the results also show consistent improvement over baselines.

### Weaknesses
1. My key concern is lack of theoretical guarantee of the graph superposition projection process. For example, it would be very helpful to have theoretical insights about under what conditions the superposition projection algorithm can succeed (or fail); what's the relationship between the layer number d and the probability of success of the algorithm (and how do we choose a small enough d); are there many hypergraphs that fail the algorithm (which is closely tied to the general applicability of your method)

2. The denoising process also seems to lack enforcement/usage of some hard combinatorial constraints. For example, how can you guarantee that the generation result is (of high confidence) containing a set of "proper" maximal cliques. A counter example would be that you generate, say, a 10-complete graph with only 1 random edge missing. We know that in reality this structure is very unlikely the clique expansion of a real hypergraph. Also,  my sense is that different layers also have some hard combinatorial constraints with each -- I think it would be helpful to explicitly figure out what these constraints are, and how we can utilize them to help with denoising. 

3. Typos: “The model ouperforms Gailhard et al. (2025)”: "ouperforms" should be “outperforms”; "an hypergraph" should be "a hypergraph"

### Questions
See weaknesses.

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
2

### Summary
The paper proposes a diffusion framework SuperHype for hypergraph generation. It decomposes a hypergraph into a small stack of ordinary layers, where maximal cliques in each layer map injectively to hyperedges. The paper borrows a graph-superposition transformer for within-layer and cross-layer message passing, augmented by hypergraph-specific features. Experiments show improved performance compared to baselines.

### Strengths
1. Hyper graph generation is a very classical yet important field. This paper tries to tackle this task via both modeling and theoretical way, which I feel is valuable.

2. Graph superposition achieves an injective mapping from layered cliques to hyperedges. If this condition could hold, the it should be a neat way to preserve high-order structure during duffision.

3. The paper integrates the discrete diffusion process with a graph superposition transformer, which looks novel to me.

### Weaknesses
1. Why is the number of hyperedges is N^2, not 2^{C(N, 2)} - 1?

2. In graph superposition projection, the paper claims that it uses a greedy algorithm to generate a graph superposition from a hypergraph in O(Ed). However, is the complexity for MaximalCliques in algo 1 linear?

3. The paper mentions memory cost and complexity analysis but does not conduct any experiments to justify this.

4. All the results from experimental session lack statistical significance. I would suggest to add error bound on them.

5. In Table 1, the proposed method does not yield the best results but still highlighted on Tree Hypergraphs.

### Questions
Please refer to my comments above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper derives a novel decomposition of hyper graphs  into layered clique decompositions which preserves the graph exactly while retaining the compactness of the clique decomposition (at the cost of not necessarily always existing). The authors then present an  adapted architecture for it based on the Digress GAT, adding sharing mechanisms that allow information to flow between the layers of the decomposition, as well as adding auxillary features constructed specifically for the hypergraph case . The method is evaluated on ER,SBM,tree and ego-hypergraphs, against HYGENE and its baselines, as well as HyperPA and HyperPLR. the framework is presented as overall competetive/best in class across the datasets

### Strengths
1. originality: resolving the tractability/exactness problem of hypergraph to graph embedding via effectively random projections similar to how sliced wasserstein approximations work  (using the terms a bit loosely) is a clever idea, even if it comes without guarantees
    
2. quality: evaluation done mainly rigorously
    
3. clarity: well written and legible
    
4. significance: strong improvements on most datasets

### Weaknesses
- line 41 “These recently proposed hypergraph synthesizers, albeit they are based on architectures that are unfit for hypergraphs’ characteristics, and bring limited generative capabilities.” parses weirdly, seems to be some editing leftover    
- would like to see multiple seeds, CIs (stochasticity in the decompositoin could blow up variance in performance)

### Questions
- address the CI issue please
- can the different layers be modeled as a single big graph? why and why not? (scaling I assume? )

### Soundness
3

### Presentation
3

### Contribution
3
