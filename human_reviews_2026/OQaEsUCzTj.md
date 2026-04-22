# Unsupervised Multi-Scale Gromov-Wasserstein Hypergraph Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
We consider the problem of unsupervised hypergraph alignment, where the goal is to infer node correspondence between two hypergraphs based solely on their structure. Hypergraphs generalize graphs by allowing hyperedges to connect multiple nodes, and they provide a natural framework for modeling complex higher-order relationships. We introduce FALCON, a framework that effectively unifies hypergraph filtration with a multi-scale Gromov-Wasserstein consensus to solve unsupervised hypergraph alignment. The multi-scale, hierarchical structure revealed by filtration provides a set of robust, nested geometric constraints that are naturally regularized and aggregated by the GW framework. This synergy is uniquely suited to overcoming structural noise, a critical challenge where prior methods fail. Experiments on real-world datasets demonstrate that FALCON substantially outperforms state-of-the-art baselines, proving especially robust to noise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents FALCON, an unsupervised framework for hypergraph alignment that combines multi-scale hypergraph filtration with a Gromov–Wasserstein (GW) consensus formulation. Instead of reducing hypergraphs to pairwise graphs, FALCON directly aligns native hypergraph structures. It constructs a sequence of filtered subhypergraphs based on hyperedge sizes and aggregates multiple per-scale GW transport plans into a consensus alignment.
Experiments on four real-world hypergraph datasets (Pollinator, NDC-Classes, Email-EU, Dawn) show robustness to structural noise and competitive runtime compared to existing graph-alignment baselines.

### Strengths
1. The paper is well-organized with detailed experiments.
2. The idea of combining filtration with multi-scale GW consensus is elegant and reasonable.
3. FALCON consistently outperforms baselines under structural perturbations.
4. Code and datasets are promised in an anonymous repository.
5. The theoretical analysis (Theorems 1–4) provides justification for the stability and aggregation behavior of the consensus coupling.

### Weaknesses
1. The paper does not contribute new theory or algorithms to the optimal transport (OT) or Gromov–Wasserstein framework. 
The GW solver, entropic regularization, and consensus aggregation are all standard techniques. 
The contribution lies more in combining existing components rather than innovating within them.
2. Prior work (e.g., GWL, SGWL, HyperAlign) already applied GW-based alignment to graphs and hypergraphs. 
The main idea of using OT for unsupervised alignment is therefore well established.
3. All datasets used are relatively small in node count (|V| ≤ 2,290). 
The claimed time complexity $O(\xi K n^3)$ suggests limited scalability, yet no experiments on larger hypergraphs (e.g., >10k nodes) are provided. 
This makes it difficult to assess the practical applicability of large-scale real-world networks.
4. While the paper includes strong graph-alignment baselines (GWL, SGWL, REGAL, PARROT, BIGALIGN), it omits more recent or domain-relevant hypergraph matching approaches, such as CURSOR [1], which explicitly targets scalable mixed-order hypergraph matching.

[1] Zheng, Qixuan, Ming Zhang, and Hong Yan. "CURSOR: Scalable Mixed-Order Hypergraph Matching with CUR Decomposition." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16036-16045. 2024.

### Questions
1. How large can FALCON scale in practice? Have you tested it on synthetic hypergraphs with ≥10k nodes?
2. Could the method integrate node attributes or partial supervision, and how would this affect performance?
3. How sensitive is the algorithm to hyperparameters such as $\gamma$ (filtration density) and $\beta$ (entropic weight)?
4. Can the filtration criterion be adapted beyond hyperedge size (e.g., centrality, density, or domain-specific weights)?

### Soundness
3

### Presentation
3

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
This paper proposes FALCON, a method for aligning hypergraphs in unsupervised fashion, based on their structure, thereby generalizing the problem of plain graph alignment. The method generalizes previous Gromov-Wasserstein-based methods, such as SGWL, to the hypergraph setting. The experimental study compares to previous work, adapting them to hypergraphs via either a clique representation or a bipartite representation. It is unclear whether the benefits of the proposed method derive from using a native hypergraph representation, or from a methodological breakthrough that would apply to plain graphs too. The comparison is using methods that are not the current state-of-the-art, expressed in FUGAL (NeurIPS 2024).

### Strengths
S1. Solid generalization of graph alignment problem to hypergraphs.
S2. Proposal of filtration to reveal a multi-scale hierarchical structure.
S3. Experimental study vs. reasonable competitors adapted to the hypergraph setting.

### Weaknesses
W1. Unclear why the method should be specifically oriented to the hypergraph setting, while critique of prior methods appears to be methodological rather that scope-oriented.
W2. Lack of illustration of performance on the non-hypergraph-setting.
W3. Lack of comparison to current state-of-the-art-method, FUGAL.

### Questions
Why is the proposed methods proposed for hypergraphs in particular?
Could it not address and be compared on plain graphs?

### Soundness
3

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
The paper addresses the problem of unsupervised hypergraph alignment and proposes a framework named FALCON. The method combines hypergraph filtration with multi-scale Gromov–Wasserstein consensus to infer node correspondences solely from the structural information of two hypergraphs, without supervision. FALCON operates directly on the native hypergraph representation, aiming to preserve high-order relational information that is often lost in clique or bipartite graph expansions. The authors conduct experiments on several real-world hypergraph datasets and evaluate robustness under three types of structural perturbations: node removal, incidence flipping, and hyperedge addition.

### Strengths
1.Extending Gromov–Wasserstein alignment to hypergraphs and combining it with a filtration mechanism is conceptually sound.
2.The algorithm pipeline is clearly illustrated, and the paper includes ablations between uniform and leave-one-out weighting schemes.
3.The proposed method shows improved accuracy over basic clique and bipartite expansion baselines on small datasets.

### Weaknesses
1.This paper merely extends the multi-scale Gromov–Wasserstein to hypergraphs, without introducing new theory or optimization algorithms.
2.Theoretical contributions are largely superficial.No proofs for convergence or perturbation bounds under the entropic regularization are provided.
3.The key assumption of equicorrelation between scales (Theorem 3) is unsupported by data or analysis.
4.The entropic GW solver is reused without modification, and lacks an analysis of β sensitivity.
5.Featuring an excessively high algorithm complexity of O(ξKn³), the work fails to specify typical values for ξ and K, and also lacks results from large-scale datasets (e.g., >10 k nodes).

### Questions
1.Can you provide formal convergence or perturbation bounds for the entropic regularized GW optimization used in FALCON? Without these, it is unclear how stable the alignment results are with respect to noise or initialization.
2.In what way does FALCON theoretically differ from existing multi-scale Gromov–Wasserstein frameworks? Beyond extending the idea to hypergraphs, what novel mathematical or algorithmic contribution does your work make?
3.Since the entropic GW solver is adopted from prior work without modification, have you investigated the sensitivity to the regularization coefficient β? How do the hyperparameters affect convergence and accuracy?
4.The proposed method exhibits O(ξKn³) complexity, but typical choices of ξ and K are not reported. Could you clarify the actual runtime and memory usage on larger datasets (e.g., >10 k nodes) ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the unsupervised hypergraph alignment problem by employing Gromov-Wasserstein (GW) distance and proposing a new method called FALCON. FALCON's core idea is to unify hypergraph filtration with a multi-scale GW consensus. The authors test FALCON on four real-world hypergraph datasets, perturbing them with three different types of structural noise. The results show that FALCON significantly outperforms a wide range of state-of-the-art graph alignment baselin

### Strengths
1. This paper proposes a novel unsupervised hypergraph alignment method (FALCON) that effectively combines multi-scale filtration with Gromov-Wasserstein consensus.
2. This paper provides solid theoretical support for the consensus mechanism, including stability guarantees (Theorem 2) and justification for uniform weighting (Theorem 3).
3. This paper empirically validates FALCON's robustness against three distinct noise types, showing it significantly outperforms a wide array of graph-based baselines on clique and bipartite reductio

### Weaknesses
1. The core of the alignment step relies on the Gromov-Wasserstein framework, which operates on pairwise dissimilarity matrices ($C^m \in \mathbb{R}^{|V|\times|V|}$). The paper's novel dissimilarity (Eq. 3) is based on the pairwise co-occurrence of nodes ($\delta^m(u,v)$). While this is a clever way to encode hypergraph structure, it is still a pairwise projection. The method is not performing a true higher-order alignment (e.g., by matching hyperedges directly or using a tensor-based approach) but rather aligning pairwise relationships that are derived from the hypergraph.
2. The paper's primary filtration method ($\omega_{size}$) is based on hyperedge size. For a k-uniform hypergraph (where all hyperedges have the same size $k$), this filtration collapses to a single scale. This completely undermines the method's "multi-scale" nature and its core noise-mitigation strategy. The paper's only tested alternative (degree-based filtration) performed poorly and had scalability issues (Appendix F.2).

### Questions
1. What strategies or alternative filtration functions would the authors propose to effectively apply FALCON's multi-scale consensus approach to k-uniform hypergraphs?
2. Could the authors discuss the potential information loss of this pairwise projection compared to a true higher-order alignment? How much of the hypergraph's unique structure is preserved in this representation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses the problem of unsupervised hypergraph alignment and introduces FALCON, a framework that effectively integrates hypergraph filtration with a multi-scale Gromov–Wasserstein consensus. Experiments demonstrate that FALCON outperforms other state-of-the-art baselines.

### Strengths
FALCON leverages structural information across multiple scales. The problem is ultimately formulated as an optimal transport problem, which can be solved efficiently.

### Weaknesses
While I am not deeply familiar with this specific area, I have several concerns regarding the writing and technical presentation. For instance, the Gromov–Wasserstein discrepancy was not proposed in this work and should be introduced in a preliminary or problem formulation section. Additionally, the theoretical results appear relatively weak. It would be valuable to provide formal assumptions under which the method can provably achieve hypergraph alignment. The overall computational complexity of the algorithm is also not clearly stated.

In the experiments section, more details about the datasets are needed. Are these established benchmarks that have been used in prior work? Furthermore, several of the compared methods were proposed seven or eight years ago—it would be helpful to explain why they are still considered state-of-the-art and whether more recent baselines have been considered for comparison.

### Questions
See weakness. I would raise my score if my concerns are addressed.

### Soundness
3

### Presentation
2

### Contribution
3
