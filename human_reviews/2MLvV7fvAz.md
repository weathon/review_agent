# Spectro-Riemannian Graph Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Can integrating spectral and curvature signals unlock new potential in graph representation learning? Non-Euclidean geometries, particularly Riemannian manifolds such as hyperbolic (negative curvature) and spherical (positive curvature), offer powerful inductive biases for embedding complex graph structures like scale-free, hierarchical, and cyclic patterns. Meanwhile, spectral filtering excels at processing signal variations across graphs, making it effective in homophilic and heterophilic settings. Leveraging both can significantly enhance the learned representations. To this end, we propose Spectro-Riemannian Graph Neural Networks (CUSP) - the first graph representation learning paradigm that unifies both CUrvature (geometric) and SPectral insights. CUSP is a mixed-curvature spectral GNN that learns spectral filters to optimize node embeddings in products of constant curvature manifolds (hyperbolic, spherical, and Euclidean). Specifically, CUSP introduces three novel components: (a) Cusp Laplacian, an extension of the traditional graph Laplacian based on Ollivier-Ricci curvature, designed to capture the curvature signals better; (b) Cusp Filtering, which employs multiple Riemannian graph filters to obtain cues from various bands in the eigenspectrum; and (c) Cusp Pooling, a hierarchical attention mechanism combined with a curvature-based positional encoding to assess the relative importance of differently curved substructures in our graph. Empirical evaluation across eight homophilic and heterophilic datasets demonstrates the superiority of CUSP in node classification and link prediction tasks, with a gain of up to 5.3\% over state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CUSP, a graph representation learning model that integrates graph discrete curvature with a geometric extension of generalized PageRank. The method is comprehensively evaluated, demonstrating strong performance against several baselines.

### Strengths
(+++) **Novelty and Relevance**: The proposed method is new and of interest to the geometric machine learning community.

(+++) **Strong Empirical Evaluation and Performance**: The method is thoroughly evaluated, demonstrating improved performance on downstream tasks over the considered baselines.

### Weaknesses
There are several issues with the presentation that detract from the strengths. These concerns should be straightforward to address.

(----) **Presentation**: The paper’s presentation is dense and, at times, unclear. Examples include:
* **Figure 3**: The figure is very busy. While each individual component is informative and of high quality, combining them without clear visual separators makes the overall figure difficult to interpret.
* **Section 4**: This section is similarly dense, as it combines mathematical background, theoretical motivations, and the presentation of the CUSP architecture. Perhaps this section could focus on the method and architecture, with theoretical discussions (e.g., of heat diffusion) moved to Preliminaries or a new dedicated section.
* **Notation**:
  * The notation in Equations 2, 3, 5, 6, 7, and 8 is dense and may be difficult to parse for readers unfamiliar with geometric machine learning or the gyrovectorspace approach. (Also refer to "Relation to Existing Literature" below.) Adding equation annotations or using plain English function names (e.g., Enc for encoding) could improve readability.
  * "ORC for nodes" is defined in line 176 without introducing the notation $\tilde{\kappa}(x)$ which is then used, e.g., in Equation 5. (There is a notation table in the appendix, but it does not cross reference the definition.)
* **Baseline Taxonomy**: The classification of baselines in Section 5 into "spatial" and "Riemannian" is inaccurate, as the Riemannian baselines are also spatial. "Spatial-Euclidean" and "Spatial-Riemannian" could be more accurate.

(---) **Mathematical Motivation**: The justification for the Cusp Laplacian (Proposition 1) and Functional Curvature Encoding (Theorem 2) are more of rationales or motivations than rigorous proofs. For example, Proposition 1 motivates the Cusp Laplacian by introducing a modified resistance term in a heat flow equation. This would perhaps become clearer if presented as a definition, framed as, “If one assumes a resistance of the form …,” which would help the reader recognize the principles from which the Cusp Laplacian is derived.

(--) **Relation to Existing Literature**: Generalizing pipelines from Euclidean to Riemannian spaces by replacing Euclidean transformations with Moebius operations is a well-established pattern in geometric machine learning. Portions of this work follow this pattern, such as adapting PageRank GNN to product manifolds (Section 4.2) and using Moebius operations in the functional curvature encoding (Section 4.3) and cusp pooling (Section 4.4). Early works in hyperbolic graph neural networks such as [1] introduced these operations with clear motivation, via, e.g., illustrations of log and exp maps between manifolds and their tangent space. Since then, these operations have also been more broadly understood and interpreted within the framework of gyrovector spaces, aligning with Ungar's original work cited in the paper. See, e.g., [2-8]. In this work, however, the geometric components of the model are not similarly well-motivated. Perhaps a brief motivation for the operations would be helpful.


[1] Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019). Hyperbolic graph convolutional neural networks. NeurIPS.

[2] Hatori, O. (2017). Examples and Applications of Generalized Gyrovector Spaces. Results in Mathematics.

[3] Kim, S. (2016). Gyrovector Spaces on the Open Convex Cone of Positive Definite Matrices. Mathematics Interdisciplinary Research.

[4] López, F., Pozzetti, B., Trettel, S., Strube, M., & Wienhard, A. (2021). Vector-valued distance and gyrocalculus on SPD matrices. NeurIPS.

[5] Nguyen, X. S. (2022). The Gyro-structure of some matrix manifolds. NeurIPS.

[6] Nguyen, X. S., & Yang, S. (2023). Building neural networks on matrix manifolds: A Gyrovector space approach. ICML.

[7] Nguyen, X. S., Yang, S., & Histace, A. (2024). Matrix Manifold Neural Networks++. ICLR.

[8] Zhao, W., Lopez, F., Riestenberg, J. M., Strube, M., Taha, D., & Trettel, S. (2023). Modeling graphs beyond hyperbolic: GNNs in SPD matrices. ECML PKDD.

### Questions
1. Do you use the same neural networks $f_\theta$ in Line 263 for all component spaces?
2. How is the Riemannian projector $g_\theta$ in Line 347 defined?
3. Could you clarify how Theorem 1 applies to CUSP? What insight does Theorem 1 provide?
4. How does the runtime of your pipeline compare to other baselines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces CUSP: integrating mixed-curvature representation learning with GPRGNN.

### Strengths
The paper introduces a new GNN model that considers spectral information and the curvature.

### Weaknesses
- The method is incremental. It’s hard to separate the new components in the paper and how they depend on the previous work. Also, there is no theoretical justification for integrating the spectral information in the frequency domain and the curvature in the spatial domain.
- More details are needed for the heat diffusion equation and heat flow in Section 4.1. For example, as the cooling depends on the direction (from x to y), what is the role of direction in the heat diffusion equation? What is the definition of heat flow? What is the ORC distribution? What is the diffusion rate? How the Wasserstein-1 distance be interpreted as the resistance? Does the resistance depend on the direction?
- It’s unclear what “$x \sim y$ denotes adjacency between nodes x and y” means in Proposition 1 in the main text. Also, more details and motivation needed for how $\\bar{w}\_{xy} = e^{\\frac{-1}{1-\\tilde{\kappa}(x, y)}}$ is designed in main texts. Does this design have favorable properties? How does Cusp Laplacian operator act differently based on $\\bar{w}\_{xy}$?
The explanation is only in the Appendix, but an overview or high-level explanation in the main texts can help in understanding the reason behind the proposed component.
- The font size in Figure 3 is too small. It makes it very difficult to follow complicated figures.
- More explanation is needed for how GPRGNN jointly optimizes node features and topological information extraction.
- It’s unclear what does curvature domain $\mathbb{K}_{\mathbb{P}}$ represent in line 321-322. In addition, it’s unclear why in line 251, the product space is $\\mathbb{P}^{d\_{\\mathcal{M}}}$ but in line 322 the authors are interested in the $d\_{\\mathcal{C}}$-dimensional product space $\mathbb{P}^{d\_{\\mathcal{C}}}$.
- Based on Eq. (4), $\widetilde{\kappa}\in\mathbb{K}$. However, it’s unclear what $\widetilde{\kappa}(x)$ represents in Eq.(5).
- It’s unclear what the Riemannian projector is in line 347.
- It’s unclear what M2 is in line 319. There is no M2 in the paper.
- There is no theory in Theorem 2. It’s confusing to call it a theorem when the claims are only definitions.
- It’s unclear why translation invariant is a desirable property in the functional curvature encoding in the proposed method.
- It’s unclear how functional curvature encoding gives more attention to differently curved substructures.
- It’s unclear which part of the implementation is adopted from Ni et al. (2019) in line 412.
- It’s unclear what do the authors mean by they “heuristically determine the signature of our manifold $\mathbb{P}$ (i.e. component manifolds) using the discrete ORC curvature of the input graph”. It’s unclear how many hyperbolic spaces and spherical spaces are considered.
- It’s unclear what the hyperparameter L represents.
- It’s unclear how many layers are considered in the experiments.
- It’s unclear what the experimental configurations and hyperparameters of the competing methods are.
- The spacing between the tables and the main texts in Section 5 is very tight and narrow.
- The paper is missing important spectral GNNs: OptBasisGNN, ChebNetII, CayleyNet, APPNP, JacobiConv, and Specformer.
- It’s unclear how L3 is resolved using the proposed method.
- The paper needs more work on proofreading. For example:
    - the English style is not consistent. Sometimes the authors use “normalised”, but sometimes they use “normalized”.
    - The word “Laplacian” sometimes has the uppercase letter L, but sometimes it is lowercase l.
    - The tangent space notation is not consistent. 
    - There is a comma in Eq.(15), but there is no sentence afterward.
    - The exponential map and logarithmic map are defined using boldface letters, but these notations are not consistent in the appendix
    - In line 366, the sentence is not finished. The punctuation in Eq. (6), Eq. (7), and Eq. (8) is missing.
    - The notation of  Wasserstein-1 distance is not consistent in the main paper and appendix
    - The reference pointer is missing in line 405

## Minor
- Unclear what is $\mathbf{W}$ in line 141
- In line 160, missing $\times \ldots \times $ in $\mathbb{P}$
- A formal definition of Wasserstein-1 distance is missing
- Unclear what is $\psi_{xy}$ in Appendix 7.3
- In line 996, it’s unclear what is the element-wise product between a matrix $\mathbf{L}$ and a scalar $e^{\frac{-1}{1-\tilde{\\kappa}(x, y)}}$
- In Eq.(25), $\mathbf{X}_{i:}$ is not defined
- $\delta$ in line 172 and $\delta_{vs}$ in line 181 are not defined
- Unclear why $\omega_{d_f}$ in line 341, there is no $d_f$ in Eq. (4)

### Questions
- The authors keep using the term “curvature signal” throughout the paper. What does this term mathematically mean?
- In κ-right-matrix-multiplication,  why do the authors choose to work with the projection between the manifold and tangent space at the origin? How does this choice affect the empirical results?

### Soundness
2

### Presentation
1

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
The paper introduces Spectro-Riemannian Graph Neural Networks (CUSP), a novel approach to graph representation learning that combines spectral and curvature information into a spectral graph neural network operating on a manifold arising as a product of Euclidean as well as hyperbolic and spherical spaces. Traditional graph neural networks (GNNs) often struggle with diverse graph topologies and varying graph geometries/curvatures. CUSP addresses these challenges by enabling to integrate (geometric) features from both negatively (hyperbolic) and positively (spherical) parts of a given graph. This allows for the creation of more natural node embeddings that better align with the underlying structure of real-world graphs.
Key components of CUSP include (1) The Cusp Laplacian, which integrates Ollivier-Ricci curvature into the definition of a Laplacian-type matrix on the graph. (2) Cusp Filtering which allows for (high- and low-pass) filtering operatoions on a product manifold where each factor has constant positive, zero, or negative curvature. (3) Cusp Pooling A hierarchical attention mechanism that evaluates the importance of substructures with different curvature characteristics.
In the empirical evaluation CUSP's performance is investigated across eight datasets. Here CUSP achieves a good performance (in node classification and link prediction tasks), with sometimes substantial gains over baselines.
The research seems to be novel and highlights the potential that  combining geometric and spectral information harbours for the design of GNNs.

### Strengths
The paper is well structured and mostly (please see next section) well written. A positive example is the explicit  description of the limitations of previous work that are addressed in this paper (i.e. L1-L3 in the introduction).

 Including the notation table in Appendix 7.1 helps to keep track of the various mathematical concepts.

The idea underlying the introduced CUSP Laplacian of including curvature information into the Laplacian-matrix describing the graph is neat. 

Furthermore the idea of taking into account locally varying curvature structures by allowing for factors of varying curvature in the product manifold is nice. 

The performance of the proposed architecture in both the node-classification and link-prediction tasks on the considered datasets is solid. 

The ablation study on the impact of the signature of the product manifold structure (c.f. Section 5.1) as well as the surrounding discussion is illuminating.

### Weaknesses
I do have _some_ concerns regarding readability. An easy fix is the image quality in Figure 1. Here the axis labeling of the histograms is not  readable if the paper is printed. Could the authors please fix this by including higher resolution images and/or using a larger font size for axis labeling. 

Regarding the paper itself, aside from some typos and grammatical errors that do not impede the flow of the paper too much, I had trouble understanding Section 4.3; especially from line 327 onward: The curvature kernel is defined twice: once using an inner product in an ambient space, and once as a translation invariant entity. I believe what the authors want to convey is that the former definition as utilized in the paper already leads to a translation invariant kernel. Is this correct?

Also the significance of the Bochner-Minlos theorem is not immediately apparent to me. I only gained some intuition about this after reading the proof of Theorem 2 and Theorem 3 in the Appendix. Could the authors comment more explicitly on the significance of Bochner's theorem here? 

It might also be good to explain (to some extent) and motivate the k-stereographic model in the main text. 
While I have some background in differential geometry, I had only ever come across standard stereographic projections.
Especially for readers from a pure CS/EE background more details here might be useful, even if the model might be central to Riemannian GNNs. 

In the same direction, it would also be good explain a bit more the respective operations in Table 8 in Appendix 7.2.4 and how they are natural.


Finally, in the experimental sections, the datasets that are being used are somewhat old and fairly small. I strongly encourage the authors to also consider newer and much larger datasets  (e.g. using OGBN) and compare their approach with baselines there.

### Questions
1) What makes GPR special and why is this used as a backbone? I believe it is equivalent to any polynomial-filter spectral GNN. Why not e.g. use ChebNet, etc.?

2) The (total) dimensions of the product manifolds  of e.g. Table 5 and Table 2 seem to always be  $d=48$. Yet in Table 13 of Appendix 6.6.4 it is indicated that $d = 64$ is the selected dimension of the product Manifold. Could the authors comment? Also, while I have seen Algorithm 1 in Appendix 7.6.5 (which seems to be used as an initial heuristic), it is still not clear to me how individual dimensions of product manifolds and respective factors are found/optimized. Is a grid search performed? If so what are the respective allowed values during this grid search? Could the authors clarify (again?)?

3) In the ablation study on the impact of the CUSP Laplacian, performance using the CUSP Laplacian is compared to performance using the adjacency matrix instead. Could the authors repeat this ablation study comparing with the usual normalized and unnormalized graph Laplacians? This would shed light on whether the performance increase comes from using a Laplacian-type matrix vs. an adjacency type matrix, or indeed from the specific properties of the the Cusp Laplacian.

4) Is it possible to include and discuss some basic spectral characteristics of the CUSP Laplacian (beyond self-adjointness and positivity)? As is, this new matrix is introduced without too much intuition beyond the heat-flow heuristic. Can something e.g. be said about the maximal eigenvalue or first-non-trivial eigenvalue (in a Cheeger-type fashion) for example? I realize the present paper is not mainly theoretical but introduces an architecture. However some additional theoretical foundation would indeed be nice.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is the first attempt that unifies spectral and curvature signals in graph representation learning. CUSP, a mixed-curvature spectral GNN, introduces three novel components: a curvature-aware Cusp Laplacian operator, a mixed-curvature spectral graph filtering framework, Cusp Filtering, and a curvature embedding method using classical harmonic analysis and a hierarchical attention mechanism called Cusp Pooling. CUSP records state-of-the-art performance on eight real-world benchmarking datasets for Node Classification (NC) and Link Prediction (LP) tasks.

### Strengths
- Originality. I think the proposed CUSP is innovative in the paradign of graph representation learning. CUSP is well-motivated, and indeed improves the performance significantly.
- Quality. The paper is well structured. A large number of experiments have been conducted, which are supportive and convincing.
- Clarity. The method proposed is deeply explored and proofs are presented as well as a detailed description of the algorithm and the rational for the design consideration that were made. The experimental section is typically sufficient. And ablation experiments fully illustrate the effectiveness of each component.
- Significance. The paper shows state-of-the-art in standard benchmarks. This paper explores how to integrate spectral and curvature signals in graph representation learning, and gives some valuable insights.

### Weaknesses
As mentioned in the questions, the proposed OCR alternative method should be described in detail, or the proposed method should be used in the experiment.

### Questions
In the appendix, the complexity of ORC is analyzed, and it is mentioned that an alternative method is provided to approximate edge ORC in linear time, which is applicable to very large (billion-scale) real-world graphs. How to prove that the proposed alternative method is useful in large real-world graphs? Are there any experiments? As far as I know, the dataset used in the experiment is on the order of 100,000.

### Soundness
4

### Presentation
4

### Contribution
3
