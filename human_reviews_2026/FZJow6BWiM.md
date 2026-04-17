# HOG-Diff: Higher-Order Guided Diffusion for Graph Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Graph generation is a critical yet challenging task, as empirical analyses require a deep understanding of complex, non-Euclidean structures. Diffusion models have recently made significant advances in graph generation, but these models are typically adapted from image generation frameworks and overlook inherent higher-order topology, limiting their ability to capture graph topology.
In this work, we propose Higher-order Guided Diffusion (HOG-Diff), a principled framework that progressively generates plausible graphs with inherent topological structures. HOG-Diff follows a coarse-to-fine generation curriculum, guided by higher-order topology and implemented via diffusion bridges. We further prove that our model admits stronger theoretical guarantees than classical diffusion frameworks. Extensive experiments across eight graph generation benchmarks, spanning diverse domains and including large-scale settings, demonstrate the scalability of our method and its superior performance on both pairwise and higher-order topological metrics. Our project page is available [here](https://circle-group.github.io/research/hog-diff/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose the Higher-order Guided Diffusion (HOG-Diff) framework that guides the diffusion process to preserve higher-order topological structures in a coarse-to-fine manner. HOG-Diff utilizes cell complex filtering (CCF), which extracts hierarchical skeletons used as a generation guidance and enables efficient computation. The effectiveness of HOG-Diff is demonstrated through extensive experiments on molecular and benchmark graph datasets with autoregressive and one-shot graph generation methods.

### Strengths
- Novel and interesting idea of integrating topology and diffusion as a guiding structure.
- Solid theoretical grounding suggests how GOU process and Doob’s h-transform can be derived to guide topological information in diffusion process, as well as the error bounds.
- Comprehensive evaluation on various graph datasets using a wide range of baseline graph generative methods.
- Insightful dataset-level analysis, in terms of the discussion on dataset-dependent effects (e.g., weaker improvement on Ego-small due to limited higher-order structures).

### Weaknesses
- Lack of topological assessment: The authors do not provide evaluation(quantitative or qualitative) on higher-order topology preservation in the generated graphs.
- Notation and presentation complexity: I understand it may be somewhat inevitable, but the excessive use of notations is hard to follow for readers.

### Questions
- As mentioned in the weakness section, can you provide quantitative and qualitative evaluation on the generated graphs that could verify the successful preservation of higher-order structures? Such information will support the claim that the performance improvement lies on the preservation of higher-order structure. (e.g., MSE of persistence image, Curvature Filtrations [1], etc.)
- What advantages does guidance have over direct conditioning on higher-order information in the diffusion process? (other than interpretability)
- The term “consistently outperforms” in L400 seems little excessive, as only the NSPDK and FCD scores in Table 1 shows improvement whereas the other three metrics differ.
- What exactly do you mean by explicitly preserves higher-order structures in L399 about Fig.3? I don’t see the alignment of specific structures like the connected components or loops. (Also, the light-blue edges are extremely hard to see. I recommend changing the color to be more visible.)
- Can you clarify the novelty of CCF? I am aware of the idea of simplifying simplicial complex while preserving topological invariants used in discrete morse theory or other TDL works. It would be nice to show how CCF differs from such methods.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new diffusion-based graph generative model that integrates higher-order topological information to guide the generation process. The proposed mechanism explicitly leverages cell complex filtering to learn higher-order structures and employs diffusion bridges to enable a coarse-to-fine generation scheme. This design aims to capture richer topological properties and improve the structural coherence of the generated graphs. Experimental results on both molecular and generic graph benchmarks demonstrate performance comparable to recent state-of-the-art graph generative methods.

### Strengths
- The use of higher-order topological structures to guide the diffusion process represents a novel contribution. The alignment of the different diffusion stages with hierarchical graph structures seems an innovative way of capturing complex structural patterns.

- The proposed framework is theoretically grounded and proves to be an effective extension of standard diffusion models to non-Euclidean domains.

- Empirical results are competitive with state-of-the-art methods, showing the potential of the proposed approach.

### Weaknesses
- The reported performance improvements are relatively modest. Without an accompanying analysis of computational efficiency or scalability, the practical advantages of introducing higher-order topology remain unclear.

- The methodological presentation of the framework could be more intuitive. Although the paper explains how higher-order structures are obtained and integrated into the diffusion process, the overall exposition is difficult to follow, as it often mixes high-level intuition with theoretical descriptions.

- The paper would benefit from a clearer discussion of the potential limitations and possible extensions of the proposed approach, such as scalability, conditional generation for broader applicability across domains, and sensitivity to the choice of higher-order representations. It would be relevant to examine how the model scales with graph size (the SBM dataset, while described in the paper as large-scale, is of relatively modest size) and how conditional generation could be incorporated alongside the higher-order structures.

### Questions
- **Hierarchical time windows:** What is the specific role of the number of hierarchical time windows $K$ in the model? If it is treated as a hyperparameter, how is its value determined and how it is related to the number of noising steps? How do changes in $K$ affect the model‚ as generative dynamics?

- **Computational cost:** The model introduces several components that could significantly impact computational efficiency, such as the Cell Complex Filtering operation, the injection of noise into the Laplacian instead of the adjacency matrix (which requires recomputing the Laplacian at each noising step), and the different modules in the score network (GNC and Transformer blocks). Although the authors provide a theoretical argument that the framework converges to the target distribution at an equal or faster rate than classical models, there is no empirical evidence supporting this claim. Having this analysis would be particularly relevant, as the reported performance gains over existing methods are not substantial. Therefore, demonstrating that the added complexity of using higher-order structures translates into practical efficiency or quality improvements (beyond the provided ablation) would strengthen the paper.

### Minor questions

- Why do the authors use different sets of baselines across experiments? Is there a specific reason for this?

- The authors claim that existing graph generative models are ill-suited to capture topological properties because they are largely adapted from image generation frameworks. They further argue that the adjacency matrix quickly degrades into a dense matrix with uniformly distributed entries, which is detrimental to graph structure preservation. However, the paper would benefit from clearer empirical or theoretical evidence supporting these claims. In particular, several recent graph diffusion models employ diffusion processes on discrete state-spaces that are adapted for graph generation and have shown strong performance in maintaining graph topology. As a justification for the need to learn higher-order topology, the paper mentions that the framework should ensure equivariance. However, many existing graph generative models already satisfy equivariance, so this motivation is a bit unclear. In particular, it would be relevant to have proof that the proposed model respects equivariance.

- Could the higher-order information be further exploited to enhance scalability, for example, by incorporating higher-order structure generation within a latent diffusion framework?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HOG-Diff, a novel diffusion framework for graph generation. The core methodology involves "lifting" graphs into a higher-order cell complex representation and make the diffusion process respect higher-order topological structures. The authors posit that this higher-order diffusion better captures the intrinsic topological structure of graphs. The paper provides a theoretical analysis of the proposed methodology and presents experimental results to demonstrate its performance.

### Strengths
- The idea of graph generation via lifted is refreshing.
- The paper is built upon a solid mathematical formulation.
- The theoretical foundations, including theorems and propositions, are well-presented, and the accompanying proofs provided in the appendix appear to be correct.

### Weaknesses
Despite its theoretical grounding, the paper suffers from significant weaknesses in its justification, clarity, and empirical evaluation.

- **Clarity of Writing and Notation.** A primary concern is the clarity of the writing and the logical flow. The notation, in particular, seems inconsistent and potentially overloaded. For instance, Proposition 2 introduces the $p$-cell complex filtered graph $G_p$, where the subscript $p$ denotes the cell dimension. However, in subsequent sections, graphs are denoted as $G_t$, where $t$ represents time. It is unclear if $G_t$ retains a connection to a specific $p$-cell filtration or if this notation is dropped. This ambiguity makes the methodology difficult to follow, thus hinders a precise understanding of the manuscript’s main contribution and the relevant algorithm design. 
- **Justification of Core Motivation and Theoretical Bound.** The central motivation of the paper is that lifting graphs to a higher-order space improves generative fidelity, which I believe requires stronger justification. While higher-order representations are beneficial for systems with inherent group-level interactions (e.g., co-authorship networks), this does not mean converting graphs to their high-order counterpart would be beneficial for downstream tasks. As shown in [1], a hypergraph and its corresponding graph clique expansion can be information-theoretically equivalent in certain contexts. The paper does not sufficiently argue why this lifting operation provides a superior inductive bias for learning the data distribution. This concern extends to the primary theoretical result (Theorem 4). The analysis demonstrates that the final step of the HOG-Diff curriculum (from $\tau_1$ to $0$) achieves a sharper error bound than a classical model's entire process (from $T$ to $0$). This comparison does not seem equitable. The proof relies on $\tau_1 < T$ and a superior (coarsened) starting point which induces $\mathcal{E}(\tau_1) \le \mathcal{E}^{\prime}(T)$. A more convincing analysis would compare the entire $K$-step generative process of HOG-Diff (from $T \to 0$) against the classical $T \to 0$ process, rather than isolating the final, and arguably simpler, refinement step. I might have misunderstanding on the theoretical bound and would appreciate if the authors can correct me. 
- **Critique of Diffusion Model Baselines.** The paper's motivation appears to be based on a critique of graph diffusion models. The claim that diffusion models are "ineffective at modeling the topological properties" (Lines 70-72) overlooks recent successes in discrete-state diffusion. Similarly, the criticism that "the graph adjacency matrix quickly degrades into a dense matrix" (Line 72) and that injecting "isotropic Gaussian noise...is detrimental" (Line 76) primarily applies to early continuous diffusion models. This critique does not hold for more recent discrete frameworks such as DiGress [3], Cometh [2], and DeFoG [4], which deviate from the Gaussian noise paradigm. The paper should position its contribution against these more relevant state-of-the-art discrete models, as the problems it claims to solve have, to a large extent, already been addressed.
- **Significance and Fairness of Empirical Evaluation.** The empirical evaluation lacks significance due to the omission of standard benchmarks for graph generation. Datasets such as SBM, Planar, and Tree graphs are critical as they possess ground-truth topological properties (e.g., planarity, acyclicity). Evaluating validity on these datasets is fundamental to assessing a model's ability to capture intrinsic graph patterns. The paper's reliance on statistics ratio (degree, clustering) and molecular validity (which is basically valency validity) does not sufficiently demonstrate that the model can learn and respect these fine-grained topological constraints, especially in plain graph generation tasks. Furthermore, the comparison to existing work is incomplete. Key SOTA diffusion models, such as Cometh [2], are not included. In the molecular benchmarks (QM9, ZINC250k), the reported improvement over DeFoG [4] appears marginal, but a proper comparison is impossible as standard deviations are not provided.
- **Scalability and Computational Complexity.** Scalability remains an unanswered question. All experiments are conducted on relatively small-scale graphs. Standard, larger benchmarks (e.g., MOSES, GUACAMOL for molecules; SBMs, the larger version of COMM-20) are absent. This is critical as the core premise of HOG-Diff relies on computing higher-order structures. It is unclear (1) whether computing these structures is computationally feasible for large graphs, and (2) how the performance and complexity of HOG-Diff scale relative to standard graph generation models on such data. Although Appendix D discussed computational complexity, it would be good to see empirical validation as well. 
- **Clarity of Algorithmic Details.** The paper would benefit from a more precise definition of its algorithmic settings in the main text. Key details, such as the exact dimensions and representations of $G$ and $G_p$, are not sufficiently clear. This lack of detail hinders a rigorous analysis of the algorithm's space and time complexity.

References

[1] B. Tang et al. Training-free message passing for learning on hypergraphs. ICLR 2025

[2] A. Siraudin et al. Cometh: A continuous-time discrete-state graph diffusion model. TMLR 2025

[3] C. Vignac et al. DiGress: Discrete Denoising diffusion for graph generation. ICLR 2023

[4] Y. Qin et al. DeFoG: Discrete Flow Matching for Graph Generation. ICML 2025.

### Questions
Most of my questions are raised in weakness. This includes the problem of writing, the main motivation of the paper, the significance of the empirical studies and the scalability. Please refer to that section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HOG-Diff, a diffusion-based graph generation framework that explicitly incorporates higher-order topological structures (e.g., cell complexes) through a coarse-to-fine generation curriculum. The method leverages generalized Ornstein-Uhlenbeck (GOU) bridges in the spectral domain and demonstrates superior performance on molecular and generic graph generation tasks, supported by theoretical guarantees.

### Strengths
Introduces a Higher-Order Guided Diffusion model, which for the first time explicitly incorporates higher-order structures (e.g., cell complexes) into graph generation tasks, addressing the limitations of existing diffusion models in capturing graph topological properties.

Designs a Cell Complex Filtering (CCF) operation to enable a coarse-to-fine generation process, simplifying the modeling of complex graph distributions.

Achieves  superior results on molecular datasets (QM9, ZINC250k) and generic graph datasets.

### Weaknesses
The proposed coarse-to-fine process and cell complex filtering could be computationally expensive, especially for large graphs or dense higher-order structures. No runtime is provided.

Although the authors emphasize interpretability as a motivation, there are few visual or quantitative analyses explaining what “higher-order topology” the model actually captures or how it improves the generation quality.

### Questions
Can the proposed filtering and bridge approach scale to larger graphs?

How sensitive are results to the number of hierarchical stages？

### Soundness
3

### Presentation
3

### Contribution
3
