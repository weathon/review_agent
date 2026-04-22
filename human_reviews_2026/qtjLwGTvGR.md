# Heat Kernel Goes Topological

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
We propose a scalable framework for topological deep learning that uses the Heat Kernel Signature (HKS) as a node descriptor on combinatorial complexes. Unlike existing topological neural networks, which rely on expensive higher-order message passing, our approach leverages the Laplacian operator to compute multiscale HKS embeddings that are permutation-equivariant and readily integrated into traditional general deep learning methods. This enables efficient learning while preserving rich structural information.
Theoretically, our method achieves maximal expressivity within the limits of spectral techniques, distinguishing complexes up to isospectral equivalence. Empirically, it outperforms existing topological baselines in computational efficiency while maintaining competitive accuracy on molecular property prediction benchmarks. On specialised topological tasks, it shows superior ability to separate complex structures and overcome known blind spots of prior methods.
Overall, our results establish HKS-based node features as a powerful primitive for topological representation learning, offering a principled trade-off between expressivity and scalability and opening new directions for applying topological methods in molecular and geometric machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on the computation of a class of node descriptors called Heat Kernel Signatures (HKS). This is achieved by applying the thermal kernel function/radial basis function (RBF)/square exponential (SE) kernel function to the eigendecomposition of a Laplacian defined on combinatorial complexes (CC). A multi-scale effect is obtained by computing this at different scales and then concatenating them. The HKS are then concatenated with the node features and passed to a neural network pipeline (either the MLP-Mixer or the Transformer-Mixer) for classification tasks.

### Strengths
The strength of this paper lies in the fact that it is a simple alternative to message-passing (MP) graph neural networks (GNNs) and also covers combinatorial complexes. 

The idea of applying a radial basis function (RBF) kernel (essentially identical, except for a constant, to the heat kernel) and, more generally, performing functional calculus on a Laplacian-based kernel is a effective approach that has been used in numerous works such as [1] [2] [3] [4] [5] [6].

[1]: Kernels and Regularization on Graphs by Smola and al.

[2]: The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains by Shuman and al.

[3]: Heat Kernels, Manifolds and Graph Embedding by Bai and al.

[4]: Matérn Gaussian Processes on Graphs by Borovitskiy and al. 

[5]: Gaussian Processes on Cellular Complexes by Alain and al. 

[6]: Hodge-Compositional Edge Gaussian Processes by Yang and al.

### Weaknesses
My first concern is that some relevant related works are missing. In particular, instead of examining the simple heat kernel, the papers [6] [7] [8] [9] [10] [11] construct multi-scale descriptors from Fourier and wavelet transforms [12]. Although these papers involve Gaussian processes, the construction of their descriptors remains very similar to those of HKS. 

A notable difference is that these methods naturally and intrinsically combine descriptors and features (by computing Fourier and wavelet coefficients and then aggregating them) instead of concatenating them. However, it is also possible to take no features and obtain a formulation similar to that of HKS. Here, aggregation can be replaced by the Transformer Mixer or MLP Mixer to achieve a similar effect. 

I believe the section 4.1 on theory takes up too much space. Most of the content is already known or largely follows from previously known results. Although it's good to have it, I don't think it was absolutely necessary, and it may not warrant two pages in the main text. 

The experimental results are not sufficiently comprehensive. More experiments should be conducted, especially on higher-order combinatorial complexes. Also, some experiments should be conducted to verify the theoretical claims of section 4.1 and demonstrate the difference in performance between the Hodge Laplacian and the CC Laplacian. 

[7]: Adaptive Gaussian Processes on Graphs via Spectral Graph Wavelets by Opolka and al.

[8]: Graph Classification Gaussian Processes via Spectral Features by Opolka and al.

[9]: Graph Classification Gaussian Processes via Hodgelet Spectral Features by Alain and al. 

[10]: Graph and Simplicial Complex Prediction Gaussian Process via the Hodgelet Representations by Alain and al.

[11]: Transductive Kernels for Gaussian Processes on Graphs by Zhi and al.

[12]: Wavelets on Graphs via Spectral Graph Theory by Hammond and al.

### Questions
The idea of capturing multi-scale information in a descriptor is already present in [7] [8] [9] [10]. How does HKS compare to these existing descriptors from signal processing? Why choose the HKS when it is possible to use a Fourier or wavelet multi-scale descriptor? 

I believe that an alternative to MP-based GNNs is an important avenue to pursue and that this paper is a step in that direction. However, I am concerned about the overall impact of this paper and the fact that it overlooks some related works. I can see that the HKS is (1) applied to a CC Laplacian and (2) given to a neural network pipeline. For the first point, using the CC Laplacian rather than the Hodge Laplacian or graph Laplacian (as in previous work) is straightforward. I am not confident about the impact of considering the highly abstract CCs, since as far as I know it is difficult to find real-world datasets that are CCs (especially that are neither graphs or cellular complexes). For the second point, I am not convinced that integrating these descriptors into a neural network pipeline represents a sufficiently significant contribution. 

Finally, for a fairer comparison, the time required to compute the eigendecomposition should be included somewhere in the experiments section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TopoHKS, a framework for topological deep learning on combinatorial complexes (CCs) using Heat Kernel Signatures (HKS) as node descriptors. The method defines a novel Laplacian operator on CCs, enabling multiscale, permutation-equivariant embeddings that can be processed by standard neural architectures such as Transformers or MLP Mixers. The authors establish theoretical guarantees regarding the expressivity of their approach (up to isospectral equivalence) and present empirical results on molecular property prediction and synthetic topological tasks, showing improvements in runtime and competitive accuracy compared to topological baselines like SMCN and CIN.

### Strengths
* Solid analysis of Laplacian properties, uniqueness, and expressivity. The authors establish meaningful results that clarify the limits and strengths of spectral descriptors on CCs.

* The method avoids higher-order message passing while retaining structural richness, leading to significantly faster inference and training, especially on large complexes.

* Benchmarks show that TopoHKS performs competitively with or better than SMCN and other baselines, particularly in topologically sensitive tasks and with fewer computational resources.

* The model is modular and can use either Transformer or MLP Mixer backbones, offering adaptability to different use cases.

### Weaknesses
* Like other spectral methods, TopoHKS cannot distinguish between complexes that are isospectral but structurally different (non-isomorphic). Although such cases are uncommon in real-world data, this limitation reflects an important theoretical weakness in the model’s expressiveness.

* The eigendecomposition step for computing HKS remains a bottleneck for extremely large-scale CCs. Although the authors mention Nyström approximations, they are not implemented or evaluated.

* The paper focuses primarily on classification tasks. It would be beneficial to see additional tasks (e.g., regression, graph similarity, transfer learning) to further stress-test the representations.

* No ablations or visualisations are provided to understand what the learned HKS descriptors capture. Given the topological motivation, this is a missed opportunity.

* The paper does not include a direct comparison of TopoHKS against standard HKS on graphs, which would help isolate the benefits of moving to CCs.

### Questions
* Can you elaborate on the feasibility of using approximate eigensolvers for large-scale datasets, and whether this affects downstream performance?

* Would combining HKS-based descriptors with message-passing mechanisms help overcome the isospectral limitation? Have you considered hybrid approaches?

* Have you experimented with approximating the Laplacian spectrum using Nyström or other fast eigen-solvers? How does this affect performance and accuracy?

* Could you include visualisations of the HKS descriptors or diffusion patterns across different CCs to better interpret what the model is learning?

* Have you considered tasks beyond classification, such as molecular generation, regression, or transfer learning?

* Could your method generalise to non-Euclidean combinatorial complexes, e.g., with noisy or partially observed incidence structure?

### Soundness
3

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
4

### Summary
While heat kernels have been well-studied on graphs and simplicial complexes, the generalization to combinatorial complexes has not yet been made. In doing so, the authors overcome this more general setting and define a new Laplacian  to characterize graphs based on principle that different combinatorial complexes dissipate heat differently (up to isospectral complexes). The claims and definitions are well worked out and an extensive analysis is provided with respect to the expressivity with respect to this approach. The authors evaluate the method on a set of real-world datasets and show that their method is significantly faster and more expressive at the same time.

### Strengths
Overall the work is well presented and motivated. The full implementation of the experiments and various empirical experiments support the theoretical claims in the paper, which is appreciated by the reviewer. In particular stating the occurrence of the number of iso-spectral graphs / complexes in real-world dataset is important and appreciated.

### Weaknesses
While it is my belief that the work is an interesting contribution to the field, some questions remain for the reviewer after reading the paper. Please see the questions section.

### Questions
- Often meshes form the discrete approximation of continuous (pseudo-)manifolds such as objects or humans. How does the method in these particular cases. It could be argued that tasks where we deal with iso-spectral manifolds are very relevant. 
- As an extension to the previous question is the question how such a method would deal with barycentric subdivision. In particular, if we were to approximate a sphere with two meshes, would that result in the same signature, or would it by definition result in separate signatures. It might be interesting to consider these questions.
- It seems the case that the method only provides global descriptors. Is this correct? If so, would the method be easy to extend to local properties as well, such as node classification?
- How how are the initial conditions chosen. All zeros and one non-zero? 
- What is the interplay between *geometric* and *topological* features? Multiple experiments and papers have already noted that sometimes the nodes in a graph (or combinatorial complex) already poses sufficient (if not all) information for the task (say a point cloud of a human vs its KNN graph). In those cases, ignoring the coordinates could be detrimental for the task.
	- Are features of the combinatorial complex coordinates ignored?
	- If they are ignored, it is good to include a discussion on potential ways to mitigate it.

- An interesting addition (if the method is indeed independent of node features) is the recently proposed MANTRA [1] dataset to support the claim. It is the suspicion of the reviewer that your method could do well here as it contains a set of truly topological tasks. 
- It can be shown that graph neural networks are in essence diffusion operators. If computing the full Laplacian becomes prohibitive, one could always resort to the a forward pass of a (higher order) GNN with fixed unlearned weights. Although the reviewer is not 1oo% sure and it would fall outside the scope of the paper, it is interesting to consider.

[1] MANTRA: The Manifold Triangulations Assemblage](https://arxiv.org/abs/2410.02392)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a framework for topological deep learning that uses Heat Kernel Signature as a node descriptor on combinatorial complexes. It generalises the notion of a graph laplacian operator to Combinatorial complexes and leverages it to compute multiscale, permutation-equivariant HKS embeddings that can be integrated into standard deep learning architectures. The method aims to balance expressivity and scalability, distinguishing complexes up to isospectral equivalence. Empirically, it demonstrates improved computational efficiency and competitive accuracy on some molecular property prediction benchmarks and specialised topological tasks.

### Strengths
[S1] The idea of applying heat kernels to Topological Deep Learning (TDL) is conceptually interesting and has not been explicitly explored in prior work.

[S2] The proposed method is simple and clear (despite some of the writing being unclear).

### Weaknesses
[W1] Clarity and presentation – The paper suffers from numerous unclear passages, typographical errors, and missing explanations. Several central definitions and notations are not clearly introduced, which makes it difficult to follow the theoretical and experimental sections. (See Questions below for examples.)

[W2] Theoretical soundness – Many of the theoretical results are either trivial, misleadingly framed, or possibly incorrect. (see Questions below for details)

[W3] Experimental limitations – The empirical evaluation is weak. The datasets are small, and the comparisons are incomplete. This makes it hard to assess the actual benefit of the proposed approach.

Overall, while the idea of leveraging heat kernels in TDL could be promising, the current version falls short in rigor, clarity, and experimental validation.

### Questions
[Q1] Under “Well-defined hierarchical structure”, The authors claim that defining a Laplacian for a combinatorial complex is challenging because such complexes “do not enforce strict rank stratification.” However, each cell in a complex has a well-defined rank. Could the authors clarify what structural difficulty they refer to?

[Q2] Line 130 states “we aim to design a Laplace operator … conserve mass and locality”: This statement is unclear. Please define these terms in the context of your Laplacian and clarify the motivation for this requirement.

[Q3] Corollary 4.1: A corollary should follow from a previously stated result. As written, this reads more like an independent proposition; consider renaming or re-framing it.

[Q4] Definition of “isospectral”: The term is used extensively but never defined. If it means “having the same set of Laplacian eigenvalues,” then Theorem 4.1 appears trivial—permuting nodes does not change the Laplacian spectrum. The theorem should either be removed or reframed to clarify its contribution.

[Q5] Lemma 4.1: The claim that the proposed Laplacian is “more expressive” than the Hodge Laplacian seems misleading. The Hodge Laplacian is defined across ranks, however, in the lemma you seem to be limiting yourself to use only the first Hodge laplacian, in which case the result is trivial.

[Q6] Equation (7): The quantity $w_{i,j}$​ is never defined. The description “zero iff there is no cell between i and j” is insufficient—what does $w_{i,j}$ represent, and why does it serve as a meaningful weight for smoothness? Please clarify the motivation behind this formulation. Additionally, please clarify why the connection between the laplacian and smoothness is relevant to the paper.

[Q7] Corollary 4.3: The statement essentially says that an MLP receiving unique Laplacian spectra can distinguish between complexes, which follows trivially from injectivity assumptions. Please re-examine whether this constitutes a meaningful theoretical result.

[Q8] Table 1: The claim that SMCN cannot distinguish between all non-isospectral complexes contradicts prior findings that subgraph-based GNNs exceed spectral methods in expressivity [1]. Either provide a proof or remove this statement.

[Q9] Scalability section: Computing heat kernels typically requires eigenvalue decomposition, which is cubic in complexity. If your implementation avoids this, please explain how.

[Q10] Expressivity experiment (torus complexes): The failure case shown could likely be resolved by adding rank-4 updates to SMCN. Please ensure that the baselines are fairly configured.

[Q11] Real-world benchmarks: The evaluation should include more established molecular benchmarks ideally those used in [2] as this is the most direct baseline. Also, the CIN results reported as “NaN” due to training failure should instead be presented as the resulting (poor) accuracy for transparency.

[Q12] The font seems off. The section headings appear unusually stretched, can the authors explain?


[1] Zhang et al. On the Expressive Power of Spectral Invariant Graph Neural Networks. 2024.

[2] Eitan et al. Topological blindspots: Understanding and extending topological deep learning through the lens of expressivity. 2024.

### Soundness
1

### Presentation
1

### Contribution
2
