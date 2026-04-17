# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication

- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Graph alignment, the problem of identifying corresponding nodes across multiple graphs, is fundamental to numerous applications. Most existing unsupervised methods embed node features into latent representations to enable cross-graph comparison without ground-truth correspondences. However, these methods suffer from two critical limitations: the degradation of node distinctiveness due to oversmoothing in GNN-based embeddings, and the misalignment of latent spaces across graphs caused by structural noise, feature heterogeneity, and training instability, ultimately leading to unreliable node correspondences. We propose a novel graph alignment framework that simultaneously enhances node distinctiveness and enforces geometric consistency across latent spaces. Our approach introduces a dual-pass encoder that combines low-pass and high-pass spectral filters to generate embeddings that are both structure-aware and highly discriminative. To address latent space misalignment, we incorporate a geometry-aware functional map module that learns bijective and isometric transformations between graph embeddings, ensuring consistent geometric relationships across different representations. Extensive experiments on graph benchmarks demonstrate that our method consistently outperforms existing unsupervised alignment baselines, exhibiting superior robustness to structural inconsistencies and challenging alignment scenarios. Additionally, comprehensive evaluation on vision-language benchmarks using diverse pretrained models shows that our framework effectively generalizes beyond graph domains, enabling unsupervised alignment of vision and language representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles unsupervised graph alignment and proposes Graph Alignment via Dual-pass encoder and Latent-space communication (GADL), which has two key parts: (i) a dual-pass spectral encoder that concatenates low-pass and high-pass GCN branches to balance structural smoothness and node distinctiveness; and (ii) a geometry-aware functional map module that aligns the two latent spaces. Experiments on semi-synthetic robustness benchmarks show higher or competitive alignment accuracy of GADL. On the real-world datasets, GADL surpasses baselines by a large margin. Moreover, GADL shows strong cross-modal generalization on vision–language alignment.

### Strengths
1. The paper explicitly identifies two crucial weaknesses of existing graph alignment methods, including oversmoothing and latent-space misalignment, and addresses them with a dual-pass spectral encoder and bijective/isometric functional maps, respectively.

2. GADL has impressive results, which consistently outperform baselines across different datasets, including synthetic data, real-world data, and multi-modal data.

### Weaknesses
1. The graph alignment method is sensitive to structural perturbations. The experiment on semi-synthetic benchmarks shows that only 5\%  structural perturbations will lead to a huge performance degeneration. As a result, we may doubt whether existing graph alignment methods can be used in real-world applications.

2. The ablation studies in Table 6 show that the proposed two regularizations, i.e., bijectivity and orthogonality, have almost no influence on model performance. Therefore, the design of GADL involves redundant components, which limits the novelty of this paper and makes its motivation inconvinced.

3. The experiments are conducted on small-scale graphs, which only contain thousands of nodes. It would be better if the authors could add experiments on large-scale datasets to validate the scalability of the proposed method.

4. The combination of low-pass and high-pass filters is explored by the previous method [1]. However, this paper does not discuss it.

[1] Beyond Low-frequency Information in Graph Convolutional Networks.

### Questions
See weaknesses.

### Soundness
2

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
The proposed work introduces GADL, an entirely new unsupervised graph alignment solution to overcome the challenges in previous works: the reduction in node discriminativity associated with GNN oversmoothing and mismatched latent spaces across graphs. The authors designed a two-pass GCN encoder comprising low-pass spectral filtering and high-pass spectral filtering to improve node discriminativity while preserving graph topology information. The work further involves the adoption of the geometry-aware functional map module to align latent spaces in an unsupervised manner. The proposed solution outperforms state-of-the-art baselines on graph and vision-language tasks.

### Strengths
1. The dual-pass encoder with low-pass and high-pass spectral filters effectively addresses node distinctiveness degradation from GNN oversmoothing.

2. Comprehensive experiments demonstrate consistent superiority over state-of-the-art baselines across eight benchmarks, particularly under structural perturbations.

3. The method successfully generalizes beyond graph domains to vision-language alignment tasks.

### Weaknesses
1. The idea of combining GNNs and low-pass/high-pass spectral filtering is not new. Chien et al. [1] proposed the GPR-GNN architecture, employing monomial basis functions to flexibly learn low-pass or high-pass filters. From an optimization perspective, He et al. [2] introduced BernNet, which approximates arbitrary spectral filters using Bernstein polynomials. Duan et al. [3] proposed Spectral GNNs via Triple Filter Ensembles. And there are more.

[1] E. Chien, J. Peng, L. Pan, O. Milenkovic, Adaptive universal generalized pagerank graph neural network. arXiv preprint arXiv:2006.07988, (2020).
[2] H. Mingguo, Z. Wei, X. Hongteng, et al., Bernnet: learning arbitrary graph spectral filters via bernstein approximation, Adv. Neural Inf. Process. Syst. 34 (2021) 14239–14251.
[3] Unifying homophily and heterophily for spectral graph neural networks via triple filter ensembles." Advances in Neural Information Processing Systems 37 (2024): 93540-93567.

2. Given that GNNs with low-pass/high-pass filtering have been intensively studied, what is the unique research challenge when applying this technique to graph alignment?

3. There are several issues in the experiments. First, excessive hyperparameters with no principled guidance for parameter selection on new problems. Second, Table 1 shows GAE accuracy drops from 86.3% to 6.5% under perturbations while T-GAE remains stable, yet the paper provides no analysis of this phenomenon. Third, a lack of computational complexity analysis and runtime comparisons.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel unsupervised graph alignment framework named GADL. Its core contributions are: 1) designing a dual-pass spectral encoder that combines low-pass and high-pass filters to effectively address the degradation of node distinctiveness caused by GNNs; and 2) introducing functional maps with geometric constraints to enforce the alignment of embedding spaces across different graphs, tackling the challenge of latent space inconsistency. Experiments demonstrate that the method achieves sota performance in terms of accuracy, robustness, and cross-domain generalization. While the method is powerful, I think its theoretical underpinnings rely on a linear alignment assumption for the latent spaces and could be susceptible to the instability of the graph Laplacian eigenbasis. These aspects represent theoretical boundaries worth further exploration.

### Strengths
1. The authors clearly articulate the two core challenges faced by current mainstream methods. The proposed solutions are well-targeted and logically sound.
2. The methodology is novel and well-designed:
   - The dual-pass encoder, by combining low-pass and high-pass spectral filters, is a clever and effective improvement to the GNN architecture for simultaneously preserving structural smoothness and node distinctiveness.
   - The Latent Space Communication module introduces the "Functional Map" concept to graph alignment. By enforcing bijectivity and orthogonality constraints, it compels geometric consistency across graph embedding spaces, offering a novel perspective for resolving the spatial misalignment problem.
3. The paper provides a comprehensive evaluation across semi-synthetic, real-world, and cross-domain tasks. The results demonstrate that GADL not only achieves sota performance on various metrics but also shows excellent robustness in noisy environments. Its success on the vision-language alignment task further validates the framework's generalization potential.

### Weaknesses
Some points for Improvement:

I find this work to be excellent. However, there are a few theoretical assumptions and limitations worth discussing. The following points stem from my own analysis:

1. The framework relies on the strong assumption of a linear relationship between latent spaces. Ideally, if the ground-truth node permutation is Π, we would expect Z_t ≈ Π Z_s R, where R is an orthogonal matrix. The functional map aims to learn C by minimizing ||C F_s - F_t||_F^2, where F_s = U_s^T Z_s and F_t = U_t^T Z_t are the spectral projections.
   However, what if the true relationship between the spaces is non-linear, e.g., Z_t ≈ Π g(Z_s), where g(·) is a non-linear function? In this case, the model attempts to fit a non-linear relationship with a linear operator C. The optimal linear solution, C_opt = F_t F_s^+ (where + denotes the pseudo-inverse), might result in a large residual ||C_opt F_s - F_t||^2. Does this imply that the linear alignment may be unable to perfectly capture non-linear geometric distortions between the spaces?
2. The method's reliance on the Laplacian eigenbasis can be a concern when the graph Laplacian has degenerate (repeated) eigenvalues, which is common in symmetric graphs. In such cases, the corresponding eigenvector basis is not unique, and minor graph perturbations can lead to drastic changes in the basis.
   For instance, let L have a degenerate eigenvalue λ_i = λ_{i+1} with corresponding eigenvectors u_i, u_{i+1}. Then, for any orthogonal matrix Q ∈ R^{2x2}, [u'_i, u'_{i+1}] = [u_i, u_{i+1}] Q is also a valid basis for that eigenspace. A small perturbation ΔG leads to L' = L + ΔL, which can break this degeneracy. According to matrix perturbation theory, even if ||ΔL|| is small, the new eigenvectors u'_i, u'_{i+1} can be drastically different from (nearly orthogonal to) the original basis. Since the spectral features F = U^T Z directly depend on U, an unstable spectral basis could lead to unstable spectral features, thereby affecting the learning of the functional map C.
   I would encourage the authors to discuss this potential issue. For example, they could analyze why this has a limited impact on real-world networks (perhaps due to their low symmetry) or argue that the framework's strong performance itself is evidence of its robustness to such spectral basis perturbations.

### Questions
See the discussion of weakness

### Soundness
3

### Presentation
4

### Contribution
4
