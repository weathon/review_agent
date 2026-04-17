# GraphCPD: Coherent Point Drift for Point Cloud Registration via Graph Signal Processing

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Probabilistic point cloud registration has attracted increasing attention due to its robustness to noise, outliers and occlusions. However, existing methods often suffer from high computational cost and neglect the role of informative priors. In this paper, we propose a new probabilistic registration method based on graph signal processing (GSP), called graph coherent point drift (GraphCPD). Specifically, we use a high-pass graph filter to extract high-frequency components, which are theoretically proven to be invariant under rigid transformations. These components are combined with point coordinates and normals to form a high-dimensional graph signal. We construct a local graph based on the graph signal and use the graph Laplacian model for registration. Compared with the classical Gaussian mixture models (GMMs), graph Laplacian models provide more discriminative geometric representations and enhances the model’s ability to capture graph structure.  Furthermore, we exploit the invariance of high-frequency components to define prior probabilities, significantly reducing the corresponding search space and improving the speed of registration. Experimental results demonstrate that our method improves runtime efficiency over most existing probabilistic methods, while maintaining competitive registration accuracy, especially on large-scale point clouds. The source code is available at https://anonymous.4open.science/r/GraphCPD-801E.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GraphCPD, a probabilistic point cloud registration method integrating Graph Signal Processing (GSP) with GMM. The method uses high-pass graph filters to extract transformation-invariant high-frequency components, replaces GMM's isotropic covariance with point-specific graph Laplacian matrices, and constructs 9-D graph signals from coordinates, normals, and high-frequency components.

### Strengths
1. Novel integration of GSP into probabilistic registration with well-motivated use of local Laplacians for capturing geometry.

2. Theorem 1 proofs the transformation invariance of high-frequency components.

3. Promising accuracy/efficiency trade-offs on some benchmarks (Table 1, Utah Teapot).

### Weaknesses
1. Experimental scope is narrow, using limited datasets (Stanford scans, Utah Teapot, KITTI) without broader evaluation on other standard benchmarks.

2. Missing critical baselines. CPD is absent from Tables 1-3 (only in Table 4), weakening comparisons to CPD-style methods. LSG‑CPD is omitted from Sec. 4.3 (multi‑view) and Sec. 4.4 (KITTI) despite using similar experimental configurations.

3. The $k_{match}$ claim lacks substantiation. The paper attributes lower accuracy vs. LSG-CPD to "small $k_{match}$" but doesn't state the value in main text (Maybe $k_{match}=100$?). Providing AngErr and Time results with different $k_{match}$ values (such as 100, 200, 500) in Table 1 would be helpful.


4. No ablation studies to validate individual components (such as Laplacian $\|\cdot\|_{\mathcal{L}_m}$. vs. isotropic).

### Questions
- Why LSG-CPD spends so much time in Table 1? Since it claims 89ms for 3500 points in their paper.

- Why the results of LSG-CPD are absent in Table 3 and Table 4?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces GraphCPD, a probabilistic point-set registration framework that integrates ideas from graph signal processing (GSP). The authors construct a local high-frequency descriptor by applying a graph Laplacian filter to point coordinates and normals, claiming that these descriptors are rigid-motion invariant and can guide correspondence estimation. The method replaces the isotropic Gaussian covariance in classical CPD with a local Laplacian-based metric and uses high-frequency responses to constrain candidate matches. Experimental results on several classical rigid-registration datasets show improved accuracy over traditional CPD variants.

### Strengths
- Mathematically sound integration of GSP principles into a probabilistic registration framework (CPD). 
- The method is interpretable and relatively general, it could potentially be extended to non-rigid or semantic alignment.

### Weaknesses
- Limited novelty. The use of graph-Laplacian high-frequency responses as local geometric descriptors is conceptually not novel, having been explored in prior graph/spectral registration work (e.g., GraphReg and classical spectral descriptors). The paper mainly repackages a hand-crafted high-pass descriptor within a standard CPD framework, which is an incremental recombination rather than a new principle.

- Missing comparisons with classical spectral descriptors (HKS/WKS). The proposed high-pass descriptor is positioned as a GSP-based local feature, yet the paper does not compare against Heat Kernel Signature (HKS) or Wave Kernel Signature (WKS), canonical spectral descriptors derived from the Laplace(-Beltrami) operator that can be viewed as graph-spectral filters on discrete meshes/point clouds. These descriptors are rigid-motion invariant (and WKS is relatively more scale-stable), and thus constitute strong, conceptually proximate baselines. It's unclear whether the proposed high-pass design is more distinctive or robust than standard spectral alternatives.

- Questionable discriminative power of the descriptor. Although the authors claim the high-pass filtering highlights salient local geometry, no visualization or empirical analysis (e.g., t-SNE embedding of descriptors across corresponding/non-corresponding points) is provided to verify its discriminative ability. Without such evidence, it is hard to judge whether the descriptor truly helps correspondence estimation or merely serves as a heuristic prior.

- Outdated baseline selection and limited comparison. The evaluation mainly compares with older non-learning methods (e.g., CPD, ECMPR, FilterReg), while no comparison to modern deep-learning descriptors (such as Predator and GeoTransformer) is provided. This limits the relevance of the reported improvements in the current research landscape.

- Efficiency concerns. The claimed “fast” method still requires over 4 seconds even for the simple Utah Teapot model, which is quite high for small-scale rigid alignment. Given the algorithm’s largely analytical nature and lack of GPU acceleration, the practicality and scalability are questionable.

### Questions
- Have you compared the proposed high-frequency descriptors with diffusion-based ones (e.g., HKS, WKS) in terms of invariance and discriminability?
- Could you provide a feature-space visualization (e.g., t-SNE) showing correspondence similarity for your descriptors?
- Would integrating a learned feature extractor outperform the hand-crafted descriptor? The authors should compare more learning-based deep geometric descriptors, such as Predator and GeoTransformer, to confirm the SOTA performance.

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
The paper proposes a new 3D point cloud registration method, leveraging some concepts from graph signal processing (GSP). In particular, for each 3D point, a local graph Laplacian is constructed, where the edge weights are computed as an exponential function of the differences in 3D locations and estimated surface normals (equation 3). The sample of a graph signal is defined by stacking the point coordinate with the surface normals, and high-frequency components (equation 2). High frequencies are computed using simple polynomials of the normalized graph Laplacian matrix $\mathcal{L}$ interpreted as the graph shift operator (GSO). The rigid transformation is estimated via an EM algorithm.

### Strengths
Leveraging GSP concepts in the 3D point cloud registration problem appears to be new.

### Weaknesses
1. The writeup is not that clearly described with terminologies that are not easy to understand. For example, what is meant by "stronger geometric significance" (pg.4)? Graph signals $\mathbf{s}_n$ and $\mathbf{s}_m$ discussed in page 3 are not defined till page 4. What is meant by the subscript $\mathbf{L}_m$ in equation (1)? If the authors mean $\ell_2$-norm, like $(\mathbf{y} - \mathbf{x}^\top \mathbf{L}_m (\mathbf{y} - \mathbf{x})$, then the fidelity term is pretty standard in signal restoration and not novel. The employed GSP terminology is not consistent with the GSP literature. A GRAPH SIGNAL $\mathbf{x} \in \mathbb{R}^N$ is typically defined as a $N$-dimensional signal, one scalar-valued sampled $x_i$ for each node $i$. Hence, $\mathbf{s}_n$ is NOT a graph signal, but a vector-valued sample at node $n$. It is not clear why $\epsilon$ is needed for "numerical stability" to define $\tilde{\mathbf{L}}_m$ (pg. 3). It is not clear why the particular "high-pass" filters in equation (4) are used, given none of them are ideal or approximately ideal high-pass filters. They seem to be defined in a ad-hoc manner. 

2. Whether high-frequency components highlight local variations depend entirely on how the edge weight are defined. If the edge weights are defined, like equation (3), where expected signal variations across nodes are ALREADY encoded as small positive edge weights, then high-frequency components DO NOT highlight local variations. High frequencies actually are components that are CONTRARY to encoded pairwise similarities in the edge weights, for example, big variations across large edge weights. Big local pairwise differences across a very small weight edge (or a negative edge) does not constitute high frequencies. There is a misunderstanding here. 

3. Graph Laplacian matrix is conventionally interpreted as the INVERSE of the covariance matrix also known as the precision matrix. For example, given an empirical covariance matrix $\mathbf{C}$, the sparse inverse covariance matrix $\mathbf{P}$ of a Gaussian Markov Random Field (GMRF) is often computed using graphical lasso (GLASSO) or its variants. In this proposal, however, the locally constructed Laplacian is replacing the covariance matrix in a Gaussian Mixture Model (GMM). This is a mismatch. 

4. The definition of a sample of a graph signal (rather than the graph signal as written) as vector quantity in equation (2) is highly unusual, because the attributes defined in this vector-valued sample $\mathbf{s}_m$ are fundamentally different quantities in different scales. (Maybe this is why the surface normal is "scaled" in a strange manner? Not clear from the text.) So it is highly unlikely that the SAME Laplacian matrix can describe pairwise similarities for all the attributes in the vector-valued samples. 

5. The experimental results in Table 1 etc are not dramatically better than previous works. 

6. There are no Appendices attached to the manuscript, despite numerous mentions in the paper.

### Questions
1. Why is the particular edge weight definition in equation (3) employed? In Dinesh et al. 2022, the defined edge weights were defined as such for 3D point cloud geometry restoration (denoising, etc). An edge weight tends to zero when two points are far in Euclidean distance (and thus bears no similarity) OR when two points have orthogonal surface normals (e.g., different sides of a table, and thus one point cannot help the other in denoising). It's not clear why the definition is reused here for registration. 

2. Why are the high-pass filters so defined in equation (4), given none of them are ideal high-pass filters? One can approximate an ideal high-pass filter, with a target cutoff frequency, using Chebyshev or Lanczos approximation with polynomials of a graph shift operator (GSO). Why is this not done?

3. Why is the invariance of high-pass components even important in this scenario?

4. Where are the Appendices?

### Soundness
2

### Presentation
2

### Contribution
2
