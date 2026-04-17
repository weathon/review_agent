# Refining Dual Spectral Sparsity in Transformed Tensor Singular Values

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
The Tensor Nuclear Norm (TNN), derived from the tensor Singular Value Decomposition, is a central low-rank modeling tool that enforces *element-wise* sparsity on frequency-domain singular values and has been widely used in multi-way data recovery for machine learning and computer vision. However, as a direct extension of the matrix nuclear norm, it inherits the assumption of *single-level spectral sparsity*, which strictly limits its ability to capture the *multi-level spectral structures* inherent in real-world data, particularly the coexistence of low-rankness within and sparsity across frequency components. To address this, we propose the tensor $\ell_p$-Schatten-$q$ quasi-norm ($p, q \in (0,1]$), a new metric that enables *dual spectral sparsity control* by jointly regularizing both types of structure. While this formulation generalizes TNN and unifies existing methods such as the tensor Schatten-$p$ norm and tensor average rank, it differs fundamentally in modeling principle by coupling global frequency sparsity with local spectral low-rankness. This coupling introduces significant theoretical and algorithmic challenges.  To tackle these challenges, we provide a theoretical characterization by establishing the first minimax error bounds under dual spectral sparsity, and an algorithmic solution by designing an efficient reweighted optimization scheme tailored to the resulting nonconvex structure. Numerical experiments demonstrate the effectiveness of our method in modeling complex multi-way data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the tensor $\ell_p$-Schatten-$q$ quasi-norm ($p$,$q$ in $(0,1]$) as a regularization for low-rank tensor modeling within the t-SVD paradigm. As claimed by the authors, this regularizer enables dual spectral sparsity control: parameter p promotes sparsity across frequency components in the transformed domain, while q enforces low-rankness within each frequency slice. Experiments on the noisy tensor completion task have been conducted to demonstrate the effectiveness of the proposed regularizer.

### Strengths
1. The proposed regularizer is well-motivated.

2. Theoretical approximation properties under the noisy scenario have been established.

### Weaknesses
1. The theoretical results and the empirical studies are indeed not consistent. Specifically, the main results, i.e., Theorem 4.2, are about the approximation properties under certain constraints, while the experiments are about the tensor completion task. The authors did not provide recovery guarantees for the tensor completion task, as has been done in many existing studies.

2. From the algorithmic perspective, it seems that there are no results about the convergence, either theoretically or empirically. In addition, why choose reweighted $\ell_{1/2}$, since it seems that reweighted $\ell_1$, which has a simpler proximal operator, can be applied. Besides, numerical solution [1] for $\ell_q$ type regularizers may also be implemented.
 
    [1] Goran Marjanovic and Victor Solo. On $\ell_q$ Optimization and Matrix Completion. IEEE Transactions on Signal Processing, 60(11): 5714-5724, 2012.

3. The experiments are insufficient. Specifically, only 6 tensors (or datasets as referred to by the authors) have been adopted. As is known, there are 31 MSIs in the Columbia MSI datasets. In addition, synthetic data experiments could also be useful as verifications for the theoretical results.

### Questions
First, the authors should address the issues mentioned in the "Weaknesses". Then, there are additional questions that should be addressed:

1. It is better to discuss whether the proposed regularization term can be extended to higher-order tensors rather than 3.

2. It seems that the proposed regularizer is related to the group sparsity regularizers for vectors. If it is true, please discuss.

3. It can be seen that different (p,q) values can affect the performance. It is better to provide practical guides for choosing their values, for example, showing that the performance is stable within certain ranges of this pair of values.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a tensor ℓp–Schatten–q quasi-norm framework to capture both inter-frequency sparsity and intra-frequency low-rankness within the t-SVD framework. The method generalizes the tensor nuclear norm (TNN) by allowing separate control over different spectral structures. The authors provide theoretical characterization through minimax bounds, propose a proximal reweighted ℓ1/2 optimization algorithm, and conduct experiments on multiple remote sensing datasets showing consistent improvements in PSNR and SSIM.

### Strengths
- The paper provides a clear motivation by identifying the limitation of TNN and establishing a theoretically grounded dual spectral sparsity framework.
- The theoretical analysis is rigorous, and the derivation of minimax bounds under hard and soft sparsity regimes adds valuable insight into tensor estimation.
- The experiments in the appendix are extensive and demonstrate consistent improvements over strong baselines with reasonable robustness across parameter settings.
- The manuscript is well-written and logically organized, with clear figures and notation.

### Weaknesses
- The structure of the paper is imbalanced. One can consider put more empirical results back in the main text.
- The theoretical results are strong but could benefit from more intuition about their practical meaning and implications.
- The experimental evaluation mainly focuses on imaging data; broader validation on non-visual tensors could strengthen the claim of generality.
- The convergence behavior of the proposed reweighted optimization algorithm is discussed empirically but lacks theoretical guarantees.
- A clearer comparison of computational cost with other tensor regularization methods would help evaluate its practical efficiency.

### Questions
- How sensitive is the performance to the choice of the transform $M(\cdot)$? Would learning or adapting $M$ from data bring further gains?
- In the theoretical analysis, what assumptions are crucial for the minimax lower and upper bounds to match? 
- Can the proposed framework be extended efficiently to tensors of order higher than three, and what challenges might arise in doing so?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the tensor ℓp-Schatten-q quasi-norm as a generalization of the TNN to jointly model intra-frequency low-rankness and inter-frequency sparsity within the t-SVD framework. Conceptually, it addresses a genuine limitation of standard TNN—the inability to capture multi-level spectral structures—by proposing dual spectral regularization. Theoretical guarantees (minimax bounds) and a reweighted nonconvex optimization algorithm are provided, with empirical improvements on several hyperspectral datasets

### Strengths
The proposed formulation is mathematically sound and unifies prior norms (TNN, Schatten-p, average rank).

Comprehensive experiments demonstrate consistent PSNR/SSIM gains.

### Weaknesses
The core assumption of this paper—that dual spectral sparsity patterns exist in the transformed (DCT) domain under the t-SVD framework—is insufficiently supported. The only evidence provided is an empirical illustration using MRI and CT examples in Figure 1, which is too limited to convincingly justify such a fundamental modeling assumption. A more systematic analysis or statistical validation would be needed to establish its generality.

Although the authors acknowledge that the proposed ℓₚ-Schatten-q quasi-norm introduces significant computational challenges, no comparison of algorithmic complexity or runtime with state-of-the-art methods is presented. This omission leaves readers uncertain about the practical feasibility of the approach.

The theoretical analysis, while ambitious, remains largely abstract. The proofs are not clearly connected to practical convergence guarantees or algorithmic stability, reducing their applied value.

Experimental evaluation is confined to tensor completion tasks. The absence of experiments on broader problems such as denoising, inpainting, or clustering limits the generalizability of the proposed framework. Moreover, the sensitivity analysis for the nonconvex parameters (p, q) is minimal—only two parameter settings are tested—which is insufficient to demonstrate robustness.

Finally, the presentation is mathematically dense and occasionally repetitive, which affects readability and clarity. A more structured exposition with stronger empirical validation would substantially improve the paper’s impact and credibility.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2
