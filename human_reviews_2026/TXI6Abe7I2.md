# Searching for the Best Polynomial Approximation for the Accurate Log Matrix Normalization in Global Covariance Pooling

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Global Covariance Pooling (GCP) has significantly improved Deep Convolutional Neural Networks (DCNNs) by leveraging richer second-order statistics. However, since covariance matrices lie on the Symmetric Positive Definite (SPD) domain, normalization is required to map them back into the Euclidean domain. The mathematically accurate approach, Matrix Log Normalization (MLN), suffers from gradient instabilities and requires eigendecomposition (EIG) or singular value decomposition (SVD), both of which are GPU-unfriendly. To address these instabilities, Matrix Power Normalization (MPN) introduced square-root normalization. Since then, most works have focused on approximating the matrix square root, typically via Newton–Schulz iterations or polynomial (Taylor and Padé) expansions, as these are GPU-friendly. Yet no prior work has attempted to approximate the more accurate MLN using polynomials, despite their inherent GPU efficiency. In this work, we explore a broad range of polynomial families—especially orthogonal polynomials (Taylor, Chebyshev, Legendre, Laguerre, Padé)—for approximating MLN, and conclude that Chebyshev polynomials offer the most accurate and efficient approximation. Experiments on large-scale visual recognition benchmarks demonstrate that our approach achieves competitive accuracy while substantially reducing training cost. For reproducibility, the code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper revisits Matrix Log Normalization (MLN) for global covariance pooling and proposes to approximate the matrix logarithm using polynomial families to avoid SVD/EIG, aiming for GPU-friendly training with more stable gradients. Several approximation schemes (Taylor, Chebyshev, Legendre, Laguerre, Padé) are evaluated, with Chebyshev emerging as the best trade-off.

### Strengths
The goal is clear: make MLN practical without expensive eigendecomposition. The use of polynomial matrix approximations is technically sound and leads to a fully GEMM-based implementation compatible with GPUs. Experiments show improved accuracy and efficiency over classical MLN/MPN and iSQRT-COV on fine-grained image classification.

### Weaknesses
The paper lacks a clear and structured motivation. From the title, abstract, and introduction, the narrative moves too quickly into related work and technical details, without establishing a compelling high-level story. A reader unfamiliar with covariance pooling or MLN will not gain sufficient intuition about why MLN matters and what fundamental problem is being solved.

Please consider:
• Briefly motivate global covariance pooling and the role of MLN,
• Explain why existing MLN implementations fail (cost, instability),
• State your contribution clearly,
before diving into related work and detailed approximations.

The claim on line 84 that small eigenvalues lead to unstable gradients in MLN needs a reference (this relates to ill-conditioning of the log on small eigenvalues and to backprop through SVD). Please cite prior work showing the instability of the matrix-log backprop or give a short explanation. Please see (W. Wang, Z. Dang, Y. Hu, P. Fua, and M. Salzmann, “Robust Differentiable SVD,” IEEE Trans Pattern Anal Mach Intell).

Line 75 states that the Newton–Schulz iteration “only requires matrix multiplications (GEMM), making it GPU-friendly”, but ignores the computational graph and backprop cost. Forward GEMM-only structure is not enough to guarantee GPU efficiency during training; backprop through repeated matrix multiplications may still introduce overhead or numerical issues. A brief discussion of the autograd graph and memory footprint would strengthen the argument.

The contribution feels incremental. Polynomial approximations of matrix functions are classical, and the paper largely benchmarks known approximation families for this specific layer. The broader conceptual significance for the community is not clearly articulated.

### Questions
-Can you provide a clearer motivation and a more substantial introduction to contextualize the problem and why it matters?

- Can you describe the computational graph in more detail and explain the implications of your approach for backpropagation (e.g., complexity, stability, differentiability)?

- Can you provide intuition for when and why each polynomial family behaves differently within your method?

- Can you strengthen the overall narrative to highlight the broader relevance of your approach beyond the specific covariance pooling module, and clarify what general insights the community should take away?

### Soundness
2

### Presentation
1

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
This paper explores various polynomial families (including Taylor, Chebyshev, Legendre, Laguerre, and Padé) to solve computational bottlenecks and gradient issues in the GCP leveraging GPU-friendly matrix multiplication. Some experimental results show that Chebyshev polynomials are the most computationally efficient, while also improving model performance.

### Strengths
1. The GCP is widely adopted in modern machine learning models.
2. The approach of using iterative polynomial approximations to avoid matrix SVD sounds highly convincing.
3. Some experiments demonstrate the efficacy of the method.

### Weaknesses
1. Lack of comparison with accurate SVD: Although the paper states that it accelerates previous GCP calculations, a comparison between the proposed method and a model using accurate SVD would further strengthen the paper.
2. Missing ablation study on the polynomial expansion degree: The paper fixes the expansion at the 3rd-degree term for all experiments, and lacks the discussion on how model performance and efficiency are affected by using a lower or higher degree.
3. Lack of evaluation experiments on more diverse tasks: The experiments in the paper are limited to image classification tasks. To my knowledge, GCP is also widely applied in image style transfer tasks. Adding experiments on this task would make the paper more solid.
4. The paper omits an introduction to the basic knowledge of the field, which poses difficulties for non-field readers in understanding this paper.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In theory, this paper mainly proposed the efficient method to compute the approximation of Matrix Log Normalization (MLN). In experiments, this paper applied this method to Global Covariance Pooling and achieves competitive accuracy while substantially reducing training cost.

### Strengths
- This paper is well organized and easy to follow.

- The theoretical conduction is sufficient.

### Weaknesses
- (Major) Some important concepts are not presented in details. For example, what is Global Covariance Pooling? Does it compute like Batch Normalizations or Convolution Layers? In this paper I did not see how it works in the networks. 
- The ideas that applying polynomial approximations of $\log A$ is not novel. There has been polynomial methods to approximate other matrix calculations like $A^{-1/2}$ [1] and apply it in the networks as normalization layers. 
- (Major) The experiment settings are not specific. One of the most important settings is not mentioned---the iteration times or the required terms. This decides the balance between approximation errors and computation efficiency. Too many iteration times may result to numerical instabilities [1]. So these discussions are required. 

[1] Huang L, Zhou Y, Zhu F, et al.  Iterative normalization: Beyond standardization towards efficient  whitening[C]//Proceedings of the IEEE/CVF conference on computer vision  and pattern recognition. 2019: 4874-4883.

### Questions
- How does Global Covariance Pooling work? (Weakness 1) 
- What is the recommended iteration times of the methods in this paper? What is the corresponding hyperparameters of the other comparing methods?  (Weakness 2)
- Why use $\log A$ rather than other calculations like $A^{-1/2}$? In this paper I did not see the importance of MLN so I am not sure how important this topic is. The log calculation is not widely used as normalizations. If we apply the Iterative Normalization [1] in the experiments, what are the results? 

[1] Huang L, Zhou Y, Zhu F, et al.  Iterative normalization: Beyond standardization towards efficient  whitening[C]//Proceedings of the IEEE/CVF conference on computer vision  and pattern recognition. 2019: 4874-4883.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper revisits global covariance pooling (GCP) by replacing the conventional SVD-based matrix logarithm normalization (MLN) with a family of polynomial and rational approximations to avoid eigen-decomposition. Five types of expansions (Taylor, Chebyshev, Legendre, Laguerre, and Padé) are formulated with explicit forward and backward propagation equations. Experiments on fine-grained classification datasets and ImageNet-1k show that Chebyshev tends to be more promising

### Strengths
• Presents a unified algebraic framework for multiple polynomial and rational approximations of matrix functions (Sec. 3).
• Implementation avoids EIG/SVD and only requires standard matrix multiplications, making it efficient and GPU-friendly (Alg. 2–6).
• Demonstrates moderate empirical improvement over the existing MLN on CUB-200-2011 and ImageNet-1k (Table 2–4).

### Weaknesses
#### Theoretical Level  
- The backward derivations in Sec. 3.2–3.4 are standard matrix differentials that PyTorch autograd can already compute. The author did not validate whether their manual backward implementation offers any improvement in runtime, stability, or memory usage.  
- The paper does not quantify the numerical error between the approximated log(A) and the true matrix logarithm. It is not clear how far the results are from the matrix log. A quantitative error analysis could offer more insights.  
- It remains unclear whether the improvement is due to the log function itself or simply the approximation scheme.  As 1/2 can also be approximated by such schemes.

#### Experimental Level  
- Experimental coverage is incomplete. Recent baselines such as 
  - *Revitalizing SVD for Global Covariance Pooling: Halley’s Method to Overcome Over-Flattening*  
  - *Learning Partial Correlation based Deep Visual Representation for Image Classification*  
- Transformer-based backbones are not tested, which limits generalization beyond CNNs.  
- Table 2 is unclear. The iteration steps and polynomial order n for each method are not reported, making it unclear whether the efficiency gain comes from the algorithm design or the lower orders.  
- Fitting time on real datasets such as ImageNet is not reported.
- Ablation studies are missing, such as the polynomial order

#### Other  
- Table 3 shows formatting and layout issues.

### Questions
see wk

### Soundness
2

### Presentation
2

### Contribution
3
