# LaplacianFormer:Rethinking Linear Attention with Laplacian Kernel

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The quadratic complexity of softmax attention presents a major obstacle for scaling Transformers to high-resolution vision tasks. Existing linear attention variants often replace the softmax with Gaussian kernels to reduce complexity, but such approximations lack theoretical grounding and tend to oversuppress mid-range token interactions. We propose LaplacianFormer, a Transformer variant that employs a Laplacian kernel as a principled alternative to softmax, motivated by empirical observations and theoretical analysis. To address expressiveness degradation under low-rank approximations, we introduce a provably injective feature map that retains fine-grained token information. For efficient computation, we adopt a Nyström approximation of the kernel matrix and solve the resulting system using Newton--Schulz iteration, avoiding costly matrix inversion and SVD. We further develop custom CUDA implementations for both the kernel and solver, enabling high-throughput forward and backward passes suitable for edge deployment. Experiments on ImageNet show that LaplacianFormer achieves strong performance-efficiency trade-offs while improving attention expressiveness. Code is available at the following site: \href{https://mike7472727.github.io/laplacianformer.github.io/}{\textcolor{black}{LaplacianFormer }}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a Laplacian-kernel-based linear attention matrix approximation. The query-key distribution of a vision transformer is heavy-tailed, whereas a Gaussian kernel assumes fast-decaying tails. This motivates the use of a Laplacian kernel, which additionally improves gradient stability and attention map representation. To achieve linear computational complexity, the Laplacian kernel matrix is approximated via the Nyström method, where the time-consuming matrix inversion is replaced with GPU-friendly Newton–Schulz iterations. The Laplacian kernel was tested on ImageNet classification, object detection, and instance segmentation, achieving higher accuracy while using fewer FLOPs than other models.

### Strengths
I am inclined to recommend acceptance of this paper because the proposed method makes a meaningful contribution to linear attention research, with sufficient empirical motivation and supporting experimental results. Additionally, the paper addresses several practical challenges related to matrix inversion, which may promote the adoption of this work in future research. There are some caveats, which are discussed in Weaknesses, but I don’t think they outweigh the strengths. Please refer to the following items and Weaknesses for additional reasons behind this recommendation.

- Although the Laplacian kernel is not driven by theoretical connections from the Softmax attention as many prior works were, the it is motivated clearly by the empirical observation. Also, the algorithm is implemented with the consideration of practicality and efficiency.
- The paper is well written and easy to follow. The proposed method is well positioned in the literature, and the Laplacian linear attention algorithm is explained clearly and thoroughly.
- The expressiveness of the Laplacian kernel is supported by experimental results. The Laplacian models achieve superior accuracy compared to other methods, and the memory savings remain within a similar range to other linear attention methods.

### Weaknesses
Although the paper is well presented, a few points could make it stronger:

1. Although the linear attention is implemented via a custom GPU kernel, the GPU latency of the Laplacian kernel is only self-compared and not compared with other methods. Although the pseudoinverse of W is implemented via a custom GPU kernel using Newton–Schulz iterations, it is unclear whether the end-to-end latency of a Laplacian-kernel transformer is within a practical range. Comparing the end-to-end latency of the proposed method with other linear attention methods—especially those without matrix inversion—could help clarify its practicality.
2. The gradient behavior of the Laplacian kernel regarding non-vanishing gradients is discussed only in the context of full attention. It is unclear whether the same behavior holds when the kernel matrix is approximated by the Nyström method.
3. It is unclear whether the diagonal matrix approximates the whitening matrix effectively.
4. Some prior works mentioned in the experimental results are not sufficiently explained in the text.

### Questions
1. Why is the injectivity of Eq. 4 important? Is Softmax attention also injective?
2. In addition to Weakness1, what is the latency of the forward and backward paths when using the basic PyTorch or CUDA matrix inversion functions (e.g., `torch.linalg.solve` or `torch.linalg.lstsq`)?

Minor comments

- The font sizes in the figures are too small.
- Eq. equation 4 → Eq. 4 or equation 4 in Line 264.

### Soundness
2

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
3

### Summary
This paper proposes LaplacianFormer, a new Transformer variant that addresses the quadratic complexity of softmax attention by replacing the commonly used Gaussian kernel with a Laplacian kernel. The authors argue that the Laplacian kernel, based on $l_1$ distance, is better suited for the heavy-tailed distribution of query-key interactions in vision models, preventing the over-suppression of mid-range tokens. To ensure efficiency, the model uses a Nyström approximation solved via a CUDA-accelerated Newton-Schulz iteration, achieving good accuracy on ImageNet.

### Strengths
- Shows with real data that current Gaussian kernels oversuppress mid-range token interactions, making the Laplacian kernel a justified alternative.
- Achieves best accuracy across all model sizes on ImageNet, showing the efficacy of the proposed method.
• Uses GPU-friendly Newton-Schulz iteration that converges better than CG under poor conditioning.

### Weaknesses
- The "provably injective" claim relies on circular reasoning (Appendix A assumes $\phi$ is already injective) and contains a mathematical error (Step 2 incorrectly claims centering preserves distinctness). The practical implementation uses diagonal whitening, further departing from the theoretical setup.
- Claims that Gaussian kernel gradients "diminish quadratically" near zero are incorrect (they're linear in $x_i - y_i$).
- The normalization (Eq. 4) includes query-dependent mean subtraction over all keys, which appears to break the precomputation trick. The paper doesn't show how Nyström factorization maintains $O(Nm)$ complexity with this coupling.
- Table 3 reports different accuracy values (81.4% vs 79.2%) than the text claims (81.1% vs 77.8%) for the same comparison, and so is the other comparison.
- Only compares against Gaussian kernels, not other alternatives (polynomial, cosine). No evaluation on truly long sequences despite abstract claims about scalability.

### Questions
1.  How does the diagonal whitening approximation (Eq. 6) preserve the claimed injectivity when Appendix A requires full-rank $\Sigma^{-1/2}$? Can you provide a proof for the diagonal case?
2. How does the query-dependent mean subtraction in Eq. 4 maintain $O(Nm)$ complexity when it couples all queries to all keys?
3. Please fix the inconsistency for Table 3.
4. The Gaussian gradient is linear in $(x_i - y_i)$ near zero, not quadratic as claimed - can you correct this analysis?
5. The abstract mentions "long input sequences" but experiments are on 224×224 images - do you have results on actual long sequences?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces LaplacianFormer, a Transformer variant that replaces the commonly used Gaussian kernel in linear attention with a Laplacian kernel to better capture heavy-tailed token distance distributions and enhance training stability.

### Key Motivations:  
Empirical analysis of query–key distance distributions in DeiT, PVT, and Swin Transformers reveals a heavy-tailed pattern, inconsistent with the rapid decay assumed by Gaussian kernels.  
The proposed Laplacian kernel more accurately models this behavior and preserves non-vanishing gradients, leading to more stable and efficient optimization.

### Methods:  
The authors design a provably injective attention mapping, ensuring that distinct queries produce distinct outputs and preserving fine-grained token relationships.  
They further employ a Nyström approximation combined with Newton–Schulz iteration to efficiently compute the kernel inverse, avoiding expensive matrix decompositions.  
Optimized CUDA implementations are provided for both the Laplacian kernel and the solver, enabling scalable training and inference.

### Experiments:  
LaplacianFormer achieves faster and more stable convergence than Gaussian-based baselines (e.g., SOFT++, Skyformer).  
It delivers strong performance–efficiency trade-offs on ImageNet and COCO benchmarks and demonstrates linear memory scaling and robustness under high condition numbers, validating both the stability and scalability of the proposed approach.

### Strengths
### 1. Convincing analysis of Laplacian vs. Gaussian kernels
The paper provides a clear and well-supported theoretical and empirical comparison between Laplacian and Gaussian kernels.
The analysis of query–key distance distributions in vision Transformers (e.g., DeiT, PVT, Swin) convincingly shows that these distances follow a heavy-tailed pattern, making the Gaussian assumption unrealistic.

### 2. Theoretical soundness and novel properties
The authors derive a provably injective attention mapping, ensuring that distinct queries produce distinct outputs.
This injectivity and the non-vanishing gradient property are both novel and important contributions that improve expressivity and optimization stability, addressing common issues like representation collapse in previous linear attention models.

### 3. Methodological clarity and efficiency
The use of Nyström approximation and Newton–Schulz iteration for efficient kernel inversion is elegant and well-motivated.
The method maintains linear complexity while improving numerical stability.
The paper provides implementation details and custom CUDA optimization, supporting reproducibility and practical relevance.

### Weaknesses
### 1. Section 4.1 lacks coherence and clarity
The description of LaplacianFormer in Section 4.1 feels disconnected from the rest of the paper.
In earlier sections, the authors introduce linear self-attention and its approximation, but Equation (4) appears abruptly without derivation from Equation (3).
Similarly, Equations (8) and (9) in the approximation section are difficult to relate back to Equation (4), making the mathematical flow hard to follow.

### 2. Limited experimental coverage
Only image classification and object detection are evaluated, which is insufficient to justify general claims about linear self-attention.
The paper should include more diverse and representative benchmarks such as:
- Vision generation: DiT (Diffusion Transformers)
- Vision–language: CLIP, BLIP, or Flamingo
- Language modeling: The Pile
- NLP understanding and reasoning: GLUE and ARC

These are widely used in prior efficient-attention works like Longformer, Performer, and FlashAttention.

### 3. Limited novelty and missing discussion of limitations
While the comparison between Laplacian and Gaussian kernels is well-analyzed, simply changing the kernel may not constitute a sufficiently strong contribution for a top-tier venue like ICLR.
The paper should also discuss the limitations of the Laplacian kernel, such as potential issues in scaling, expressivity trade-offs, or behavior in non-vision tasks.

### Questions
1. Equation (4) appears without derivation or explanation. Could the authors clarify how Equation (4) is obtained from the preceding formulation, and what assumptions or approximations lead to this step?  

2. Why is the whitening matrix defined as $\Sigma^{-1/2}A$ instead of $\Sigma^{-1}A$?  
   Typically, symmetric whitening uses $\Sigma^{-1/2} A \Sigma^{-1/2}$.  
   Please explain the rationale behind this choice and how it affects the theoretical properties of the transformation.

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
This paper revisits linear attention and argues that the usual Gaussian kernel is not the best match for how query–key distances actually behave in vision Transformers. The authors show that distance distributions from real models (DeiT, PVT, Swin) are heavier-tailed, so a slower-decaying kernel is more appropriate. They therefore use a Laplacian kernel based on L1 distance, and make it practical with a Nyström-style approximation, a Newton–Schulz solver for the small kernel inverse, and a lightweight whitening step to keep the representation from collapsing. The method is evaluated on ImageNet-1K and COCO? (I guess..) (detection/segmentation) across several model sizes and generally matches or improves over existing efficient/linear-attention backbones. The paper is readable and implementation-minded.

### Strengths
1. Motivation is concrete. The paper does not just say “we use another kernel,” but first measures Q–K distances from trained vision models and shows they are heavy-tailed. That gives a reasonable justification for trying a Laplacian kernel rather than the default Gaussian.
2. The method goes beyond a trivial kernel swap. The injective feature mapping and diagonal whitening are intended to address the expressivity loss that often comes with low-rank/Nyström variants. Even if the theoretical part is short, the design is more careful than many incremental kernel papers.
3. The implementation path is clear. Using Nyström plus Newton–Schulz and providing a CUDA-friendly formulation makes it more likely that people can actually run it.
4. Experiments are not limited to a single classification model. The authors test on classification and dense prediction, and across multiple scales, which helps show the effect is not a one-off.

### Weaknesses
1. The comparison space is narrow. The central claim is that Gaussian is not ideal and Laplacian fits the data better, but the experiments mostly compare against Gaussian-style baselines. Including at least one other simple non-Gaussian/linearizable kernel (e.g. cosine-type) would make the claim more specific to Laplacian rather than “any slower decay works.”
2. The injectivity argument is under-documented in the main text. It is mentioned and deferred to the appendix, but the reader cannot see clearly under what assumptions it holds or how much of it remains after the Nyström approximation. Since this is one of the few parts that differentiates the method conceptually, it should be stated more explicitly.
3. The relation to recent Nyström/efficient ViT work could be clearer. Many recent papers also do landmark selection, low-rank approximation, and stabilization of the inverse. Here the novelty is framed as “Laplacian + Newton–Schulz + whitening,” but it would help to say more precisely what is new compared to those works.

### Questions
1. Can you include at least one non-Gaussian baseline to show that the improvement is tied to the Laplacian choice?
2. Can you move a short version of the injectivity result into the main paper and state whether the Nyström step affects it?
3. Do you use the same kernel scale for all stages/resolutions, or do you tune it per stage?

### Soundness
2

### Presentation
3

### Contribution
3
