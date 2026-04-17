# GRLA: Bridging Softmax and Linear Attention via Gaussian RBF Kernel for Lightweight Image Super-Resolution

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Lightweight image super-resolution (SR) requires effective modeling of long-range dependencies under stringent computational constraints. Although self-attention mechanisms are highly effective for this task, their quadratic computational complexity imposes a prohibitive constraint in lightweight SR applications. Existing linear attention methods reduce complexity to linear but significantly underperform compared to Softmax attention due to their inability to explicitly model the Euclidean distance between query and key vectors. Through mathematical derivation, we demonstrate that the core operation of standard Softmax attention, $\exp({Q}_i^T {K}_j)$, is equivalent to an unnormalized Gaussian Radial Basis Function (GRBF) kernel. Building on this insight, we propose a GRBF-based linear attention mechanism (GRBFLA), which reformulates a distance-aware GRBF kernel that is amenable to Taylor series expansion, enabling linear approximation. This kernel progressively approximates the behavior of standard Softmax attention while maintaining linear complexity. Based on GRBFLA, we develop a lightweight image SR architecture termed GRLA. Experimental results show that for ×4 SR on the Manga109 dataset, GRLA outperforms the representative self-attention model SwinIR-light by 0.57 dB in PSNR while reducing computational cost FLOPs by 11\%. Compared to the state-of-the-art Mamba-based lightweight model MambaIRv2-light, GRLA achieves a 0.25 dB higher PSNR with a 25\% reduction in FLOPs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents GRLA, a novel lightweight image super-resolution architecture. Its core contribution is the Gaussian Radial Basis Function-based Linear Attention (GRBFLA) module, which leverages a mathematical equivalence between the Softmax attention kernel and an unnormalized Gaussian RBF kernel. By applying a first-order Taylor approximation, the authors derive a linear-complexity attention mechanism that explicitly preserves distance awareness, a key property often lost in prior linear attention methods. The paper is well-structured, with thorough experiments demonstrating state-of-the-art performance on standard benchmarks, outperforming strong baselines like SwinIR-light and MambaIRv2-light in both accuracy and efficiency.

### Strengths
The key insight—mathematically linking the standard Softmax attention's core computation exp(Q_i^T K_j) to an unnormalized GRBF kernel—is significant and elegantly derived. This provides a principled foundation for building a linear attention mechanism, moving beyond ad-hoc kernel designs.

The paper successfully addresses a major weakness of existing linear attention methods: their lack of explicit Euclidean distance modeling. The proposed GRBFLA mechanism convincingly bridges the performance gap between efficient linear attention and powerful Softmax attention, as evidenced by the strong experimental results.

The method is described clearly, with step-by-step mathematical derivations. The appendix provides necessary details on the network architecture, training setup, and datasets, supporting the reproducibility statement.

Beyond standard PSNR/SSIM metrics, the paper includes visual comparisons (LAM, output images), computational cost (FLOPs, Params), and critically, inference latency and training memory footprints, which are crucial for lightweight applications.

### Weaknesses
The first-order Taylor approximation is central to the method. While Figure 1(b) and the choice of a small γ suggest it's accurate, the paper would be strengthened by a more formal analysis or discussion of its limitations. For instance, under what conditions (e.g., with high-dimensional or unnormalized features) might this approximation break down, and how does the model's performance correlate with the approximation error?

The overall GRLA architecture incorporates several components (TWSA, TLA, Multi-layer Polymerization Blocks, 1x1 conv for aggregation). While the ablation studies show the importance of TLA and aggregation connections, the specific design choices (e.g., why this particular synergy between TWSA and TLA? How was the number of MPBs optimized?) lack deep justification. The architecture feels somewhat complex, and its novelty is somewhat overshadowed by the core GRBFLA contribution.

The related work section and experiments could more directly position GRLA against other recent attempts to inject spatial or distance awareness into efficient architectures. While compared against general linear attention methods, a discussion on how it specifically improves upon other potential distance-sensitive kernels would sharpen the contribution.

The GRBFLA module itself is a solid conceptual contribution. However, the GRLA network, as a whole, can be perceived as an incremental architectural advance that integrates this new module into a now-standard design paradigm for lightweight SR (residual blocks, feature distillation, windowed attention). The performance gains, while clear, are not revolutionary. The borderline recommendation stems from this balance between a strong conceptual contribution and a more incremental systems/application contribution.

### Questions
The first-order Taylor approximation relies on 2γ Q_i^T K_j being small. Did you observe any instability or performance degradation during training on datasets with significantly different feature distributions than DIV2K? Is there a risk of the approximation becoming less valid in deeper layers of the network?

The paper focuses on image SR. Have you explored or do you plan to explore the applicability of the GRBFLA module in other vision tasks that rely heavily on long-range dependencies (e.g., segmentation, detection)? A brief discussion on its generalizability would be valuable.

Given that the core innovation is the GRBFLA module, was a simpler baseline model (e.g., replacing the attention module in SwinIR-light with GRBFLA) tested? This would help isolate the performance gain purely from the new attention mechanism versus the overall GRLA architecture.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors claim the weakness of standard self-attention is too computationally expensive (quadratic complexity), while existing linear-attention methods are inefficient because they fail to properly model the distance between pixels. The authors' main insight is that the standard Softmax attention mechanism is mathematically equivalent to a Gaussian Radial Basis Function kernel, then they claim this is important because the GRBF kernel is inherently aware of Euclidean distance. The authors then propose a new linear attention, GRBFLA, which uses a reformulated GRBF kernel. They then use a Taylor series expansion to approximate this kernel, which allows them to achieve linear computational complexity while still retaining the distance-aware properties of the original Softmax attention. They build a lightweight SR model called GRLA using this new mechanism.

### Strengths
1. The overall presentation is good and easy to follow.

2. The authors provided a mathematical insight linking Softmax attention to GRBF kernels, improving the current linear attention methods that lack this distance-awareness.

3. The quantitative results seem good. The proposed GRLA model outperforms two different state-of-the-art lightweight models from different architecture families.

4. The authors provided a visual comparison showing that their approximated GRBF kernel's behavior better matches the standard Softmax kernel.

### Weaknesses
1. The method employs a first-order Taylor approximation of the reformulated GRBF kernel and claims convergence as a hyperparameter. However, several questions arise: How accurate is this approximation in practice? Why is γ=1/2 optimal in Table 4 rather than larger values being better? How do higher-order Taylor approximations perform?

2. The comparative analysis is insufficient. The proposed problem regarding linear attention's weakness exists not only in lightweight SR tasks but also in larger-scale and other image restoration tasks. The authors' focus on only lightweight SR tasks for comparison raises concerns about inadequate validation, limiting the paper's overall insights.

3. In SR task research, the overall network pipeline is crucial; therefore its placement in the appendix rather than the main text is not recommended.

4. Although the proposed methods demonstrate better qualitative results in terms of PSNR and FLOPs, they also lead to an increase in parameters.

### Questions
1. Please see the weakness section.

### Soundness
3

### Presentation
2

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
This paper introduces a linear attention for lightweight image super-resolution, deriving its formulation from a GRBF kernel generalization of Softmax attention. The proposed model achieves competitive results with lower computational cost than existing lightweight models, offering a practical solution for resource-constrained devices.

### Strengths
1. The mathematical derivation linking the Softmax to the GRBF kernel is the most notable advantage of this work. It provides a principled and insightful explanation for the limitations of existing linear attention methods and a solid foundation for the proposed GRBFLA.
2. The first-order Taylor approximation is a simple yet effective way to achieve linear complexity while preserving distance awareness, as validated by the ablation studies.

### Weaknesses
1. To evaluate model efficiency, the inference latency of the GRLA was measured and compared against other methods. However, the paper fails to provide an ablation experiment utilizing the method in Equation 7 to validate the effectiveness of the first-order Taylor approximation within the proposed GRLA framework.
2. The first-order Taylor approximation, $\mathbf{e}^x = 1 + x$ , yields a sufficiently small approximation error only when the magnitude of $x$ is sufficiently small. While normalization and $\gamma$ selection are theoretically motivated, the paper lacks empirical proof that $2\gamma Q_{i}^{T}K_j$ remains sufficiently small.
3. In Table 3, the majority of methods selected for comparison with GRLA are outdated, and only MambaIR-light and MambaIRv2-light are lightweight SR methods from the past two years.
4. In Table 3, the comparative analysis of GRLA lacks comparisons against other linear attention-based methods.
5. In Table 5, the studies lack an ablation experiment investigating the incorporation of the TLA module under the condition of a TMHSA window size of 8.
6. In Equation (11), the notation is inconsistent: the symbol φ in the numerator is replaced by ϕ in the denominator. This should be corrected to maintain notational consistency throughout the equation.

### Questions
1. Does the method employing the first-order Taylor approximation surpass its counterparts not employing this approximation in terms of computational efficiency?
2. The first-order Taylor approximation, $\mathbf{e}^x = 1 + x$, yields a sufficiently small approximation error only when the magnitude of x is sufficiently small. Can it be theoretically explained that the effectiveness of the first-order Taylor approximation for lightweight SR tasks?

### Soundness
2

### Presentation
2

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
The paper addresses the challenge of modeling long-range dependencies in lightweight image SR under computational constraints. It proposes a GRBF-based linear attention mechanism, which reformulates the GRBF kernel to approximate standard Softmax attention while reducing complexity from quadratic to linear. Key contributions include mathematical derivation of equivalence between GRBF and Softmax attention, a first-order Taylor approximation for linear computation, and the GRLA architecture. Experimental results on datasets like Manga109 show PSNR improvements with reduced FLOPs, as detailed in the abstract and Table 3.

### Strengths
The paper demonstrates clear mathematical derivations linking GRBF to Softmax attention (Sec. 3.2), systematic experimental evaluation across multiple datasets (Sec. 4), and computational efficiency gains (e.g., reduced FLOPs and latency in Tables 3 and 7). Visualizations (Figs. 2, 4, 8) effectively highlight the method's ability to capture long-range dependencies, and the architecture integrates local and global features (Appendix A.1).

### Weaknesses
[1] Seems incremental? The core idea of using a GRBF kernel in linear attention is a specific instantiation within an established research direction. It builds directly upon prior linear attention works like (1,2,3), offering an incremental improvement rather than a significant conceptual shift.

[2] Performance improvements are small (e.g., +0.02-0.03 dB PSNR on BSD100 in Table 3), and the Taylor approximation (Eq. 8) may not hold for all input distributions, lacking error bounds (specifically, error analysis and statistical significance analysis). Besides, the ablation studies (Tables 1-2) validate components but lack depth in error analysis.

[3] Comparisons omit recent state-of-the-art methods, and the model complexity increases with multi-layer aggregation without proportional gains (Table 6).

[4] The central claims are partially supported by mathematical derivations (e.g., Eqs. 3-7 in Sec. 3.2) and experiments on multiple benchmarks (Tables 1-3). However, the Taylor approximation (Eq. 8) relies on strong assumptions (e.g., small inner products). 

[5] Network architecture details in Appendix A.1 are somewhat vague.

(1) Fan, Qihang, et al. "Rectifying magnitude neglect in linear attention." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
(2) Shen, Zhuoran, et al. "Efficient attention: Attention with linear complexities." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.
(3) Qiu, Yuwei, et al. "Mb-taylorformer: Multi-branch efficient transformer expanded by taylor formula for image dehazing." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

### Questions
[1] How does the GRBF kernel fundamentally differ from other potential kernel choices (e.g., polynomial, Laplace RBF) in the linear attention framework, both theoretically and empirically? Could you provide a more general analysis of kernel selection?

[2] The first-order Taylor approximation is valid for small values. What is the quantitative distribution of this term during training on typical SR datasets? How does the model behave when this assumption is violated? For example, vectors are not normalized or gamma is large? 

[3] Beyond PSNR/SSIM, are there specific types of image structures or textures where GRBF-LA demonstrates a more pronounced advantage over other linear attention methods, based on your analysis?

[4] The method combines window-based attention (TWSA) and linear attention (TLA). What is the relative contribution of each component to the final performance? Is the GRBF-LA module effective enough to potentially replace local window attention entirely in some scenarios?

[5] Why are the performance gains modest despite the claimed equivalence to Softmax attention, and have you tested generalization to other tasks like video SR?

### Soundness
2

### Presentation
3

### Contribution
2
