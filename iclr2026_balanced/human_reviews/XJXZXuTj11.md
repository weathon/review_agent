## Human Reviewer 1

### Summary
This paper introduces QVGen, a quantization-aware training (QAT) framework specifically designed to enable high-performance, low-bit (≤4-bit) quantization of large-scale video diffusion models (DMs) based on the diffusion transformer (DiT) architecture. QVGen incorporates auxiliary modules ($\Phi$) to mitigate the quantization error during training, supported by a theoretical analysis linking reduced gradient norm to improved convergence. To remove the inference burden of these modules, the paper proposes a rank-decay strategy using singular value decomposition (SVD) and a rank-based regularization for progressively pruning $\Phi$ without significant loss in quality. Extensive evaluations across multiple state-of-the-art video DMs show QVGen achieves performance on par with full-precision models at 4-bit, and significantly outperforms competing quantization methods, with both quantitative metrics (VBench) and qualitative visualizations.

### Strengths
The work addresses the challenging and critical task of efficient, high-fidelity video generation under ultra-low-bit quantization, an area with clear importance for practical deployment.

Provides a regret-based convergence analysis (see Theorem 3.1, Page 4) linking gradient norm to QAT performance, justifying the introduction of $\Phi$.

The auxiliary module ($\Phi$) is elegantly conceived and is integrated with a flexible, theoretically justified rank-decay scheme, allowing benefits during training while incurring no inference cost.

Experiments cover a wide range of SOTA video DMs (from 1.3B to 14B parameters) with comprehensive ablations (Tables 3, 4, 5; Figs. 3–6). Quantitative results (Tab. 1, 2, H, K; Figs. 5, 6) convincingly show QVGen’s superiority over PTQ and QAT baselines in essentially all relevant metrics.

### Weaknesses
While the use of singular value decomposition is effective (Fig. 4, Section 3.2), the alternative strategies (Sparse, Residual Quantization) examined in Table 6 are somewhat strawman/naive and do not fully explore more sophisticated structured pruning or adaptive fading that could yield competitive trade-offs. There is little discussion of possible pathological cases where the SVD approach might fail, particularly if singular spectrum decays slowly.

The key result (Theorem 3.1, Page 4) relies on convexity of $f_t$, which is non-standard for deep networks, and the analysis lacks formal proof linking reduced regret to generalization in nonconvex settings. No concrete evidence connects gradient norm changes to video-specific generative performance beyond empirical plots.

While the main approach is clearly stated, critical aspects of $\Phi$ require deeper explanation, such as its initialization scheme across different models, architecture, and potential sensitivity to scale or data distribution shifts. Section J.2 offers two initialization methods, but does not thoroughly evaluate edge cases or sensitivity to poor initialization. There is also no discussion of stability if $\Phi$'s rank is diminished too rapidly relative to the learning rate schedule.

### Questions
While the use of singular value decomposition is effective (Fig. 4, Section 3.2), the alternative strategies (Sparse, Residual Quantization) examined in Table 6 are somewhat strawman/naive and do not fully explore more sophisticated structured pruning or adaptive fading that could yield competitive trade-offs. There is little discussion of possible pathological cases where the SVD approach might fail, particularly if singular spectrum decays slowly.

The key result (Theorem 3.1, Page 4) relies on convexity of $f_t$, which is non-standard for deep networks, and the analysis lacks formal proof linking reduced regret to generalization in nonconvex settings. No concrete evidence connects gradient norm changes to video-specific generative performance beyond empirical plots.

While the main approach is clearly stated, critical aspects of $\Phi$ require deeper explanation, such as its initialization scheme across different models, architecture, and potential sensitivity to scale or data distribution shifts. Section J.2 offers two initialization methods, but does not thoroughly evaluate edge cases or sensitivity to poor initialization. There is also no discussion of stability if $\Phi$'s rank is diminished too rapidly relative to the learning rate schedule.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper proposes a quantization method for video diffusion transformers. The key idea is to introduce additional modules containing full-precision lora parameters to mitigate aggressive training losses, whose effectiveness of reducing gradient norm and quantization error has been demonstrated by the authors. Then, the authors devise a rank shrinking strategy, which reduces the rank to 0 to avoid additional stroage of full-precision weights. Extensive experiments demonstrate the effectiveness.

### Strengths
1. The work could be highly impactful for the community of quantized video generation models due to its state-of-the-art performance.
2. The analysis of the importance of reducing the gradient norm is valid and motivating for the proposed method.
3. Although the method first introduces full-precision parameters, the authors devise effective solutions to reduce the rank to even 0, which means eliminating the need for additional full-precision storage. From the results, such a two-stage pipeline is highly effective and superior to one-stage methods.
4. The experiments are very extensive and sufficient to demonstrate the effectiveness of quantizing video diffusion models.

### Weaknesses
I don't find so many weaknesses, but would like to list some minor points below:
1. It seems that the method is not tailored for video diffusion models and has potential for other models, like image generation and image backbone. The authors are encouraged to conduct experiments on these widely adopted benchmarks.
2. It is encouraged to include another baseline of fine-tuning the model using the same data under full precision, which is useful to reflect the effect introduced by additional data and fine-tuning.

### Questions
* Is it possible to directly regulate gradient norm by gradient normalization, clip, or direct optimization? These methods may yield inferior results, but can they outperform the original baseline introduced in Sec. 2?
Please refer to the weaknesses part above for other questions.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces QVGen, a novel Quantization-Aware Training (QAT) framework designed specifically for very low-bit (≤4-bit) video quantization. The core of this method is the introduction of an auxiliary module ($\Phi$) to stabilize the QAT process. To eliminate the inference overhead from this auxiliary module, the authors propose a "rank-decay" strategy, which uses Singular Value Decomposition (SVD) and rank regularization to progressively remove these modules during training. Experiments on video DMs ranging from 1.3B to 14B parameters show that QVGen achieves quality comparable to full-precision at 4-bit settings.

### Strengths
1. The first full QAT method for video generation models I'm aware of.

2. Experiments are conducted on four SOTA open-source video DMs (CogVideoX and Wan), with parameter scales from 1.3B to 14B, providing broad coverage.

3. It validates practical efficiency gains and demonstrates orthogonality with other acceleration techniques like SVG.

4. The provided experimental materials are comprehensive, and the ablation studies are extensive.

### Weaknesses
1. Since some other QAT methods are trained using only LoRA, a comparison of training time and memory (GPU VRAM) requirements against these methods should be provided for a comprehensive assessment of algorithm efficiency.

2. Quantization-related initialization settings should be specified, such as the choice of quantizer (e.g., granularity, symmetric/asymmetric) and which layers, if any, are not quantized.

3. In Fig.3, why the inital training loss of the proposed method is bettern than Q-DM? Did the proposed method use a better initialization strategy? Since the authors reproduce other QAT baselines, a detailed training loss curve comparison would greatly enchance the soundness of the paper.

### Questions
Please see the weaknesses above, and:

1. Can the proposed method be combined with SVDQuant? If so, could this combination be trained efficiently by updating only LoRA parameters?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces QVGen, a quantization-aware training (QAT) method designed to enable high-quality video diffusion models under extremely low-bit quantization (4-bit or 3-bit). The paper begins with a theoretical analysis showing that reducing the gradient norm is key to stabilizing and improving convergence in training. Based on this insight, the paper introduces auxiliary modules $\Phi$ during QAT to compensate for weight quantization errors and smooth optimization. To avoid extra inference overhead from these modules, the paper proposes a rank-decay mechanism, progressively eliminating $\Phi$ by decomposing its weights via singular value decomposition (SVD) and decaying low-contributing components using a rank-based regularization schedule. Extensive experiments on state-of-the-art models demonstrate that QVGen achieves full-precision comparable performance in 4-bit settings and sets new records under 3-bit quantization, outperforming existing QAT and PTQ baselines such as Q-DM, EfficientDM, and SVDQuant. The method also improves memory efficiency and inference speed while maintaining compatibility with standard low-bit kernels.

### Strengths
1. **Well-motivated methodology**: The proposed approach is grounded in a clear motivation. The authors observe that the auxiliary module $\Phi$ exhibits rank decay during quantization-aware training and effectively leverage this property through their rank-decay mechanism. This insight is both intuitive and novel.
2. **Comprehensive experimental validation**: The paper provides extensive experimental results across multiple large-scale video diffusion models, supported by detailed ablation studies and qualitative visual comparisons.
3. **Strong empirical performance**: QVGen achieves consistently superior results compared to existing QAT and PTQ baselines, showing that the proposed framework is highly effective in maintaining video generation quality under ultra-low-bit quantization while improving efficiency.

### Weaknesses
1. **Theory disconnected from the main method**: The theoretical analysis in the early part of the paper appears largely disconnected from the core contributions. Although the authors claim that the auxiliary module $\Phi$ is motivated by this theory, the linkage is tenuous, and the theoretical result itself is rather weak. As a result, the theory does not substantially contribute to the understanding or justification of the proposed framework (although I think it is totally fine to be motivated empirically).

2. **Limited acceleration gains**: The acceleration gains reported by QVGen are not particularly large, which somewhat limits its practical impact on efficiency. However, the authors explicitly acknowledge this limitation and attribute it to the absence of kernel fusion optimizations, which is reasonable.

### Questions
1. The proposed QVGen framework focuses on compensating quantization errors in the weights through the auxiliary module $\Phi$. However, the paper does not analyze the effect of activation quantization separately. Could the authors provide results or discussion on how the model performs when only activations are quantized to low-bit precision while keeping the weights in full precision? This would help clarify whether the main difficulty in quantizing video diffusion models arises more from weights or activations.

2. The paper mentions that the current acceleration is limited because kernel fusion is not applied. To better understand the efficiency aspect, could the authors provide additional profiling results—such as the achieved TFLOPs of the INT4 GEMM kernel used—and an end-to-end inference time breakdown? It would be very helpful if the authors could use a profiling tool (e.g., nsys profile) with `torch.cuda.nvtx.range_push` and `torch.cuda.nvtx.range_pop` tags to visualize where most time is spent and estimate how much of the overhead could be mitigated by kernel fusion, even without implementing it. The author is encouraged to provide a table that describe how much time the qkv projection, o projection, ffn up projection, ffn down projection, self attention, and all other memory bound modules take in a single dit block.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper introduces QVGen, a novel Quantization-Aware Training (QAT) framework designed to enable high-quality video generation under extremely low-bit (3/4-bit) quantization, a task where previous methods fail. The key innovation is a two-stage process: first, it stabilizes training and improves convergence by adding lightweight auxiliary modules to mitigate quantization error, which is theoretically and empirically shown to reduce the gradient norm; second, it progressively eliminates these modules during training via a "rank-decay" strategy that uses Singular Value Decomposition (SVD) and a rank-based regularization to identify and shrink low-impact components to zero, resulting in a final quantized model with no inference overhead.

### Strengths
1. Well-Motivated Contribution: This paper addresses the challenge of ultra-low-bit QAT for large-scale video diffusion models, filling a significant gap in the literature which has primarily focused on image models.

2. Strong Theoretical and Empirical Foundation: The paper provides a solid theoretical analysis linking gradient norm reduction to improved QAT convergence, and then designs a method (the auxiliary module Φ) specifically to achieve this. The empirical results consistently show lower gradient norms and training loss.

3. Impressive and Extensive Experiments: The evaluation is comprehensive, testing on four state-of-the-art models ranging from 1.3B to 14B parameters. The results are compelling, showing that QVGen is the first method to achieve full-precision comparable quality with 4-bit quantization and significantly outperforms all baselines in 3-bit settings.

4. Practical and Efficient Solution: The proposed "rank-decay" strategy is a clever way to gain the training benefits of the auxiliary modules without incurring any inference cost, making the final model directly deployable with standard low-bit kernels. The reported ~4x memory reduction and up to 1.7x speedup are substantial.

### Weaknesses
1. Computational Cost of QAT: While the final model is efficient, the QAT process itself is expensive, involving iterative SVD operations and training on up to 32 H100 GPUs for large models. The paper does not deeply discuss the trade-offs between this training cost and the resulting inference savings.

2. Limited Analysis of "Rank-Decay": The strategy is shown to work, but the analysis of why the singular values of W_Φ evolve to have an increasing number of small components is somewhat surface-level. A deeper investigation into the dynamics between the quantized model and the auxiliary module during training would be valuable.

3. Ablation on Simpler Alternatives: The paper compares against other fine-grained decay strategies (Sparse, Residual Quantization), but it would be strengthened by also ablating against a simple scheduled decay (e.g., linearly reducing the magnitude of all parameters in Φ to zero) to more clearly isolate the benefit of the SVD-based, rank-aware approach.

### Questions
Hyperparameter Sensitivity: How sensitive is the final performance to the key hyperparameters, such as the initial rank r=32 and the shrinking ratio λ=1/2? Was there a systematic process for selecting these values across different model architectures and sizes?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3