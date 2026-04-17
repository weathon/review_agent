# QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Diffusion transformers exhibit remarkable video generation capability, yet their prohibitive computational and memory costs hinder practical deployment. Model quantization and attention sparsification are two promising directions for compression, but each alone suffers severe performance degradation under aggressive compression. Combining them promises compounded efficiency gains, but naive integration is ineffective. The sparsity-induced information loss exacerbates quantization noise, leading to amplified attention shifts. To address this, we propose **QuantSparse**, a unified framework that integrates model quantization with attention sparsification. Specifically, we introduce *Multi-Scale Salient Attention Distillation*, which leverages both global structural guidance and local salient supervision to mitigate quantization-induced bias. In addition, we develop *Second-Order Sparse Attention Reparameterization*, which exploits the temporal stability of second-order residuals to efficiently recover information lost under sparsity. Experiments on HunyuanVideo-13B demonstrate that QuantSparse achieves 20.88 PSNR, substantially outperforming the state-of-the-art quantization baseline Q-VDiT (16.85 PSNR), while simultaneously delivering a **3.68$\times$** reduction in storage and **1.88$\times$** acceleration in end-to-end inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes QuantSparse, a framework to jointly compress large video diffusion transformers using model quantization and attention sparsification. The authors argue that naively combining these two techniques leads to severe performance degradation due to "amplified attention shifts," where quantization noise exacerbates sparsity-induced information loss. To solve this, they introduce two techniques: 1. Multi-Scale Salient Attention Distillation (MSAD): A post-training quantization (PTQ) calibration method that uses a memory-efficient distillation loss. It combines (a) global guidance from downsampled attention maps to preserve coarse structure and (b) local guidance focused on high-saliency tokens to preserve fine-grained details. 2. Second-Order Sparse Attention Reparameterization (SSAR): An inference-time technique that aims to recover information lost to sparsity. It posits that the second-order residual (the change in the attention residual between timesteps) is more temporally stable than the first-order residual, especially under quantization. It caches this second-order residual (projected via SVD) to apply a more accurate correction at subsequent steps. Experiments on large models (Hunyuan-13B, Wan2.1-14B) show that QuantSparse achieves significant storage (3.8x) and latency (1.88x) reductions while claiming to maintain or even exceed the quality of the full-precision models.

### Strengths
1. Compressing large-scale video generation models is a critical bottleneck for their practical deployment. The paper identifies the interaction between quantization and sparsification as the key challenge, which is a more nuanced and accurate problem statement than addressing each in isolation.
2. The two proposed components are well-motivated.
3. The method achieves good performance, substantially outperforming existing quantization-only (Q-VDIT) and sparsification-only (SVG) methods, as well as their naive combinations (e.g., Q-VDIT+SVG). 
4. The authors test on multiple, very large models and use a comprehensive suite of evaluation metrics, including VBench and standard video quality metrics (PSNR, SSIM, LPIPS, VQA).

### Weaknesses
1. Unexplained "Better-than-FP" Results: The most significant concern is that the compressed model (QuantSparse) outperforms the 16-bit Full Precision (FP) baseline on key quality metrics.

In Table 5 (Hunyuan-13B, W6A6), QuantSparse (15%) achieves a VQA score of 82.26, while the FP model scores 81.23.

In Table 6 (Wan2.1-14B, W4A8), QuantSparse (15%) achieves an "Imaging Quality" score of 63.81, while the FP model scores 63.38.

While quantization can occasionally act as a regularizer, such outperformance on large-scale generative models is an extraordinary claim that requires an explanation. The authors do not acknowledge, investigate, or explain this phenomenon. This could suggest an undertrained FP baseline, a flaw in the evaluation metrics, or a fundamental misunderstanding of the model's behavior. Without a thorough discussion, these results cast doubt on the validity of the entire evaluation.

2. Inconsistent/Confusing Ablation Study: The paper's narrative motivates MSAD as the primary solution to the core "amplified attention shift" problem. However, the ablation study in Table 3 seems to tell a different story and is presented in a highly confusing manner.

Assuming the "Cache Analysis" block's "None" row (68.00 VQA) represents an MSAD-only model (with no cache) and the "Distillation Analysis" block's "None" row (81.92 VQA) represents an SSAR-only model (with no distillation).

If this interpretation is correct, it implies that SSAR-only (81.92 VQA) provides a vastly larger performance gain over the "None (Cache)" baseline (68.00 VQA) than MSAD-only does. In fact, it suggests SSAR is the dominant contribution, and MSAD provides a smaller, secondary boost (from 81.92 VQA to 91.98 VQA).

This directly contradicts the paper's core motivation, which focuses on the quantization/sparsity interaction (MSAD's job). The authors must clarify the baselines in Table 3 and reconcile this apparent contradiction.

3. The SSAR method introduces a key hyperparameter: the "cache-refreshing interval" (set to 5). This parameter dictates the trade-off between the accuracy of the cached residual and the computational overhead of re-calculating the full-attention map. This is a critical parameter that is not ablated in the paper or appendix.

Minor:
4. I wish the author could reorganize the tables to combine the results table with the Efficiency comparison table to better illustrate.

### Questions
See weakness. I hope the authors could clarify these issues, and I am willing to reevaluate based on the response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents **QuantSparse**, a unified framework for compressing large video diffusion transformers (DiTs) by combining model quantization and attention sparsification. The core problem it addresses is that naively integrating these two compression techniques leads to compounded errors and severe performance degradation. The authors propose a two-part solution: 1) **Multi-Scale Salient Attention Distillation** (MSAD), a post-training alignment method to mitigate quantization-induced bias , and 2) **Second-Order Sparse Attention Reparameterization** (SSAR), an inference-time technique to recover information lost to sparsity. The method achieves impressive results, demonstrating significant reductions in model storage and latency while maintaining the generation quality of the original full-precision models.

### Strengths
1. **Alignment of Non-Trivial Techniques:** The main strength of this work is the **synergistic alignment** of two orthogonal and error-introducing acceleration techniques (quantization and sparsification). The paper correctly identifies that the key challenge lies in managing the compounded errors, and it proposes a dedicated solution for each error source (MSAD for quantization bias, SSAR for sparsity loss) within a single, unified pipeline. This method could also be combined with other methods like SageAttention and TeaCache.

2.  **Excellent Empirical Results:** The method delivers on its promise. The experiments show a significant reduction in model size (e.g., **3.80x** for Wan2.1-14B) and inference latency (e.g., **1.74x** to **1.88x**) across multiple large-scale models. Crucially, this efficiency gain comes with almost no loss—and in some cases, a slight improvement—in video quality metrics compared to the full-precision baselines.

3.  **Good Presentation:** The paper is well-written and clearly presented. The problem statement is well-motivated, and figures like Figure 2, 3, and 4 effectively illustrate the proposed mechanisms and the empirical phenomena that motivate them (e.g., saliency distribution, residual stability).

### Weaknesses
1.  **Complex, Engineering-Heavy Pipeline:** One possible weakness is the **high complexity of the overall system**. The QuantSparse pipeline is highly engineered and involves multiple, intricate components:
    * MSAD is a two-branch distillation process requiring both global average pooling and local top-k salient token selection.
    * SSAR is a multi-stage inference process involving caching, calculating second-order residuals, and then further applying SVD-based projection to denoise these residuals.

    This makes the solution appear less like a simple, generalizable algorithm and more like a complex, bespoke system, which may hinder its broader adoption.

2. **Post-Training Alignment Requirement:** The method relies on a post-training calibration/alignment step (MSAD). This adds overhead compared to a zero-shot compression method, though this is also necessary for PTQ. This alignment step allows the authors to identify and exploit key internal patterns of the DiT, such as the heavy-tailed token saliency distribution shown in Figure 3a, which is a useful finding.

3.  **Theoretical Rigor:** The main weakness lies in the theoretical justification, which is **not tight**. The "theory" section feels more like a *post-hoc* justification for the engineering choices rather than a *first-principles* derivation.
    * **Proposition 3.3 as an Assumption:** The entire SSAR method hinges on **Proposition 3.3**, which states that the second-order residual is more temporally stable than the first-order. This is the most critical claim, yet it is presented as a "proposition" when it is, in fact, an **empirical assumption** justified only by Figure 4a. The paper attributes this to quantization noise being a "slow-varying stochastic process", which is another unproven assumption.
    * **Tautological Theorem:** **Theorem 3.4** is presented as a theoretical guarantee but is effectively **tautological (a circular argument)**. The proof in Appendix A essentially demonstrates that *if* one assumes Proposition 3.3 is true (i.e., the second-order error is smaller), *then* the second-order approximation method has a smaller expected error. This theorem does not provide any new insight beyond restating its own assumption.

### Questions
I find it surprising that the compressed model not only matches but, in several instances, **slightly improves upon the full-precision (FP) baseline** on key video quality metrics (e.g., VQA scores in Table 1 2).

This is an interesting result. Do you have insights on why this occurs?

* Is this improvement simply an artifact of the post-training alignment (MSAD), which may act as a light fine-tuning step, effectively regularizing the model or cleaning up minor noise from the original FP checkpoint?
* Or, does this hint at a more fundamental property of these large video DiTs? For example, could it be that the models are **significantly over-parameterized**, and that the aggressive, aligned compression from QuantSparse acts as a powerful regularizer, forcing the model to rely on its most robust features and leading to a more generalizable output?
* Or is it the metric VQA itself that has some special property for DiTs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes QuantSparse, a unified framework that jointly applies model quantization and attention sparsification to compress video diffusion transformers efficiently. To address the performance drop caused by their naive combination, the authors introduce: Multi-Scale Salient Attention Distillation (MSAD) for aligning quantized and sparse attention using global–local supervision, and Second-Order Sparse Attention Reparameterization (SSAR) for recovering information lost under sparsity via temporally stable residuals.
Experiments on HunyuanVideo and Wan2.1 models show that QuantSparse achieves up to 3.8× compression and 1.7× speedup with minimal quality loss, outperforming prior quantization and sparsification baselines.

### Strengths
The paper demonstrates strong methodological innovation by proposing the first unified framework that integrates model quantization and attention sparsification, along with a systematic analysis of the “amplified attention shift” problem, showcasing both theoretical depth and engineering value. The proposed MSAD and SSAR modules effectively mitigate performance degradation from the perspectives of distillation calibration and dynamic reconstruction, balancing global and local features with good generality and scalability. Extensive experiments on multiple video generation models validate the approach, showing that QuantSparse significantly outperforms existing baselines in metrics such as PSNR, VQA, and LPIPS, while achieving notable model compression and inference acceleration.

### Weaknesses
1. Figure 2 serves as the core illustration of the QuantSparse framework; however, the manuscript does not provide sufficient explanation of each module’s function, their interrelations, or the meaning of the symbols used. It is recommended that the authors include detailed annotations and descriptions of key steps and variables in the main text or appendix to enhance the clarity and reproducibility of the method.

2. The main objective of this paper is to achieve efficient video diffusion model deployment on consumer-grade GPUs through model quantization. Therefore, the specific experimental hardware configurations (e.g., GPU model, memory capacity, and system environment) are crucial for assessing the practical feasibility of the proposed method. The authors are encouraged to explicitly include these hardware details in the main text.

3. In the Introduction, the authors state that directly combining quantization and sparsification causes an “amplified attention shift,” leading to performance degradation. However, only theoretical analysis is provided without corresponding experimental data or visual evidence. It is suggested that the authors include quantitative comparisons or visualizations to more convincingly demonstrate the existence and impact of this phenomenon.

4. The experiments employ a 15% attention sparsity ratio, yet the rationale or tuning process behind this choice is not explained. The current setup appears to be empirical and lacks systematic hyperparameter analysis. The authors are advised to clarify the basis for selecting this sparsity level and to provide comparative experiments, such as results at 10%, 20%, and 30% sparsity levels, showing the trade-off between model performance and inference acceleration in the main text or appendix to strengthen the persuasiveness of the results.

### Questions
For more details, see weakness

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
3

### Summary
In this work, QuantSparse is proposed for compressing video diffusion transformers, with both attention sparsification and quantization applied. The authors point out that, when both techniques are naïvely applied, the error introduced by quantization will induce attention shift, which negatively impacts model performance. In QuantSparse, MSAD is used to efficiently perform attention distillation, leveraging both global and local guidance, and SSAR is proposed to exploit the temporal stability of the second-order residual to enhance sparse attention. The results show that diffusion transformers compressed by the proposed method can achieve SOTA performance while reducing both storage and latency simultaneously.

### Strengths
- The paper provides a detailed analysis of the problem caused by the combined use of attention sparsity and quantization. The proposed method is rational, easy to implement, and has low overhead, while demonstrating promising performance.
- The results are impressive. It improves the inference efficiency of SOTA video generation models with minimal degradation in generation quality.

### Weaknesses
- There are some clarity issues in the paper. Some symbols are reused for different hyperparameters. For example, the pooling stride is denoted as $s$ in Eq. (6) and the SVD rank is denoted as $r$ in Section 4.3, but the pooling stride becomes $r$ in Section H. See also the questions below.

### Questions
- Is the proposed method applicable to other transformer models that are not designed for video generation?
- How does the attention distillation perform when using full attention as the target?
- Could you clarify the calculation of the saliency? In Eq. (7), it is written as $s_{i}=\sum_{j}A_{j, i}$, but in the figure 2 it is $\sum_{h, i}A_{h, i, j}$.
- Given the large pooling stride used, what are the attention maps resolution? How do they impact the distillation time?

### Soundness
3

### Presentation
3

### Contribution
3
