# SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. Interestingly, we find that attention weights can be decoupled into two matrices: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (**S**parse-**L**inear **A**ttention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible, applying $\mathcal{O}(N^2)$ attention to critical weights, $\mathcal{O}(N)$ attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a $\textbf{20x}$ reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by $\textbf{95}$\% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a $\textbf{13.7x}$ speedup in attention computation and a $\textbf{2.2x}$ end-to-end speedup in video generation on Wan2.1-1.3B. The code is available at https://github.com/thu-ml/SLA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed SLA, a sparse-linear attention for efficient diffusion transformer generation. SLA finds that the attention weights can be decoupled into two types: a small fraction of large weights with high rank and the remaining weights with very low rank. SLA further designed a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. Experiments shown that SLA reduces attention computation by 95% without degrading end-to-end generation quality, with 2.2 times speedup for video generation.

### Strengths
1.SLA designed efficient forward and backward propagation modules, which are very complete and valuable in engineering.

2.It is very reasonable and meaningful for SLA to decompose attention into two computing modes, and achieve advanced performance.

3.SLA has great advantages in efficiency and has achieved state-of-the-art acceleration performance.

### Weaknesses
1.Is the attention mode mentioned in SLA commonly present in existing advanced video generation models? Reporting more models such as CogVideoX's results can further demonstrate the universality of this phenomenon.

2.How is Eq3 calculated specifically? For example, directly calculating the average value or calculating the rank? If it is the average, is there a direct correlation between this and the rank mentioned before?

3.Does the implementation of SLA distinguish between model architectures, such as Wan2.1 using DiT architecture, while other models such as HunyuanVideo and CogVideoX use MM-DiT architecture? Will this lead to differences in attention calculation methods and implementation?

4.It is meaningful to conduct additional supplementary experiments on other architectures such as CogVideoX-2B based on MM-DiT architecture.

5.The paper should report the resources consumed during training, such as the number of GPUs and time. And exploring more efficient fine-tuning paradigms such as LoRA can further enhance the practicality of SLA.

### Questions
Please see above weaknesses.

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
4

### Summary
This paper proposes Sparse–Linear Attention (SLA), a trainable hybrid attention mechanism for Diffusion Transformers (DiTs), particularly targeting video generation models. The core observation is that attention weight matrices can be decomposed into a small set of high-rank, high-magnitude entries and a large set of low-rank, low-magnitude entries. SLA classifies block-wise attention weights into critical, marginal, and negligible categories, applying standard sparse FlashAttention to the critical blocks, linear attention to the marginal blocks, and skipping negligible blocks. The authors implement a fused GPU kernel covering both forward and backward passes and demonstrate—on Wan2.1-1.3B video DiT and LightningDiT image models—that SLA achieves up to 95 % sparsity, 13.7× attention kernel speedup, and 2.2× end-to-end generation speedup with minimal quality degradation.

### Strengths
* Originality: The sparse/linear decomposition of attention weights for DiTs is a compelling insight. Rather than naively combining sparse and linear attention, SLA’s dynamic block classification and fused kernel design show genuine innovation tailored to diffusion workloads, pushing beyond previous sparse-only or linear-only acceleration schemes.
* Quality: The methodology is carefully engineered: Algorithm 1 and 2 detail the fused forward/backward passes; precomputation, lookup tables, and Method-of-Four-Russians optimizations indicate a strong systems perspective. Experimental evidence spans both video (Wan2.1) and image (LightningDiT) domains, with comprehensive metrics including VBench dimensions, VR, VA/VT, FID, FLOPs, sparsity, and latency.
* Clarity: The paper is generally well written. Figures 1–6 effectively motivate the decomposition, illustrate quality impacts, and benchmark performance. The block-level notation and discussion of online softmax, pooling, and projection clarify the mechanism.
Significance: Attention cost is the dominant bottleneck in DiTs, especially for long video sequences (10K–100K tokens). Demonstrating a 20× reduction in attention FLOPs with negligible quality loss is highly impactful for scalable diffusion-based video generation. The approach could influence both research and production systems.

### Weaknesses
* Dependence on fine-tuning: The method relies on retraining/fine-tuning to adapt models to SLA. While 2K steps are claimed to be minor, the paper provides limited analysis of how SLA behaves in zero-shot replacement or under constrained finetuning budgets (e.g., small datasets, limited steps). A sensitivity study on fine-tuning duration and data diversity would strengthen the practical narrative.
* Generalization beyond the tested architectures: Experiments focus on Wan2.1-1.3B and LightningDiT. Both employ similar attention layouts. It remains unclear whether SLA’s block classification and kernel fusion generalize to other DiT variants (e.g., different block sizes, mixed precision, cross-attention layers, text-video architectures). More diversity in model families (or a detailed discussion of expected behavior) would be valuable.
* Ablation breadth: While Table 2 covers fusion, activation, and kh variations, the key hyper-parameter pair (kh, kl) and block size (bq, bkv) are not jointly explored. Similar attention heads with differing dimensions may require different thresholds; without a deeper ablation, reproducibility across setups could be challenging.
* Evaluation scope: The paper reports strong aggregate metrics but lacks qualitative failure analysis. For instance, where does SLA still degrade outputs (e.g., motion consistency over long horizons, rare events)? Additionally, comparisons to other hybrid approaches (if any exist) or to recent structured sparsity methods in diffusion (e.g., block-diagonal or strided patterns) are absent.
* Kernel integration specifics: Although Appendix A.3 sketches optimization tricks, concrete profiling (e.g., breakdown of compute vs. memory bottlenecks, impact on backward pass peak memory) is missing. Readers aiming to reimplement SLA might struggle without more hardware-level detail.

### Questions
* Fine-tuning sensitivity: How does SLA perform if fine-tuning resources are restricted (fewer steps, smaller batch, or lower-resolution data)? Could you provide a curve illustrating quality vs. fine-tuning budget to justify the “few steps” claim quantitatively?
* Mask prediction robustness: The compressed mask Mc relies on pooled Q/K projections. How sensitive is SLA to inaccuracies in Pc during inference, especially when encountering distribution shifts (e.g., different video domains)? Have you evaluated adaptive thresholds or confidence measures?
* Backward pass efficiency: The backward kernel reportedly yields a 6.8× speedup, but details are sparse. Can you share profiling numbers on gradient computation vs. memory traffic? Are there cases where the backward pass becomes the new bottleneck?
Model generality: Have you attempted SLA on cross-attention or text-conditioned video generation models? If not, what modifications would be necessary to extend SLA to multi-head, multi-source attention patterns?
* Failure cases: Could you provide examples or statistics of prompts where SLA underperforms full attention (e.g., higher error metrics, visual artifacts)? Understanding the boundaries of applicability would help practitioners adopt the method safely.
* Hyper-parameter tuning: For practitioners, how should kh and kl be selected for a new model? Do you recommend starting from a certain sparsity target (e.g., 95%) and adjusting based on validation loss, or is there a principled heuristic derived from attention statistics?

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
4

### Summary
This paper addresses the quadratic attention bottleneck in Diffusion Transformers (DiTs), particularly for video generation where sequence lengths are large. Based on the key observation that DiT attention weights can be decomposed into a small high-rank component (a few large weights) and a large, extremely low-rank component (many small weights), the authors propose Sparse-Linear Attention (SLA). SLA is a trainable, hybrid mechanism that classifies attention blocks into three categories: 1)Critical: Computed with $O(N^2)$ sparse attention. 2) Marginal: Computed with $O(N)$ linear attention. 3)Negligible: Skipped entirely.By fusing these operations into an efficient GPU kernel and applying minimal fine-tuning, SLA achieves a 95% reduction in $O(N^2)$ attention computation. This results in a 13.7x kernel speedup and a 2.2x end-to-end video generation speedup on the Wan2.1-1.3B model, all while maintaining generation quality.

### Strengths
1. The core observation decoupling attention weights into a sparse/high-rank component and a dense/low-rank component is novel and insightful. It clearly explains the limitations of using only sparse or only linear attention and provides a strong foundation for the proposed hybrid approach
2. The proposed SLA method, which dynamically assigns different computational complexities ($O(N^2)$, $O(N)$, or $O(0)$) based on a block's predicted importance, is an elegant and highly effective solution.
3. The paper demonstrates truly significant gains. A 95% reduction in expensive $O(N^2)$ computation translating to a 13.7x kernel speedup and a 2.2x end-to-end speedup with no loss in quality is a major contribution for efficient DiTs

### Weaknesses
1. The "95% sparsity" figure is confusing. Based on the hyperparameters ($k_h$=5%, $k_l$=10%), this 95% refers to blocks that avoid $O(N^2)$ computation. It does not mean 95% of blocks are skipped. In fact, 85% (100% - 5% - 10%) are still processed by $O(N)$ linear attention. This terminology should be clarified
2. All latency benchmarks (e.g., 13.7x speedup) are reported on an RTX 5090 GPU. While impressive, providing benchmarks on more widely accessible hardware  would greatly strengthen the paper's practical and immediate value to the research community.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work adopts the linear attention for the intermediate stage between the sparse attention and full (dense) attention.

This work first adopts the pooling to $Q$ and $K$ to get the sparse mask, then apply the full attention to the important area, skip the unimportant area, and adopt linear attention to those intermediate important area.

This work adopt the fine-tuning for the new introduced linear attention weights, which I think is the main contribution of this work. This work makes a lot effort on the forward and backward implementation of the linear attention for the insert to the original full attention and sparse attention technique.

according to the results, this work achieves good results.

### Strengths
1. This work makes a lot of effort for the implementation of the linear attention - the forward and backward - during the fine-tuning.

### Weaknesses
1. The equation (2) which adopts the mean pooling to identify the attention importance is the same as the work [1] which also use the mean pooling to identify the attention importance for the acceleration of video generation. Thus, I think the novelty is limited to this work.

2. This work does not provide any similarity metric like PSNR, SSIM, and LPIPS in their results, then the comparison to the work [1] is not direct. As the work [1] also use the sparse attention for the acceleration, and achieves 90% sparsity, so the exact performance improvement from the linear attention remains unknown. 

3. This work focus more on the new CUDA kernel designs rather than some algorithm-level novelty.

---

[1] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

### Questions
1. This work may focus on the contribution of the implementation of the forward and backward for the insert of linear attention to the full attention and sparse attention, then I think the novelty is limited.

2. Can the authors provide the similarity metric like PSNR, SSIM, and LPIPS to compare to other works like [1] [2]. Especially, compare to the work [1]. Theoretically, the linear attention is slower than the skip (i.e., the sparse attention) and this work takes a lot of training effort for fine-tuning, authors need to show that the linear attention exactly improve much performance than the pure sparse attention.

3. According to the result in work [1] (Table 1), the 90% sparsity ratio does not lead to much accuracy drop on image quality, while the result in Table 2 of this work show that the 'Sparse Only' result shows a large accuracy drop on metric like 'Image Quality'. Can the author explain this?

4. For this work, which adopts the linear attention for the intermediate stage between full and sparse attention, I would like to ask for the results in low-resolution. I think it may be enough to adopt the sparse attention for acceleration in low-resolution video generation, as there may not be so many 'Marginal' blocks in the low-resolution video generation process.

5. I want to further ask the authors about: the fine-tuning results for the pure sparse attention technique with downsampled attention (i.e., equation (2) in this paper). I think it is quite important ablation to show the effectiveness of the linear attention under the same training effort. I believe if apply same fine-tuning effort to pure sparse attention, the results will be good.

---
[1] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

[2] Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity

### Soundness
2

### Presentation
2

### Contribution
2
