# DVD-Quant: Data-free Video Diffusion Transformers Quantization

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Diffusion Transformers (DiTs) have emerged as the state-of-the-art architecture for video generation, yet their computational and memory demands hinder practical deployment. While post-training quantization (PTQ) presents a promising approach to accelerate Video DiT models, existing methods suffer from two critical limitations: (1) dependence on computation-heavy and inflexible calibration procedures, and (2) considerable performance deterioration after quantization.
To address these challenges, we propose DVD-Quant, a novel Data-free quantization framework for Video DiTs. Our approach integrates three key innovations:
(1) Bounded-init Grid Refinement (BGR) and 
(2) Auto-scaling Rotated Quantization (ARQ) for calibration data-free quantization error reduction, as well as
(3) $\delta$-Guided Bit Switching ($\delta$-GBS) for adaptive bit-width allocation.
Extensive experiments across multiple video generation benchmarks demonstrate that DVD-Quant achieves an approximately 2$\times$ speedup over full-precision baselines on  advanced DiT models while maintaining visual fidelity. Notably, DVD-Quant is the first to enable W4A4 PTQ for Video DiTs without compromising video quality. Code and models will be released to facilitate future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper focuses on post-training calibration-free quantization of diffusion models under low-bit width settings (w4a4, w4a6, w4a8). The paper proposes bounded-init grid refinement to minimize the **reconstruction error on weight matrices** using **asymmetric quantization**. The paper proposes auto-scale rotated quantization to deal with the outlier and high variance of the activation. The method first rotate weight and activation using hadamard matrix, and rescale activation on the fly during inference, **without fusing the diagonal scaling factor into weight matrices**. Finally, the paper proposes mixed precision quantization for activation, by using higher bitwidth when error accumulates to a certain threshold. Experiments are conducted on several video diffusion models and benchmarks, comparing the proposed method against standard PTQ baselines. The results claim improved perceptual quality and reduced quantization degradation at extremely low bitwidths, demonstrating that the proposed calibration-free approach can maintain generation fidelity.

### Strengths
1. **Calibration-free quantization.** The proposed method does not require any calibration data to do quantization, which reduces deployment time and makes the method more convenient to use.
2. **Handling of activation/weight distribution.** The paper observes the distribution of weight, and tries to use optimization based quantization method instead of vannila max-scaling based method on weight quantization. For activations, the paper observes their high variance and uses online-scaling to increase quantization accuracy.
3. **Detailed comparison on quantization error with baselines.** The paper conducts experiments on a great number of model, benchmarks and provided sufficient ablation study to support its claim.

### Weaknesses
**The proposed method is not possible to achieve real world end-to-end acceleration.** Although Table 5 in the main paper shows end-to-end acceleration on w4a4, w4a6 and w4a8 setting on RTX4090 of HuyuanVideo, **I, as an experienced kernel programmer with rich experience in quantization, doubt its soundness**. Detail is as follows:

1. **BGR introduces asymmetric quantization.** This is a small problem. Compared with symmetric quantization, asymmetric quantization requires calculating the token sum of activation before GEMM, and use an epilogue to include the contribution of zero point. This will incur non-negligible overhead if not dealt carefully. The author are encouraged to clarify the detailed implmentation of asymmetric quantization and how the quantized GEMM functions under this asymmetric configuration.
2. **ARQ makes it not possible to use low-bit Tensor Core.** This is a big problem. As demonstrated in section 3.2, the author uses
$$
\widehat{\mathbf{X}}=\mathcal{Q}\left(\mathbf{X} \mathbf{H} \mathbf{\Lambda}^{-\mathbf{1}}\right), \quad \widehat{\mathbf{W}}=\mathcal{B} \mathcal{G} \mathcal{R}(\mathbf{W} \mathbf{H}), \quad \mathbf{Y}=\widehat{\mathbf{X}} \mathbf{\Lambda} \widehat{\mathbf{W}}^{\top}
$$
to calculate activation-weight GEMM, where $\mathbf{\Lambda}$ is the online calculated scaling factor. With a diagonal scaling factor between activation and weight, I cannot think how low bit tensor core can be used to get acceleration. As stated in SmoothQuant[1], this is **the per-channel activation quantization that is not compatible with INT GEMM kernel**. **So I am very curious how the author solves this problem** and get the latency reduction result (in particular how to leverage the low bit tensor core to accelerate GEMM).
3. **Claim of using widely adopted kernels**. In section 4.4, the author claims "This allows direct use of the widely adopted W4A4 and W4A8 GEMM kernels, eliminating the need to design dedicated mixed-precision kernels". I personally do not know any open-source kernel that can realize the proposed ARQ method and asymmetric weight quantization (I personally do not think this is realizable with improved efficiency), so I would ask the author how (and which) common kernels can be directly adopted.
4. **Weird speed result.** In section 4.4,  the author report the speed up of w4a4, w4a6, w4a8, with w4a4 being fastest, followed by w4a6 and finally w4a8. w4a6 and w4a8 should in theory leverage the INT8 tensor core (since there is not 6 bit tensor core on RTX4090 on which the author does all experiment). However, w4a6 requires dequantizing both weight and activation to 8bit, while w4a8 only needs to dequantize the weight. So **w4a6 should in principle be slower than w4a8**, and I wonder why the result presented paper shows the opposite conclusion.

In summary, while the paper presents an interesting set of ideas for data-free quantization, the core technical claims regarding efficiency and deployability are highly questionable, which significantly limits its practical value. 

[1] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han.

### Questions
Already stated in weakness. I would list the questions here again for completeness.

1. **Asymmetric Quantization Implementation**: How is asymmetric quantization implemented in practice? Specifically, what is the computation flow and corresponding kernel? What is the overhead compared with symmetric quantization?
2. **Tensor Core Utilization under ARQ**: Given that the proposed ARQ introduces a diagonal scaling factor between activation and weight, how can the method leverage low-bit Tensor Cores for acceleration? Please give a detailed formulation.
3. **Use of “Widely Adopted Kernels”**: In Section 4.4, the author claim that the method allows direct use of “widely adopted W4A4 and W4A8 GEMM kernels.” Could you specify which existing kernels (and how these kernels are modified to) support your asymmetric quantization and ARQ design?
4. **Speedup Inconsistency**: The reported latency results in Table 5 show w4a4 being the fastest, followed by w4a6 and then w4a8. On RTX 4090 (on which all the experiment are conducted claimed by the author), which lacks 6-bit Tensor Core support, w4a6 should require additional dequantization compared with w4a8, implying slower execution. Could you explain this discrepancy and clarify how these results were obtained?

If the author can provide satisfactory explanation to the problems above, I would raise my score.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes DVD-Quant, a novel data-free post-training quantization (PTQ) framework designed to reduce the computational and memory costs of Video Diffusion Transformers (Video DiTs). The method introduces three core techniques: 1) Bounded-init Grid Refinement (BGR) for more accurately quantizing Gaussian-like weight distributions, 2) Auto-scaling Rotated Quantization (ARQ) which uses online Hadamard rotation and scaling to handle activation outliers without a calibration dataset, and 3) δ-Guided Bit Switching (δ-GBS) which dynamically allocates higher or lower bit-widths to activations at different denoising timesteps based on feature change. The authors demonstrate that DVD-Quant achieves a significant speedup (up to ~2x, and 4.85x when combined with caching) and is the first method to successfully enable W4A4 (4-bit weights and activations) quantization for Video DiTs without catastrophic failure, maintaining performance close to the full-precision baseline.

### Strengths
1. Successfully enabling W4A4 PTQ for complex Video DiT models is a notable achievement, as this extreme quantization level typically causes existing methods to fail completely.

2. Comprehensive and Synergistic Approach: The three proposed components (BGR, ARQ, δ-GBS) address distinct and well-motivated challenges (weight distribution, activation outliers, temporal redundancy) and are shown to work effectively together.

3. Strong Empirical Validation: The paper includes extensive experiments on established models (HunyuanVideo) and benchmarks (VBench), showing clear quantitative and qualitative superiority over several strong baselines across multiple bit-width settings. The ablation studies effectively demonstrate the contribution of each component.

### Weaknesses
1. Hyperparameter Sensitivity: The performance of the adaptive δ-GBS mechanism depends on a threshold δ. While mentioned, the paper does not deeply explore the sensitivity of the results to this value or provide a robust method for selecting it across different models or tasks.

2. Limited Model Scope: While tested on HunyuanVideo and briefly on Wan2.1, it's unclear how generalizable the method is to the wider family of DiT-based models (e.g., Latte, Sora's architecture) or other diffusion tasks (e.g., image generation). The claim of being "the first" is strong but might be limited to the specific models tested.

3. Computational Overhead of ARQ: Although the Hadamard transform is described as having "marginal overhead," this is not quantified. For a method focused on acceleration, a more detailed analysis of the latency/throughput trade-off introduced by the online rotation and scaling would be beneficial.

### Questions
N/A

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
This paper focuses on post-training quantization (PTQ) for text-to-video generation models and proposes three key techniques **weight refinement**, **rotation-aware transformation**, and **big switching area** to enhance quantization performance.

### Strengths
1. The motivation behind the proposed BGR method is clear and intuitively appealing.
2. The paper is well written, with clear explanations and easy-to-follow reasoning.

### Weaknesses
1. While rotation techniques have been extensively applied in LLM quantization, the paper only discusses QuaRot and lacks a broader comparison or discussion with several relevant works such as **DuQuant**, **RoSTE**, and **ResQ**.
2. The proposed auto-scaling rotation mechanism appears similar to the approach adopted in DuQuant, which combines SmoothQuant with rotation—this overlap should be clarified.
3. The paper lacks comparisons with several state-of-the-art baselines, such as **SVDQuant**.
4. The evaluation is limited to a single benchmark and one model, which weakens the generalization claims.

### Questions
1. Could you provide additional experimental results across multiple benchmarks and different text-to-video models to verify the generality of your method?
2. Could you offer more details about the latency measurements (e.g., batch size, sequence length, and hardware configuration)?
3. I am curious about the runtime efficiency of the proposed online rotation and scaling operations. Could you include more details or analysis to clarify how these affect the overall speedup and computational cost?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes DVD-Quant which is a deployment-friendly quantization framework for Diffusion Transformers (DiTs). The paper identifies three key properties of large-scale Video DiTs—Gaussian-like weight distributions, substantial activation scale discrepancies across denoising timesteps, and timestep-varying latent features—and introduces three corresponding techniques: Bounded-init Grid Refinement (BGR) for weights, which iteratively refines the quantization scale and zero-point with tightening bounds to better fit Gaussian-like distributions; Auto-scaling Rotated Quantization (ARQ) for activations, a calibration-free method combining online scaling with Hadamard rotation to handle timestep-variant scales and reduce quantization error; and δ-Guided Bit Switching, a temporal mixed-precision mechanism that adaptively assigns activation bit-widths per timestep with negligible inference overhead. Together, these components narrow the accuracy–deployment gap in DiT compression, enabling robust low-bit quantization without costly calibration or retraining.

### Strengths
- A systematic analysis reveals three key insights, motivating the following solutions to quantization. These finds are valuable for the future research.
- Strong low-bit performance without retraining is achieved
- Broad applicability to video DiTs: Designed around Video DiT characteristics (e.g., temporal variations), making it more suitable than generic PTQ baselines for large-scale video generation models.
- Modular and complementary components: Each module targets a distinct bottleneck (weights, activations, temporal allocation), allowing flexible adoption and integration with existing pipelines.

### Weaknesses
Although the better accuracy–deployability trade-off is achieved, I remain to wonder
- Generalization beyond Video DiTs (Hunyuan): The framework leverages timestep dynamics typical of Video DiTs. It is unclear how well it transfers to other generative models, like Wanx or even image generator.
- Effect of various details: Choices like rotation block size, scaling granularity (per-channel vs per-tensor), and quantization granularity (weight group size) can materially affect outcomes; the method may need careful tuning per model size and dataset to reach reported gains. More analysis on those hyperparameters will help the audience follow.

### Questions
Please check Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
