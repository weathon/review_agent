# Cross-Timestep: 3D Diffusion Model with Trans-temporal Memory LSTM and Adaptive Priori Decoding Strategy for Medical Segmentation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Diffusion models have recently demonstrated significant robustness in medical image segmentation, effectively accommodating variations across different imaging styles. However, their applications remain limited due to: (i) current successes being primarily confined to 2D segmentation tasks—we observe that diffusion models tend to collapse at the early stage when applied to 3D medical tasks; and (ii) the inherently isolated iteration along timesteps during training and inference. To tackle these limitations, we propose a novel framework named Cross-Timestep, which incorporates two key innovations: an Adaptive Priori Decoding Strategy (APDS) and a trans-temporal memory LSTM (tLSTM) mechanism. (i) The APDS provides prior guidance during the diffusion process by employing a Priori Decoder(PD) that focuses solely on the conditional branch, successfully stabilizing the reverse diffusion process. (ii) The tLSTM integrates convolution and linear layers into the LSTM gating structure, and enhances the memory cell mechanism to retain temporal state, explicitly preserving and propagating continuous temporal states across timesteps. Experimental results demonstrate that Cross-Timestep performs favorably on heterogeneous 3D medical datasets. Three experiments further analyze the collapse phenomenon in 3D medical diffusion models and validate that APDS effectively prevents initial-stage collapse without excessively constraining the model, while tLSTM facilitates the performance and scalability of diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a 3D diffusion framework for medical image segmentation that addresses the instability of conventional diffusion models when applied to volumetric data. It introduces two key components: the Adaptive Priori Decoding Strategy (APDS), which provides structural guidance to prevent early-stage collapse during high-noise timesteps, and the trans-temporal memory LSTM (tLSTM), which preserves temporal information across diffusion steps for consistent refinement. Together, these innovations stabilize the reverse diffusion process and improve segmentation accuracy across heterogeneous datasets, achieving state-of-the-art results and demonstrating strong robustness under domain shifts.

### Strengths
Enhanced temporal coherence: The trans-temporal memory LSTM (tLSTM) explicitly retains and propagates structural and contextual information across diffusion timesteps, turning each step into a coherent continuation rather than an independent reconstruction.

Improved stability under high noise: By combining tLSTM with the Adaptive Priori Decoding Strategy (APDS), the model accumulates temporal evidence effectively, preventing early-stage collapse and ensuring reliable recovery from highly noisy initial states.

### Weaknesses
What is the reason for the initial-stage collapse? Why are 2D models utilized in a 3D scenario?
Some methods, such as Diff-UNet, which use 3D diffusion models, can avoid the initial-stage collapse — is that correct?
Sometimes, the authors describe their methods in a confusing and complex way. For example, what does “explicit temporal evidence accumulation” mean?
There are no diffusion-based methods included in the comparison experiments — why?

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Cross-Timestep, a novel 3D diffusion framework designed to address the challenge of initial-stage collapse in 3D medical image segmentation. The authors propose two key components: 1) Adaptive Priori Decoding Strategy (APDS), which provides time-weighted structural guidance during high-noise stages of the diffusion process; 2) Trans-temporal memory LSTM (tLSTM), a recurrent module that maintains temporal state across denoising steps to ensure coherence and stability. The method is evaluated on two multi-center 3D medical imaging datasets (LNCTVSeg and OAseg) and demonstrates state-of-the-art performance in terms of Dice, IoU, and HD95 metrics. The paper also includes extensive ablation studies and visualizations to validate the contributions of each component.

### Strengths
1. **Significance:** The paper identifies and addresses a critical failure mode (initial-stage collapse) in 3D diffusion models, which has been largely overlooked in prior work.
2. **Technical Rigor:** The proposed APDS and tLSTM modules are carefully designed and thoroughly evaluated. Multiple variants of tLSTM (Conv-tLSTM, Linear-tGRU, SC-tLSTM, FFT-tLSTM) are introduced, showcasing flexibility and scalability.
3. **Comprehensive Evaluation:** Experiments are conducted on two challenging multi-center datasets with significant domain shifts. The method outperforms several strong baselines and is supported by both quantitative and qualitative results.
4. **Reproducibility:** The paper includes detailed methodology, ablation studies, and references to publicly available datasets and code, enhancing reproducibility.

### Weaknesses
1. **Computational Cost:** While the authors propose Linear-tGRU to reduce computational demands, the overall framework (especially with 3D convolutions and recurrent units) is likely still resource-intensive. A more detailed analysis of training/inference time and memory usage would be helpful.
2. The design of the time-weighting function $\omega_t$ in APDS is heuristic. While it works well, a more systematic analysis or learning-based approach for $\omega_t$ could further strengthen the method.

### Questions
1. How does the performance of Cross-Timestep scale with the number of diffusion steps? Is there a trade-off between segmentation accuracy and inference speed?
2. Could the tLSTM mechanism be adapted to other generative or discriminative tasks beyond segmentation, such as 3D image synthesis or reconstruction?
3. The APDS relies on a prior decoder trained only on the conditional branch. How sensitive is the model to the quality of this prior? What happens if the prior is noisy or inaccurate?
4. Have the authors considered using learned or adaptive schedules for $\omega_t$ instead of the fixed exponential decay? Could this further improve performance?
5. The paper mentions that APDS prevents over-interference in later stages. Is there a risk of under-guiding in early stages if the prior is too weak?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Cross-Timestep, a novel diffusion model framework for 3D medical image segmentation. The work aims to resolve the "initial-stage collapse" issue frequently observed in 3D diffusion models during early denoising steps. To this end, the authors introduce two main components: a Trans-Temporal Memory LSTM (tLSTM) to accumulate structural information across diffusion time steps, and an Adaptive Priori Decoding Strategy (APDS) to stabilize the reverse process using a weighted prior. Although the method achieves competitive performance on some medical datasets, the overall framework is overly complex, lacks adequate demonstration of flexibility and generalizability, and fails to provide sufficient experimental analysis and key visualizations to justify its high design and computational cost.

### Strengths
1. The paper addresses a practical and significant pain point in applying 3D diffusion models to the medical domain, specifically the "initial-stage collapse," which is a valuable research direction in volumetric medical data processing.

2. The combination of tLSTM and APDS reflects a novel approach to addressing time-step dependency and denoising stability. Conceptually, tLSTM, acting as a memory mechanism across time steps, has some inherent reasoning.

### Weaknesses
1. The proposed Cross-Timestep framework is excessively complex and potentially redundant. Integrating the tLSTM module inside the 3D U-Net significantly increases the model's parameter count and computational complexity, while also introducing multiple new hyperparameters. This over-engineered design reduces the framework's generality and flexibility for deployment.

2. Given the extremely high resource demands of the 3D U-Net itself, the additional memory and training time overhead introduced by the tLSTM module are not adequately and rigorously quantified or justified. The paper fails to clearly demonstrate that the performance gain is worth such a substantial increase in design complexity and computational cost.

3. The experimental results rely heavily on quantitative metrics like the Dice Score in tables. Crucial visual evidence and generalization analysis are lacking, especially a direct visualization of how tLSTM and APDS internally mitigate the "collapse" across early time steps, which weakens the conviction of the core claim.

4. Although an ablation study is provided, the discussion regarding the strict necessity of both tLSTM and APDS is not deep enough. The authors should explore whether simpler, lighter mechanisms could achieve similar stabilization effects to justify the complexity of the current design.

### Questions
1. Could the authors provide a detailed report on GPU memory usage and training time for the basic 3D diffusion model versus the full Cross-Timestep model to quantify the specific computational overhead introduced by the tLSTM module?

2. Considering the complexity of APDS and tLSTM, have the authors explored a more simplified and flexible design (e.g., using only APDS or a simpler cross-step attention mechanism)? How does the performance of these simpler variants compare to the full model?

3. Can the authors provide a more convincing visual analysis that clearly demonstrates how the memory information accumulated by tLSTM and the prior guided by APDS specifically suppress or reverse the "initial-stage collapse" during the early denoising time steps?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Cross-Timestep, a framework for 3D medical image segmentation using diffusion models. The method introduces two key innovations. First, an Adaptive Priori Decoding Strategy (APDS) generates a time-weighted prior mask to stabilize the model during the early, high-noise stages. Second, a trans-temporal memory unit (tLSTM) with variants (SC-tLSTM and FFT-tLSTM) is designed to overcome the isolated, step-by-step nature of the denoising process, allowing the model to accumulate evidence across timesteps. Experiments on two multi-center datasets confirm that APDS successfully mitigates “initial-stage collapse” and tLSTM improves the coherence of the denoising trajectory.

### Strengths
1. Identifying an Under-Explored Problem: The paper identifies and addresses the "initial-stage collapse" problem in 3D medical diffusion models. As demonstrated in Fig. 1, the authors show that standard 3D diffusion models fail when sampling from high-noise timesteps, but succeed when starting from mid-level noise. This finding renders the proposed Adaptive Priori Decoding Strategy (APDS) an intuitively appealing and well-motivated solution.
2. Achieving SOTA Performance: The paper thoughtfully combines effective components (SC-tLSTM and FFT-tLSTM) to create a powerful, stateful denoiser. This architecture achieves state-of-the-art performance on two multi-center 3D datasets (LNCTVSeg (CT) and OAseg (MRI)), demonstrating the framework's robustness and practical utility.

### Weaknesses
1. Limited Architectural Novelty: The proposed framework appears to be a complex integration of existing and well-established components, rather than a new algorithmic contribution. 
    + On the SC-tLSTM: The core idea of tLSTM is to apply a recurrent model (LSTM or GRU) to maintain a state across the diffusion timesteps. However, the combination of tLSTM and diffusion models tends to be inelegant. It forces an external, stateful memory (tLSTM) onto the diffusion model's fundamentally Markovian process, which seems unjustified from a theoretical standpoint. As demonstrated by prior work [1], LSTM-based U-Net backbones have already been explored and proven effective for 3D medical segmentation. The SC-tLSTM module itself appears to be a straightforward combination of the standard spatial-channel attention mechanism [2] and the tLSTM component.
    + On the FFT-tLSTM: The novelty of FFT-tLSTM is questionable. Leveraging the frequency domain (via FFT) within a diffusion model framework for medical image segmentation has already been published [3]. The proposed FFT-tLSTM thus appears to be a minor variation that simply inserts the tLSTM components into an established FFT pipeline.
    + On the APDS: The APDS uses a segmentation decoder that operates solely on the conditional image to generate a coarse structural prior. This concept of using the conditional input to create explicit guidance is a well-established technique in conditional diffusion models and thus cannot be regarded as a significant innovation.
The authors should more clearly highlight their components' algorithmic novelty over prior research. They should also strengthen the explanation of how the integration of these modules provides synergistic benefits that validate their effectiveness.

2. Insufficient Experimental Validation
    * Lack of Comparative Qualitative Visualizations: The paper's quantitative claims are not supported by adequate qualitative evidence. While Appendix G (Fig. 10) shows the proposed model's output, it fails to provide essential side-by-side visual comparisons against the SOTA baselines. Without these direct comparisons, it is impossible to qualitatively assess the model's claimed advantages.
    * Missing Computational Cost Analysis: The framework introduces significant architectural complexity (e.g., APDS, tLSTMs) without any analysis of the computational overhead. The authors should provide metrics on training/inference time and GPU memory usage. This information is crucial for evaluating the method's practical feasibility and understanding the trade-off between performance and efficiency.
    * Inadequate Comparison with State-of-the-Art Diffusion Models: The authors cite Diff-UNet [4] in the related work section, which makes their omission from the quantitative comparison a significant weakness. I strongly recommend the authors include comparisons against these recent diffusion-based models [4-6]. Such a comparison is essential to properly benchmark the "Cross-Timestep" framework and convincingly validate its claimed segmentation superiority.

3. Clarity and Writing Quality
    * Discrepancy in Abstract: The abstract explicitly claims: "Three real-world cases further analyze the collapse phenomenon...". However, these three specific case studies are not clearly presented in the main paper or appendices. 
    * Minor writing issues: There are duplicate headings ("2 Related Work" and "3 Related Work") , and scattered typos/formatting issues in Section 5 (e.g., the use of quotes around ’Diff Out’, ’APDS Out’ , ’Conv-tLSTM’, and ’Linear-tGRU’ ).

[1] Chen, Tianrun, et al. "xlstm-unet can be an effective 2d & 3d medical image segmentation backbone with vision-lstm (vil) better than its mamba counterpart." BHI 2024.

[2] Si, Yunzhong, et al. "SCSA: Exploring the synergistic effects between spatial and channel attention." Neurocomputing 634 (2025): 129866.

[3] Jiang, Yuxuan, et al. "Diff-sfct: A diffusion model with spatial-frequency cross transformer for medical image segmentation." BIBM 2023.

[4] Xing, Zhaohu, et al. "Diff-UNet: A diffusion embedded network for robust 3D medical image segmentation." Medical Image Analysis (2025): 103654.

[5] Chen, Tao, et al. "HiDiff: Hybrid diffusion framework for medical image segmentation." IEEE TMI (2024).

[6] Shuai, Zhihao, et al. "Diffseg: a segmentation model for skin lesions based on diffusion difference." arXiv preprint arXiv:2404.16474 (2024).

### Questions
1. Computational Cost: Could the authors provide a table comparing inference time and VRAM requirements for the proposed model and other baselines?
2. t-cell Explanation: Could the authors clarify what the "t-cell" in Table 1 refers to? How is it architecturally distinct from the Conv-tLSTM and Linear-tGRU modules?
3. Sensitivity of ω_t: The time-weight ω_t (Appendix A) has a rather complex formulation. How sensitive is the model's performance to this specific function? Did the authors experiment with simpler decay functions (e.g., linear, exponential decay)?

### Soundness
2

### Presentation
2

### Contribution
2
