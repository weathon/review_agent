# Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Diffusion models have transformed image synthesis by establishing unprecedented quality and creativity benchmarks. Nevertheless, their large-scale deployment faces challenges due to computationally intensive iterative denoising processes. Although post-training quantization (PTQ) provides an effective pathway for accelerating sampling, the iterative nature of diffusion models causes stepwise quantization errors to accumulate progressively during generation, inevitably compromising output fidelity. To address this challenge, we develop a theoretical framework that mathematically formulates error propagation in Diffusion Models (DMs), deriving per-step quantization error propagation equations and establishing the first closed-form solution for cumulative error. Building on this theoretical foundation, we propose a timestep-aware cumulative error compensation scheme. Extensive experiments on multiple image datasets demonstrate that our compensation strategy effectively mitigates error propagation, significantly enhancing existing PTQ methods. Specifically, it achieves a 1.2 PSNR improvement over SVDQuant on SDXL W4A4, while incurring only an additional $<$ 0.5\% time overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Timestep-aware Cumulative Error Compensation (TCEC), a method designed to mitigate quantization error accumulation across timesteps in diffusion models. TCEC introduces a timestep-aware error compensation mechanism using a timestep-conditioned channel-wise scaling matrix K. Since the correction term K is derived directly from the DDIM sampling equation, it does not require backpropagation-based training and can instead be computed using a small calibration dataset. By applying this compensation mechanism on top of quantization techniques such as SVDQuant and ViDiT-Q, the proposed TCEC consistently achieves improvement in generation quality compared to models quantized solely with these quantization techniques.

### Strengths
- The proposed TCEC complements rather than replaces existing quantization techniques (e.g. SVDQuant and ViDiT-Q), so it has the potential to benefit a wide range of models and quantization techniques.

### Weaknesses
- While this paper proposes a quantization error compensation mechanism, it lacks a thorough comparison with previous error compensation methods [1, 2] and only briefly mentions them at the end of Section 2.2. The paper claims that prior approaches lack rigorous theoretical analysis (line 161 of the paper), but these works actually provide concrete and well-grounded derivations. Furthermore, the claim that “their experimental designs rely on ad-hoc fitting to specific tasks and are only validated on simple datasets such as CIFAR-10” (line 186 of the paper) is difficult to agree with. Prior studies [1, 2] demonstrate error compensation using lightweight calibration—similar to the method proposed in this paper—and report results on more complex datasets, including ImageNet, LSUN-Bed, and even text-guided image synthesis with Stable Diffusion.
More importantly, the core concept of TCEC, which estimates accumulated error and corrects it across timesteps, closely aligns with [2], yet this connection is not explicitly acknowledged. This raises concerns regarding the originality and novelty of the work.


- The paper presents an unnecessarily complex derivation that ultimately leads to an intuitive conclusion that compensating quantization error at each timestep can mitigate cumulative error. This excess complexity may confuse readers.
 
- There is a mismatch between the experimental setups for image/video quality and latency evaluations. The proposed error compensation term (Eq. 18) is derived for DDIM/DPM++ solvers under a noise scheduler, and image/video quality in Tables 1 and 2 is evaluated using diffusion models with these solvers. However, the inference latency is measured on Flux.1-dev, which is a flow-matching model. Thus, the reported latency comparison is inconsistent with the theoretical formulation. 

[1] Huanpeng Chu, Wei Wu, Chengjie Zang, and Kun Yuan. “Qncd: Quantization noise correction for diffusion models.” ACM International Conference on Multimedia, 2024

[2] Yuzhe Yao, Feng Tian, Jun Chen, Haonan Lin, Guang Dai, Yong Liu, and JingdongWang. “Timestep-aware correction for quantized diffusion models.” In European Conference on Computer Vision, 2024

### Questions
Please check weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TCEC is a novel method designed to mitigate error propagation across successive denoising steps in quantized diffusion models.
The paper introduces a theoretical framework that models how per-step quantization errors accumulate over time, deriving the first closed-form solution for this propagation and simplifying it through practical approximations.
At inference, TCEC injects a lightweight correction term to counteract this accumulated error on the fly.
As a result, it significantly improves the performance of low-precision diffusion models and is compatible with a wide range of quantization approaches.

### Strengths
1. The work provides the first rigorous analysis of how the quantization errors propagate through sampling steps.

2. The work derives a closed-form solution for cumulative error and simplifies it into a practical, efficient correction mechanism that requires no model retraining. As a result, TCEC is model-agnostic and can be seamlessly integrated with existing post-training quantization methods.

3. The paper presents rich analysis, including both theoretical derivations and empirical ablations that support the method’s design choices.

4. The writing is clear and the structure is well-organized, making the paper easy to follow overall.

### Weaknesses
1. **Assumption of Uniform Error Distribution:**
The paper assumes that quantization error is uniformly distributed across the tensor, but in practice, quantization is typically applied channel-wise or block-wise. As a result, the error pattern is more structured and non-random. For example, channels with larger value ranges are likely to experience greater quantization error due to coarser quantization scales. This suggests the theoretical model may oversimplify how real quantization noise behaves in practice.

2. **Limited and Incomplete Experimental Coverage:**
Several key experiments are missing or underdeveloped. For instance, Table 1 omits ViDiT-Q + TCEC results for image diffusion (despite its presence in Table 2 for video), without explanation. More critically, comparisons against error-aware baselines like QNCD or [1] are missing. This is a notable omission given the shared goal of runtime error correction.
The paper also lacks evaluation across a broader range of conditions (e.g., varying sampling steps per model) and provides only one latency example in the main text. Further ablations on hyperparameters (such as the sliding window size m, beyond Table 5) would help understand sensitivity.

3. **Marginal Gains in Image Models:**
While TCEC yields measurable accuracy gains in video diffusion models, the improvements in image generation tasks appear marginal. This raises questions about the general effectiveness of the method across domains. Additionally, Table 5 shows that the difference between Eq. (10) (exact form) and Eq. (13) (practical approximation with m = 1) is small, suggesting that the simplified formulation may not offer substantial gains — and setting m = 1 might not be a sufficiently accurate approximation in some settings.


[1] Timestep-Aware Correction for Quantized Diffusion Models, Yuzhe Yao et al., ECCV 2024.

### Questions
Mainly listed in the weakness. Below are the additional questions.

1. At line 214, why is it valid to assume that the input x_t​ remains unchanged, even though quantization error from the previous step should propagate and alter the input over time? A more detailed justification would help clarify this approximation.

2. I suggest adding the latency breakdown from Table 6 into the main body of the paper for better visibility and clarity.

3. Minor typos:
(i) “protal” → should be “portal” in Figure 2 caption.
(ii) “SVDqunat” → should be “SVDQuant” in the reference title.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the accumulation of quantization errors during the denoising process. The proposed **TCEC** (Timestep-aware Cumulative Error Compensation) is a lightweight module that dynamically corrects for cumulative quantization errors during the diffusion sampling process. It models error propagation, derives a closed-form solution for the cumulative error, and then uses a timestep-aware online estimation to apply a correction term with minimal (<0.5%) overhead. TCEC consistently boosts the performance of underlying PTQ methods across multiple models (SDXL, PixArt, OpenSORA), tasks (image, video), and metrics (FID, PSNR, VBench)

### Strengths
1. The authors simplify the computationally theoretical solution into a practical algorithm. The final formulation (Eq. 18) is elegant, relying only on the outputs of the two immediately preceding steps (negligible latency addition).
2. The experimental results are compelling.

### Weaknesses
1. I'm concerned about the "Approximation 1" ($J_{x_t} ≈ 0$), which claims that "a well-trained diffusion model is insensitive to local changes in the input", this is a strong and potentially flawed assumption. The Jacobian `J_xt`captures the model's sensitivity to input perturbations. Ignoring it entirely likely oversimplifies the true error propagation dynamics, especially around detailed or high-frequency features where the model is highly sensitive. This approximation might explain why the method, while effective, does not achieve perfect correction. 

2.  The results on SDXL-Turbo (4 steps) show smaller gains than on SDXL (50 steps). What is the minimum number of steps required for TCEC to provide a meaningful benefit?

### Questions
1. How does the performance of TCEC change if Approximation 1 is relaxed, for instance, by using a low-rank approximation of the Jacobian? 
2. The calibration process requires a dataset (1K images/128 videos). How sensitive is TCEC to the size and domain of the calibration data? What are the specific conditions under which TCEC provides diminishing returns?

### Soundness
2

### Presentation
2

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
This paper introduces TCEC, a framework that addresses performance degradation in quantized diffusion models by targeting cumulative error. It pioneers a theoretical model for error propagation and an efficient compensation mechanism, significantly boosting the fidelity of low-bit models with negligible computational overhead.

### Strengths
- The paper's primary strength is its pioneering theoretical framework that mathematically models the previously overlooked problem of error accumulation.
- The proposed method is highly practical, delivering significant performance improvements on low-bit models with a negligible (<0.5%) increase in inference latency.
- Its high degree of generality allows it to be orthogonally combined with various model architectures, samplers, and existing PTQ algorithms as a plug-and-play module.

### Weaknesses
- The method's performance relies on a calibration dataset, making it potentially sensitive to distribution shifts between the calibration and inference data.
- The selection of a key hyperparameter (λ₁) depends on an empirical rule, which lacks the robustness of a more adaptive or principled selection strategy.
- The paper's scope is focused exclusively on Post-Training Quantization (PTQ), without exploring the solution's applicability in Quantization-Aware Training (QAT).

### Questions
- An ablation study on the compensation steps (m) is recommended to empirically validate the theoretical finding from Appendix B, confirming that m=1 provides the optimal trade-off between performance and latency.
- The paper should analyze the method's sensitivity to the calibration dataset—for example, by conducting cross-domain evaluations (e.g., calibrating on A and evaluating on B)—to better quantify its robustness and generalization.
- The evaluation should be extended to more aggressive, few-step sampling settings (e.g., 10 or 20 steps) to verify TCEC's robustness and practical value in high-speed generation scenarios.

### Soundness
3

### Presentation
3

### Contribution
2
