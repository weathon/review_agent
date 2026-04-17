# VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Recent advancements in visual autoregressive models (VAR) have demonstrated their effectiveness in image generation, highlighting their potential for real-world image super-resolution (Real-ISR). However, adapting VAR for ISR presents critical challenges. The next-scale prediction mechanism, constrained by casual attention, fails to fully exploit global low-quality (LQ) context, resulting in blurry and inconsistent high-quality (HQ) outputs. Additionally, error accumulation in the iterative prediction severely degrades coherence in ISR task. To address these issues, we propose VARestorer, a simple yet effective distillation framework that transforms a pre-trained text-to-image VAR model into a one-step ISR model. By leveraging distribution matching, our method eliminates the need for iterative refinement, significantly reducing error propagation and inference time. Furthermore, we introduce pyramid image conditioning with cross-scale attention, which enables bidirectional scale-wise interactions and fully utilizes the input image information while adapting to the autoregressive mechanism. This prevents later LQ tokens from being overlooked in the transformer. By fine-tuning only 1.2\% of the model parameters through parameter-efficient adapters, our method maintains the expressive power of the original VAR model while significantly enhancing efficiency. Extensive experiments show that VARestorer achieves state-of-the-art performance with 72.32 MUSIQ and 0.7669 CLIPIQA on DIV2K dataset, while accelerating inference by 10 times compared to conventional VAR inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes VARestorer, a one-step super-resolution framework that addresses error accumulation in visual autoregressive models through distribution matching distillation. The method incorporates cross-scale pyramid conditioning to enable multi-scale information interaction while maintaining the original model's generative capabilities. Experiments demonstrate superior performance, achieving an optimal balance between efficiency and restoration quality.

### Strengths
1. The paper adapts VAR models to the real-world image super-resolution (Real-ISR) task. The proposed distillation approach directly tackles the critical issue of error accumulation inherent in autoregressive models for restoration.
2. The pyramid image conditioning with cross-scale attention enables bidirectional scale-wise interactions and fully utilizes the information of the input image while adapting to the autoregressive mechanism.

### Weaknesses
1. The computational cost of the cross-scale pyramid conditioning should be discussed, with quantitative analysis to justify its overhead in relation to performance gains.
2. The labels and arrows in Figure 3 are unclear. Please define $\mathcal{D}$ and $\mathcal{E}$ (possibly Encoder and Decoder) in the caption or figure, and redraw the arrows to unambiguously show the process flow and component interactions.
3. The paper employs BLIP to generate textual prompts, yet its necessity remains unverified. The claim that text guidance improves restoration quality lacks support from ablation studies.
4. Although the high efficiency of training only 1.2% of the parameters was emphasized, no performance comparison was made with full parameter fine-tuning. It is suggested to add the baseline experimental results of full parameter fine-tuning to conclusively prove that LoRA can balance efficiency while maintaining performance.
5. The computational efficiency claims lack comprehensive quantification. While inference time is reported, metrics like FLOPs or MACs are missing, making cross-architecture comparison difficult.

### Questions
1. How does the distillation process prevent the student model from learning and reproducing any inherent errors or suboptimal predictions made by the teacher model during distribution matching?
2. By transforming a sequential decision process into a one-shot mapping, the model might sacrifice some expressive power for computational efficiency. Could you discuss the theoretical limitations this imposes, especially when addressing highly complex or unseen degradations?

### Soundness
3

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
4

### Summary
This paper proposes a VAR-based distillation framework that distills the generative prior of a pretrained VAR model into a one-step model to achieve more efficient real-world image super-resolution. In addition, the authors propose a full-attention-based cross-scale conditioning mechanism to further improve generation quality. Experiments on multiple datasets demonstrate significant improvements on no-reference metrics.

### Strengths
1. The paper’s narrative is coherent, the language is clear and fluent, and it is easy to follow.

2. Beyond super-resolution, the authors also conduct experiments on de-raining and low-light enhancement, demonstrating the effectiveness of the method.

3. Through ablation studies, the paper validates the effectiveness of each component.

### Weaknesses
From Table 1, although the method shows clear gains on no-reference metrics, its performance on reference-based metrics is poor. This result may suggest a fidelity–perception trade-off; the authors should explain possible reasons for the weak reference-based performance.

### Questions
Why is a restorer used as a preprocessing step? How does the VAR model perform without it?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes VARestorer, a one-step visual autoregressive (VAR) distillation framework for real-world image super-resolution (Real-ISR). The method addresses two key challenges in applying VAR to ISR: (1) the next-scale prediction mechanism's limited ability to exploit global low-quality context, and (2) error accumulation during iterative prediction. VARestorer distills a pre-trained text-to-image VAR model into a one-step model using distribution matching, and introduces cross-scale pyramid conditioning with full attention to better leverage input information. Experiments on DIV2K-Val, DrealSR, and RealSR show improvements on perceptual metrics (72.32 MUSIQ, 0.7669 CLIPIQA) with 10× faster inference than vanilla VAR.

### Strengths
1. This paper has clear problem motivation, Identifies specific limitations of VAR for ISR (error accumulation, limited context)
2. Solid experimental setup: LODO protocol, multiple datasets, standard metrics
3. Good visual quality: Figure 1, 4 show appealing results
4. Ablations validate components: Table 3 shows each component contributes
5. Practical efficiency: One-step inference is desirable for deployment

### Weaknesses
1. This paper has weak theoretical foundation, including the following points:
- KL divergence formulation (Eq. 3) with mismatched conditioning lacks justification
- No analysis of distillation convergence or optimality
- Cross-scale full attention's compatibility with autoregressive generation unclear
- Missing theoretical characterization of when method should succeed/fail

2. Insufficient experimental rigor:
- Training 1 epoch (10K steps) without convergence analysis is concerning
- Teacher fine-tuning on 512×512 (vs. original 1024×1024) impact not studied
- Baseline comparison potentially unfair (VARSR-10 vs VARestorer-1)
- No sensitivity analysis of loss weights (λKL, λperc, λMSE)
- "w/o distill" baseline terrible but not investigated

3. Method limitations underexplored:
- Caption dependency critical but only qualitatively discussed (Figure G), also failure on severe degradation acknowledged but not analyzed.
- Fixed resolution (512×512) limits practical utility
- PSNR/SSIM degradation vs. baselines suggests fidelity issues


4. For presentation:
- Parameter count inconsistencies (1.2% vs 0.09% vs 27.3M)
- Cross-scale conditioning implementation vague
- Missing algorithmic details for key components
- No discussion of why VAR is preferable to diffusion

### Questions
1. In Equation (3), pT and pS have different conditioning. How do you compute DKL(pT(rk | rHQ,<k) || pS(rˆk | rLQ))? Is this importance sampling, or are you assuming some form of equivalence?
2. How does "cross-scale full attention" preserve the autoregressive property of VAR? Don't future scales condition on past scales in VAR? Providing the attention mask explicitly would help.
3. Teacher fine-tuning: Why fine-tune the 1024×1024 VAR on 512×512? Doesn't this lose the high-resolution knowledge? Have you tried using the original teacher directly?
4. Training sufficiency: Why only 10K steps (1 epoch)? Have you verified convergence? What happens with more training?
5. Which is correct: 1.2% (abstract), 0.09% or 1.78M (Appendix A), or 27.3M (Table 2)?
6. Table 3 shows "w/o distill" performs very poorly. Does this mean VAR is inherently unsuitable for ISR without distillation?
7. Why does VARSR need 10 steps? Have you tried VARSR with 1 step for fair comparison?

### Soundness
2

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
4

### Summary
This paper proposes a distilled VAR framework for real-world image super-resolution. The method distills a single-step student model from a multi-step VAR teacher to achieve faster inference while maintaining visual quality. A multi-step guidance strategy is introduced to reduce accumulated errors during training. Experiments on several benchmarks demonstrate competitive no-reference metrics and visually sharp results compared with existing SR methods.

### Strengths
The paper is clearly written and tackles an important problem in real-world image super-resolution. The idea of distilling a multi-step VAR model into a single-step version is practical and well-motivated for efficiency. Experiments are thorough, covering multiple benchmarks, and the proposed method achieves strong results on several perceptual quality metrics.

### Weaknesses
- **Inconsistency in the core motivation and training design.**

The paper claims that multi-step VAR models suffer from cumulative errors, yet the proposed approach distills knowledge from a multi-step VAR teacher. Since the teacher’s outputs are already affected by the same multi-step bias, the student may inherit those errors rather than eliminating them. This raises a conceptual inconsistency between the stated problem and the actual training setup, making it unclear whether the proposed method truly mitigates error accumulation or simply compresses it into a single step.

- **Mismatch between metrics and visual realism.**
Although the method achieves significantly higher no-reference quality scores on RealSR, the visual results (e.g., the flower example in Fig. 4) exhibit unrealistic, hallucinated textures. The generated details appear sharp but not natural, suggesting that the improvement in perceptual metrics may come from artificial high-frequency patterns rather than faithful detail recovery. This discrepancy questions whether the claimed perceptual gains reflect genuine visual quality improvements.

If these concerns are properly addressed and clarified, I would be willing to consider raising my score.

### Questions
please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
