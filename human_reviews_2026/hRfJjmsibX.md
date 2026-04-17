# Pyramid Patchification Flow for Visual Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Diffusion Transformers (DiTs) typically use the same patch size for $\operatorname{Patchify}$ across timesteps, enforcing a constant token budget across timesteps. In this paper, we introduce Pyramidal Patchification Flow (PPFlow), which reduces the number of tokens for high-noise timesteps to improve the sampling efficiency. The idea is simple: use larger patches at higher-noise timesteps and smaller patches at lower-noise timesteps. The implementation is easy: share the DiT's transformer blocks across timesteps, and learn separate linear projections for different patch sizes in $\operatorname{Patchify}$ and $\operatorname{Unpatchify}$. Unlike Pyramidal Flow that operates on pyramid representations,, our approach operates over full latent representations, eliminating trajectory ``jump points'', and thus avoiding re-noising tricks for sampling. Training from pretrained SiT-XL/2 requires only $+8.9\%$ additional training FLOPs and delivers $2.02\times$ denoising speedups with image generation quality kept; training from scratch achieves comparable sampling speedup, e.g., $2.04\times$ speedup in SiT-B. Training from text-to-image model FLUX.1, PPFlow can achieve $1.61 - 1.86 \times$ speedup from 512 to 2048 resolution with comparable quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces "Pyramidal Patchification Flow" (PPFlow), a method designed to improve the inference efficiency of Diffusion Transformers (DiT) / Flow Transformers (e.g., SiT, FLUX.1). Standard DiTs use a fixed patch size (e.g., 2x2) across all denoising timesteps, resulting in a constant number of tokens. PPFlow's core idea is simple: use larger patch sizes (e.g., 4x4) for earlier, higher-noise timesteps and smaller patch sizes (e.g., 2x2) for later, lower-noise timesteps, thereby reducing the token count in early stages and accelerating inference.

The implementation is straightforward:

1. Shared DiT Body: The main Transformer blocks (DiT Blocks) are shared across timesteps (and thus, different patch sizes).

2. Independent Patchify/Unpatchify Layers: Separate linear projection layers ($W_s$ for Patchify, $W_s^u$ for Unpatchify) are learned for each distinct patch size (e.g., one set for 4x4, another for 2x2).

A key distinction from methods like Pyramidal Flow (Jin et al., 2024) is that PPFlow operates consistently on full-resolution latent representations, merely changing how they are patchified, whereas Pyramidal Flow operates on pyramid representations of varying resolutions. This allows PPFlow to avoid "jump points" caused by resolution changes and eliminates the need for complex re-noising tricks during sampling.

Experiments show that fine-tuning PPFlow (three-stage) from a pretrained SiT-XL/2 model requires minimal additional training cost (+7.1% FLOPs) while achieving ~2.02x inference speedup and maintaining comparable image generation quality (FID). Similar speedups are observed when training from scratch. When applied to the text-to-image model FLUX.1, PPFlow achieves 1.61-1.86x speedup across various resolutions with comparable quality.

### Strengths
1. **Simple and Effective**: The core idea—dynamically adjusting patch size based on noise level to reduce token count—is intuitive, simple, and easy to implement (only modifying Patchify/Unpatchify layers).

2. **Inference Acceleration**: Experimental results (Tables 1, 2, 3) clearly demonstrate that PPFlow achieves substantial inference speedups (approx. 1.6x to 2x) across multiple models (SiT-B, SiT-XL, DiT-XL, FLUX.1) and resolutions (256 to 2048), while maintaining comparable image generation quality (FID). This is a highly practical contribution.

3. **Avoids "Jump Point" Issues**: Compared to methods like Pyramidal Flow that switch between different representation resolutions, PPFlow's consistent operation on full-resolution latents avoids the complexities associated with "jump points" and re-noising sampling strategies, keeping the inference process simple.

4. **Easy Adaptation from Pretrained Models**: The paper shows that PPFlow can be efficiently adapted from pretrained DiT models via fine-tuning (Table 2), requiring only about 7-9% additional training FLOPs to gain significant inference speedups. This greatly increases the practical appeal of the method.

5. **Clear Concept**: The paper is well-written with a clear concept. The experimental design is sound, covering both training-from-scratch and fine-tuning scenarios, validated on class-conditional and text-to-image generation. Ablation studies (Tables 6-8) are reasonably thorough, supporting the design choices.

### Weaknesses
1. **Limited Novelty**: While effective, the core idea (adjusting computation/token count based on noise level) is not entirely new. Related works like multi-scale/cascaded Diffusion, Pyramidal Flow (Jin et al., 2024), and especially concurrent works FlexiDiT (Anagnostidis et al., 2025) and Lumina-Video (Liu et al., 2025) explore the exact same core mechanism (time-varying patch sizes). PPFlow's primary novelty lies in its specific implementation (avoiding jump points via full-resolution operation) and its staged training strategy.

2. **Insufficient Comparison with Concurrent Work**: The paper mentions concurrent works FlexiDiT and Lumina-Video in Sec 2 and 4.4. Although it claims PPFlow's training strategy (staged training) is superior to their approach (full-timeline training) and experiments (Tables 4, 5) show better results for PPFlow, the argumentation for why PPFlow's strategy is better, and the trade-offs against the claimed "flexibility" of FlexiDiT, could be discussed more deeply.

3. **Arbitrary Stage Divisions and Patch Sizes**: The timestep points for stage transitions (e.g., 0.5, 0.75) and the patch sizes used (e.g., 4x4, 4x2, 2x2) are manually chosen. The paper lacks sensitivity analysis on these hyperparameters or exploration of more automated selection methods. Increasing the number of stages (Table 7) seems to have diminishing returns and requires more training.

4. **Missing Key Visualizations**: While Table 3 presents metric results for applying PPFlow to FLUX.1 (text-to-image model), the paper lacks corresponding visual samples. Given the subjective nature of text-to-image tasks, adding visual comparisons between PPF-FLUX.1 and the baseline FLUX.1-ft would significantly strengthen the conclusions.

### Questions
**Question 1: What are the fundamental trade-offs between PPFlow's staged training and FlexiDiT/Lumina-Video's full-timeline training?**

The paper suggests PPFlow's staged training allows models to better specialize for specific noise levels, and experiments show better performance than a re-implementation of the Lumina-Video approach. However, FlexiDiT claims their full-timeline training offers more "flexibility," potentially allowing the use of any patch size at any timestep during inference. Could the authors elaborate on the trade-offs? Is PPFlow's specialized approach inherently better under fixed schedules, while FlexiDiT's generalist approach offers more adaptability if the inference schedule (patch sizes vs. timesteps) needs modification? A direct experimental comparison of these training strategies (e.g., using PPFlow's architecture but FlexiDiT's training method) would be valuable.

**Question 2: How sensitive are the results to the choice of stage divisions (timesteps) and patch size combinations?**

The current stage divisions and patch sizes are set manually. How much do the final speedup and image quality depend on these specific choices? Are there better combinations? Could these parameters potentially be learned?

**Question 3: How does PPFlow perform on tasks requiring fine spatial structure or long-range dependencies?**

Does using larger patches (which discard early high-frequency information) in high-noise stages negatively impact tasks that require precise spatial layout or fine-grained textures (e.g., complex scenes, generating images with text)? While ImageNet and T2I benchmark results are strong, how does performance hold up in more challenging scenarios? Suggest adding relevant experiments or discussion.

**Question 4: Could 1x1 patch sizes be considered for the final, very low-noise stages?**

To potentially capture the finest details in the last stages (close to t=1), would incorporating a 1x1 patch size (equivalent to operating directly on pixels/tokens) be beneficial? This might further improve image quality but would require careful consideration of the trade-off, given the drastic increase in token count (e.g., from $(I/2)^2$ to $I^2$) and the associated computational cost surge. What are the authors' thoughts on this possibility?

**Question 5: Clarification on Patchify Implementation and Weight Initialization?**

Original DiT often implements Patchify using a Conv2D layer with kernel_size=stride=patch_size. However, Fig 3 and Sec 3.1 describe it as "extract patches + linear projection ($W_s$)". Are these formulations equivalent? If not, why choose the latter? Furthermore, could the authors clarify precisely how the weight initialization described in Sec 3.3 (averaging/duplicating) is applied to these independent linear layers $W_s$ and $W_s^u$ to avoid confusion?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Pyramidal Patchification Flow (PPFlow), an efficient diffusion model that reduces the number of tokens at high-noise timesteps to accelerate generation and potentially also training. Instead of using a fixed patch size, PPFlow uses a multi-stage patch size schedule, which uses larger patches for earlier timesteps and smaller patches for later timesteps. All denoising transformer blocks are shared across stages. Thanks to this design, PPFlow speeds up inference by at least $1.6\times$ without compromising image generation quality.

### Strengths
- Speedup without compromising quality: PPFlow achieves $1.6\times - 2.0\times$ faster inference on image generation tasks while maintaining comparable image generation quality.
- Thanks to its design, all denoising steps happen in the same resolution, and it naturally resolves the "jump points" issues introduced in prior works. This leads to a simpler sampling scheme, which does not require tricks like renoising but also leads to a better image generation quality.
- Extensive experiments demonstrate the effectiveness of the method.

### Weaknesses
- Varying CFG scales: My primary concern lies in attributing the reported performance improvements to architectural innovations rather than to the varying classifier-free guidance (CFG) scales applied across different intervals. As stated in lines 357–359, the model adopts distinct CFG levels at different denoising stages. While this is a reasonable design choice for PPFlow, it closely parallels the Guidance Interval technique [1], which argues that guidance should not be applied uniformly throughout the denoising process since guidance at earlier timesteps can negatively impact sample diversity and quality. It would strengthen the paper if the authors could provide a direct comparison with Guidance Interval applied to SiTs to confirm that the observed quality gains are not merely due to this scheduling strategy.

- Diminishing returns with more stages: While adding more pyramid stages can further cut computation, it also compromises image generation quality. This seems like a trade-off to me, where aggressive token reduction demands more training but also negatively impacts the generation quality.

---
[1] Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., & Lehtinen, J. (2024). Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37, 122458-122483.

### Questions
- Regarding lines 280–283, are the time intervals empirically determined? It would be interesting to explore whether adaptive interval selection, based on certain metrics such as the signal-to-noise ratio (SNR), could further improve performance or stability.
- The choice of a $4\times 2$ patch configuration seems interesting, as most patch-based designs adopt square-shaped patches. Have the authors experimented with larger square configurations, such as $8\times8$ patches, especially for higher-resolution synthesis? Such an ablation could clarify whether the asymmetric patching contributes meaningfully to the model’s efficiency or generative quality.

### Soundness
3

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
4

### Summary
The paper explores adaptive patch sizes for DiT models trained for image generation. In contrast to prior works, it argues for separating the patch sizes for different time segments to better align with the optimal test-time inference strategy. It tests on top of SiT for class-conditional image generation on ImageNet and Flux-dev for text-to-image generation. It reports better qualitative results for both setups.

### Strengths
- The paper presents several useful ablations for adaptive patch size strategies: patch-level embedding, stage-wise CFG, 
- The paper is written well
- Overall, this paper looks like a useful reference for adaptive patch size exploration

### Weaknesses
- FLOPs are not a good metric to measure training speed. Clock wall time should be reported as well.
- Figure 3 is not too informative of the patchification process since it presents the input and output the same. Also, the input to patchification is a 2D image patch, not flattenning is a part of patchification.
- Typo on L248: [WWWW] reads like a matrix multiplication of 4 matrices W. Same for L252
- The paper claims to train Flux-dev, but does not show a single generated image from it. GenEval is not a reliable benchmark and cannot be the only source of assessment.
- The authors claim that Lumina-Video's strategy to use all the patch sizes for all the timesteps is inferior. But it's unclear if its cfg is just as tuned and if the proposed strategy supports autoguidance like Lumina-Video does.

### Questions
- Is it possible to support autoguidance (using a large-patch model as the weak one)?

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
4

### Summary
This paper proposes Pyramidal Patchification Flow (PPFlow), a method that dynamically varies the patch size of DiT across timesteps—using larger patches in higher-noise timesteps to reduce token count and smaller patches in lower-noise timesteps to preserve visual fidelity. PPFlow operates on the full-resolution representation throughout the generation process and trains the model for each patch size only over the corresponding timestep range. As a result, PPFlow achieves faster inference without generation quality degradation.

### Strengths
- The paper is well-structured, providing clear explanations of design choices and experimental results. It includes comprehensive evaluations and ablation studies that effectively validate the proposed approach.
- The proposed PPFlow patchification strategy can be applied to various transformer-based diffusion models due to its simplicity and comparable generation quality to conventional fixed patch-size models.

### Weaknesses
- There are several typographical errors in the paper. For example, there are double commas in the abstract (line 36, p1); in line 222 (p.5), the word “depent” should be “dependent”; and in Table 5 (p.8), “pyrimid rep” appears to be a typo.

### Questions
- In Table 7, the results show that increasing the number of training steps helps recover generation quality (line 445 vs 446 & line 447 vs 448). In Table 2, instead of matching the number of training steps, what happens if we match the training FLOPs between the two-level and three-level methods? How does the generation quality compare under this training-cost-matched condition?

### Soundness
3

### Presentation
3

### Contribution
3
