# VLSA: Enhancing Vision-Language Understanding via Perception and Cognition Alignment

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Prevalent Vision-Language (VL) alignment techniques within Multi-modal Large Language Models (MLLMs) struggle to adequately align the language model with visual inputs, resulting in hallucinations and undermining reliability.  We rethink the modality alignment in MLLMs from the perspective of reducing information loss and present an efficient plug-in, VL Superior Alignment (VLSA), which decouples the alignment into two stages. The first stage, referred to as Perception Alignment, minimizes information loss in visual encoding through compressive encoding for high-resolution images and innovative reconstructive training leveraging latent diffusion models. The second stage, termed Cognition Alignment, reduces information loss in response generation by enhancing the language model's ability to grasp both high-level visual semantics and low-level image appearances, achieved by novel auxiliary self-supervised fine-tuning (SSFT) objectives. Extensive experiments across over 25 MLLM benchmarks and 7 MLLM architectures, thorough ablations, and analyses of computational overhead underscore the improvement of both performance and efficiency brought by VLSA. In service to the MLLM research community, our code and model checkpoints will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents VLSA, which reframes multimodal alignment as reducing information loss along the vision→language pipeline. The method has two stages: Perception Alignment that injects high-res patch details into a low-res structural stream, paired with reconstruction training (LDM-based) to mitigate encoding loss; and Cognition Alignment, where the LLM predicts VQ-VAE code indices and sampled pixels to lessen decoding loss. In inference, auxiliary modules are disabled; the visual token length is fixed, yielding both accuracy and efficiency gains across many benchmarks and several MLLM backbones.

### Strengths
1） Clear two-stage framework with practical appeal. The design explicitly targets perception-side and cognition-side loss, is plug-and-play, and keeps inference lightweight.

2）Good empirical coverage and portability. The approach delivers consistent gains across diverse tasks and multiple MLLMs, suggesting robustness of the alignment recipe.

### Weaknesses
1）Motivation–method disconnect. Although framed under “information-loss reduction,” the three components, compressed high-res encoding, LDM-based reconstruction, and SSFT, are presented more as parallel tricks than a tightly coupled mechanism. The paper does not convincingly argue why all three are necessary together, how their gradients/targets complement rather than compete, or how each specifically addresses the stated losses beyond high-level intuition. This reads as a stitched trio rather than a single, strongly motivated design.

2）Compressed high-res input is not novel; conflict-avoidance remains unclear. Multi-resolution schemes that mix high-res patches with a low-res global stream have prior art. The paper should clearly articulate how SA-Perceiver prevents conflicts between high-res local details and low-res global semantics during injection. What controls (gating, residual scaling, normalization, attention head specialization) ensure injected details don’t overwhelm global semantics? Any controlled comparisons that show an optimal regime and failure modes?

3）Key hyperparameters lack guidance. The paper sets defaults but does not provide sensitivity/selection rationale for crucial knobs (k, τ, λ₁, λ₂, δ). Even a brief stability range or small sensitivity curves/tables would improve clarity for practitioners.

4）Ablation granularity on cross-component synergy is limited. Existing ablations mostly compare within components. To support the “triad” motivation, the paper should include cross-component causal ablations under a fixed budget: only compressed encoder, +reconstruction, +SSFT, full, with convergence curves and analysis of additive vs. synergistic gains.

### Questions
Please refer to Weakness

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
The paper reframes VL alignment as reducing information loss along the vision→language path and proposes a plug-in, two-stage method: Perception Alignment (compressive SA-Perceiver plus reconstructive training with an LDM so that embeddings (V,P) can recover the input) and Cognition Alignment (SSFT that has the LLM predict VQ-VAE codebook indices and pixels as auxiliary targets). The approach is trained in two stages and evaluated across 7 architectures and 25+ benchmarks; at inference the visual sequence is fixed at 576 tokens and all auxiliary models are disabled.

### Strengths
1. Clear, principled decomposition. Casting hallucination as information-loss (encoding/decoding) motivates the dual alignment design with concrete mechanisms and objectives. 

2. Reconstruction-driven alignment. Using an LDM to enforce recoverability from (V,P) is well justified and supported by ablations on alternative decoders.  

3. Practical efficiency. A constant 576-token visual sequence and deactivated auxiliaries at test time.

4. Broad applicability with measurable gains. Improvements shown on LLaVA variants and other MLLMs.

### Weaknesses
Limited diagnosis of why and when VLSA helps. Although ablations are included (SA-Perceiver design, LDM variant, encoders/backbones, loss ratio), the paper lacks error taxonomies explaining failure modes.

### Questions
Please refer to weaknesses.

### Soundness
3

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
The VLSA (Vision-Language Semantic Alignment) framework redefines multimodal alignment by explicitly minimizing information loss throughout the visual inference process in large multimodal language models. It introduces two complementary stages: Perception Alignment, which employs a Self-Attention Perceiver (SA-Perceiver) for compressive high-resolution encoding and an LDM-based reconstructive training mechanism to preserve detailed visual semantics while maintaining efficiency; and Cognition Alignment, which enhances the LLM’s understanding of visual inputs through self-supervised fine-tuning that predicts VQ-VAE codebook indices (for high-level semantics) and pixel values (for low-level details). Integrated into models such as LLaVA, Qwen-VL, and MiniGemini, VLSA demonstrates consistent improvements across over 25 benchmarks, effectively mitigating original, encoding, and decoding information losses and improving both perception fidelity and reasoning accuracy.

### Strengths
- The paper clearly separates different sources of information loss across stages, making the objective of reducing each component straightforward and intuitive.
- The results show steady improvements across multiple benchmarks and model variants, indicating generally positive performance.
- The experiments demonstrate faster inference and lower computational cost, particularly on LLaVA-Next, highlighting practical efficiency gains.

### Weaknesses
- Although performance improvements are consistent across benchmarks, they appear incremental; gains on alternative architectures remain positive but relatively small, making it difficult to justify the added model complexity.
- The motivations behind key design choices are insufficiently explained. Despite some ablations and reasoning in the main text and appendix, the use of an LDM for reconstruction, the adoption of a VQ-VAE codebook in the alignment stage, and the single-layer MLP design for the Epigone lack clear theoretical grounding or conceptual justification. See questions for details.
- Several formulas and technical descriptions are ambiguous, and some claims—particularly those related to efficiency and attention mechanisms—are not well supported by rigorous analysis or empirical evidence. Also see the questions for details.
- The comparisons mainly focus on architectural baselines; additional baselines addressing visual–language alignment (e.g., LaViT, Ovis) should be included for a fairer evaluation.

### Questions
## Questions

1. Formulas
- Formula at 207 could be misleading. why use w_kv for both K and V? does that mean the weight matrices are shared, and K and V are the same?
- Formulas at 208 and 212 look non-standard. especially, what exactly is happening in 212? why is self-attention defined this way?
- For 214, how is the output split into the visual embedding and the global embedding? what determines this separation?

2. Design decisions
- I am not fully clear about the intuition behind using diffusion-based approaches for *understanding-oriented* alignment. While unified models often integrate understanding and generation jointly, most existing works rely on vision encoders (e.g., CLIP, SigLIP) for understanding and VAE–diffusion pipelines for generation. Given that MLLMs primarily focus on visual understanding rather than generation, what is the underlying motivation for introducing VAE and diffusion mechanisms in this context? What intuition supports their role in improving semantic alignment?
- For the reconstructive training, why does it need to be based on an LDM? The inclusion of a diffusion model seems to complicate both the training and generation processes. Although this is discussed in lines 250–255 and the appendix, the motivation is still unclear. Given the known training instability of diffusion models and the fact that most VLLMs focus on understanding rather than generation, this design choice feels excessive. Is there any mathematical or conceptual intuition supporting why diffusion, specifically, is suitable for this alignment objective?
- Regarding the Epigone structure, is a single-layer MLP sufficient to translate visual embeddings into prompt embeddings? Given that this mapping is highly non-trivial and involves bridging distinct semantic spaces, it is unclear whether such a simple structure can effectively perform this translation. Could the authors elaborate on this?

3. Clarifications
- The claim that “Monkey and LLaVA-UHD employ Q-former-like resamplers to reduce visual tokens, which may exacerbate E-IL” (line 133~134) needs clarification — why would reducing visual tokens through resampling necessarily increase encoding-level information loss? An explanation or supporting evidence would make this point clearer.
- Could the authors clarify the claim that “the SA-Perceiver is also responsible for achieving VL alignment” (line 215) ? Beyond being jointly trained with language-based losses, what mechanism within the SA-Perceiver itself enforces or contributes to vision–language alignment? It would be helpful to understand whether this alignment arises from the architecture’s design or solely from downstream training objectives.
- The claim that “the VQ-VAE is capable of encapsulating a broad spectrum of semantics that transcend specific, human-defined categories” (line 287-290) could be further clarified. Since VQ-VAE is a relatively early model, it is uncertain whether it can effectively capture such broad semantics compared to more recent discretization approaches. Alternative methods, such as residual quantization [1], might offer stronger representational capacity or finer semantic granularity. Also, what is the codebook size used in the experiment?
    
    
    [1] Lee et al., *Autoregressive Image Generation using Residual Quantization*, CVPR 2022.
    

4. Experiments
- Considering that the LLaVA family already employs multi-resolution visual processing and explicitly handles such cases, could this be the reason why the proposed method shows greater improvement on LLaVA-based models compared to other MLLMs? In other words, is the model inherently more compatible or better aligned with this architecture, leading to stronger relative gains?

5. Minor
- The citation style such as *LLaVA Liu et al. (2023a)* or *PaLI Chen et al. (2022)* feels a bit informal. Using a cleaner model-based reference format like *LLaVA (Liu et al., 2023a)* or *PaLI (Chen et al., 2022)* would look more consistent.
- Just out of curiosity, is the bottom part of Figure 1 based on an actual example? The transition between the real image and the perceived visual representation seems quite drastic.

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
This work proposes two additional alignment stage to improve the vision-language alignment between the visual encoder and language decoder in common LVLMs. First of these stages, "Perception Alignment", aims to reduce the potential losses occurring during the visual encoding process through aligning the initial visual features with encoder output features whereas the second of these stages, "Cognition Alignment", aims to mitigate any such losses occurring during the decoding phase. The work provides several quantitative and qualitative results to support its claims.

### Strengths
Here are the main strengths of this work:

- The central methodology of the paper, combining the three existing ideas of low-res/hig-res alignment, reconstructive training and using auxiliary self-supervised objectives for the decoder is novel and interesting.
- The formulation of the modality alignment problem in three stages in this manner, from the initial lossy compression errors to decoding phase errors is clear and technically sound.
- The narrative of the work is easy to follow, notations are clear and technical details are discussed extensively.
- Several results demonstrate the effectiveness of the proposed method, e.g. some of the results on Tables 1 and 2.

### Weaknesses
Here are the main weaknesses of the work:

**W1: Complex Pipeline with Potentially Redundant Aspects:** Although the overall pipeline is novel and interesting, the work introduces several hard-to-engineer phases, with performance gains failing to justify them fully. While some parts of this pipeline is well-known to be working well in the field (e.g. the low-res/high-res setting, also evidenced by the ablation results on Table 4), several parts seem to bring relatively marginal improvements (same ablation table highlights this to an extent). With the additional parameters, hyperparameters and other training settings introduced by the method, this multi-stage pipeline could be undesirable for downstream applications.

**W2: Lack of Empirical Validation:** The work lacks empirical validation in several dimensions:

- The results presented in the work are still text-heavy benchmarks, as discussed by [A] in the literature for example. Given the goal of the work around improving the vision-language alignment, having more visual-heavy benchmarks would have been better, e.g. OCR-VQA or RefCOCO.

- Furthermore, the improvements brought in by this multi-stage pipeline are relatively low, except for a few others, such as VizWizVQA. Having confidence intervals around the evals with marginal improvements (e.g. GQA) would have been great. 

- Evidence for “information-loss reduction” is indirect. Gains on VQA-like tasks don’t isolate O/E/D-IL. There’s no direct measure (e.g., reconstruct-from-embedding PSNR/LPIPS) showing retained information, making the exact contributions of the multiple stages hard to grasp.

- Finally, reporting the vision-only performance (e.g. image recognition of the visual encoder) and the text-only performance of the individual components would also have been great to see the effects of the additional alignment stages introduced on them.


---
[A] Tong, P., Brown, E., Wu, P., Woo, S., IYER, A. J. V., Akula, S. C., ... & Xie, S. (2024). Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37, 87310-87356.

### Questions
- How well do you think VLSA work for OCR-heavy benchmarks, e.g. OCR-VQA, and benchmarks requiring finer-grained visual analysis compared to the ones presented in the work, e.g. RefCOCO?

- How well do you think that the end visual encoder you get is performing on its own, e.g. in its image recognition capabilities? 

- Orthogonally to this question, Cambrian [A] also introduced a very neat methodology for checking, involving adapting the model with a frozen LLM. How well do you think the visual encoder and the projector in this work would perform under Cambrian's settings?

- Does VLSA bring in a regression in text-only benchmarks, e.g. on HellaSwag?

---
[A] Tong, P., Brown, E., Wu, P., Woo, S., IYER, A. J. V., Akula, S. C., ... & Xie, S. (2024). Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37, 87310-87356.

### Soundness
3

### Presentation
4

### Contribution
2
