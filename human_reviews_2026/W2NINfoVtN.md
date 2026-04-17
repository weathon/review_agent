# VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
We introduce Value Sign Flip (VSF), a simple and efficient method for incorporating negative prompt guidance in few-step (1-8 steps) diffusion and flow-matching image and video generation models. Unlike existing approaches such as classifier-free guidance (CFG), NASA, and NAG, VSF dynamically suppresses undesired content by flipping the sign of attention values from negative prompts. Our method requires only a small computational overhead and integrates effectively with MMDiT-style architectures such as Stable Diffusion 3.5 Turbo and Flux Schnell, as well as cross-attention-based models like Wan. We validate VSF on a proposed challenging dataset, NegGenBench, with complex prompt pairs. Experimental results on our proposed dataset show that VSF significantly improves negative prompt adherence (reaching 0.420 negative score for quality settings and 0.545 for strong settings) compared to prior methods in few-step models (scored 0.320-0.380 negative score) and even CFG in non-few-step models (scored 0.300 negative score), while maintaining competitive image quality and positive prompt adherence. Our method also suppressed a generate-then-edit pipeline, while also having a much faster runtime. Code, ComfyUI node, and dataset are available in https://github.com/weathon/VSF/tree/main.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Value Sign Flip (VSF), a lightweight, training-free negative-guidance method for few-step diffusion/flow models that flips the sign of negative-prompt value vectors inside attention. For cross-attention models VSF concatenates positive/negative keys/values and applies a \alpha scaling to the negative values; for MMDiT-style models (e.g., SD 3.5 Turbo, FLUX Schnell) it further duplicates the negative tokens and masks attention paths so only image -> negative interactions are affected, optionally adding a bias to stabilize quality. VSF adapts token-, layer- and step-wise, aiming to avoid oversaturation and “mixing” failures seen when forcing CFG or using fixed-strength attention methods like NASA and NAG. On a new negation benchmark (NegGenBench) built from challenging positive/negative prompt pairs, VSF improves negative adherence while keeping quality and positive adherence competitive, and runs in ~3 s with few steps; human and MLLM-based evaluations broadly agree with these trends.

### Strengths
- The proposed VSF is simple and plug-and-play. And it does not include many runtime overhead.
- It performs better than baseline methods on the collected benchmarks. It achieves better negative-prompt compliance at similar quality vs. NAG/NASA and even CFG (multi-step) in reported settings. The visualization results look visually good.
- Clear ablations/trade-offs (α, β, masking/duplication, Whole-Embedding Flip)

### Weaknesses
- Evaluation relies heavily on MLLM judges (LLaMA/Qwen variants) for pos/neg/quality scoring; such metrics can be biased or insensitive to artifacts, and the dataset/hyperparameters are author-curated. 
- Some quality trade-offs remain at higher negative strength; masking/bias choices introduce extra knobs and implementation complexity in MMDiT stacks. 

Writing:
-  Figure 3, 5, 6 is too small, could be scaled up for better organization.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- Since previous negative prompts approach generally employ score matching, they are not compatible with few-step generation models.
- To address the problem, they simply flip the value in the attention layer, which results in effectively removing the undesired contents in the final images.

### Strengths
- The proposed method is simple but effective
- The proposed method can be applied to a few-step model.

### Weaknesses
- The proposed method is not novel. Manipulating attention has been employed for image editing with diffusion models and the flow-matching model. (e.g., [Attend-and-exit], [self-guidance], [BoxDiff])
- In Figure 5, VFS not only eliminates the undesired contents but also changes the other components. Specifically, in the starry night examples, the city has gone. Also, in Figure 5(right), the car is still reflected in the image with the proposed method.
- Figures should be well illustrated. ( font size of figure 3,5 is too small, and figure 6 is too small)

[Attend-and-exit]: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models
[self-guidance]: Diffusion Self-Guidance for Controllable Image Generation
[BoxDiff]: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion

### Questions
- Compared to the previous works using attention manipulation, does the proposed method contain a specific technique or contribution for being compatible with few-step models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Value Sign Flip (VSF): a negative guidance.

Negative guidance is useful in generative pipelines: progressively adjusting the results by removing something from the image.

Problem statement:
* CFG does not work in few-step configurations.
* Negative Steer Away Attention (NASA) is currently limited to cross-attention models.
* Normalized Attention Guidance (NAG) primarily targets quality control rather than avoiding negative prompts.
* NASA and NAG subtract negative attention.
* They do not generalize to various timesteps, layers, or image regions.

Method (VSF)
* (Cross-attention-based models) Flipping the sign of negative prompt values within the attention calculation.
* (DiT models) Duplicating negative prompts, one remains unflipped, another is flipped.

Advantages
* VSF removes some concepts from generated images: wheels from bicycle, hands from clock, etc.
    * maintaining image quality and adherence to the positive prompts
* VSF works in few-step (1~8 steps) diffusion and flow matching models.
* VSF generalizes to SD 3.5 turbo, FLUX schunell, and Wan.
* VSF is computationally cheap.

New dataset: NegGenBench
* Positive-negative prompt pairs from ChatGPT o3.
* Negative prompts are core components from positive prompt; it is intentionally challenging.

Evaluation: fine-tune a VLM (Qwen) for measuring faithfulness to negative prompts

### Strengths
Originality:
1. The proposed method is new.

Quality:
1. The related work section covers relevant literature: CFG, Negative Guidance, and Few-step generators
2. The competitors are aggressively chosen, even Nano Banana.
3. Discussion is thorough
    1. trade-off between positive and negative prompts
    2. trade-off between quality and negative prompts
    3. attention maps
    4. ablation study

Clarity:
1. The explanations are kind to the readers, step-by-step from NASA to the proposed method.

Significance:
1. The method is simple and effective.
    1. simple: concatenate the values and keys of the positive and negative prompts, then flip the sign of the negative prompt values
    2. effective: Table 2

### Weaknesses
minor
1. Please properly use \citet and \citep
2. Fonts are too small in the figures.
3. Is “unbrulla” a typo? or is there a message?

### Questions
1. Why does NASA have points with negative score below 50 only in Figure 6?

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
4

### Summary
This paper proposes a simple approach for negative guidance of the text-to-image (T2I) models. Current approaches flip the sign of the attention output for this, but this ends up on applying the same scale of the negative guidance across different areas of the image, and all different layers of the model. Instead, they use the attention map to calculate a per-token wieght for this negative guidance. They first develop this for approaches that use cross-attention mechanism (like latent diffuoin architecture), and then propose a new mechanism to adapt this to multimodal diffusion transformer (MMDiT)-based models like SD3, and others.

### Strengths
The idea of this paper is simple, but I like how they have distilled knowledge from the literature, and based on that—as well as their solid understanding of the attention mechanism—they have proposed this simple idea.

### Weaknesses
1) There are some grammatical errors and confusing parts in the paper that need to be addressed:
- Should $x_{t-1}$ be $x_{t+1}$ in Eq. (1)?
- line 166: *"The method NASA applies the guidance in intermediate states instead of the predicted noise or velocity."* — this statement is somewhat ambiguous.
- line 188: "*However, it also limits the model’s ability to follow negative prompt guidance if the constraint is set to be too tight ...*" — this sentence could be improved for clarity and readability.

2) The concepts used to illustrate the issue with generation quality in Figure 2, under the presence of negative and positive prompts, are not optimal. Since winter and snow are strongly related concepts, they may be entangled in the diffusion model’s learned distribution. Using one as a positive and the other as a negative prompt may not clearly demonstrate the intended issue with current models, as the observed effect could stem from this conceptual dependence rather than the model’s capability to interpret negative guidance.

3) This paper mentions that "*rendering prompts containing negations ineffectively or made the negative prompt appears even more (e.g., a prompt like “a scientist who is not wearing glasses” will often generate a scientist with glasses—sometimes even more frequently than a simple prompt like “a scientist”).*".  
There are some fairness-oriented approaches, such as ITI-Gen [1] and FairQueue [2], that discuss related issues. They point out that this cannot be addressed using **hard prompts**, but can be mitigated through prompt learning. While I understand that your setup is different, discussing the similarities and differences between your approach and these works could strengthen the paper.

4. I believe the task of **negative guidance** can be viewed as a special case of **image editing**, where the prompt explicitly describes the removal of a concept while no input image is provided. From this perspective, using Qwen-Image as an external baseline (Table 1) is an interesting choice. Given that it can edit images while keeping other regions intact, it could serve as an informative upper bound for the negative guidance task. The performance gap could also highlight directions for future research. However, the details of how Qwen-Image is used, including the experimental setup, should be included in the main paper (at least at a high level).

$ $

*References:*

[1] ITI-GEN: Inclusive Text-to-Image Generation, ICCV'23

[2] FairQueue: Rethinking Prompt Learning for Fair Text-to-Image Generation, NeurIPS'24

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
