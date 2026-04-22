# Massive Activations are the Key to Local Detail Synthesis in Diffusion Transformers

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Massive Activations (MAs) are a well-documented phenomenon across Transformer architectures, and prior studies in both LLMs and ViTs have shown that they play a substantial role in shaping model behavior. However, the nature and function of MAs within Diffusion Transformers (DiTs) remain largely unexplored. In this work, we systematically investigate these activations to elucidate their role in visual generation. We found that these massive activations occur across all spatial tokens, and their distribution is modulated by the input timestep embeddings. Importantly, our investigations further demonstrate that these massive activations play a key role in local detail synthesis, while having minimal impact on the overall semantic content of output.
Building on these insights, we propose Detail Guidance (DG), a MAs-driven, training-free self-guidance strategy to explicitly enhance local detail fidelity for DiTs. Specifically, DG constructs a degraded ``detail-deficient'' model by disrupting MAs and leverages it to guide the original network toward higher-quality detail synthesis. Our DG can seamlessly integrate with Classifier-Free Guidance (CFG), enabling joint enhancement of detail fidelity and prompt alignment. Extensive experiments demonstrate that our DG consistently improves local detail quality across various pre-trained DiTs (\eg, SD3, SD3.5, and Flux).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study the role of massive activation (MA) in DiT models. Massive activation is a common observation in transformer models. The authors attribute massive activations in DiT to the generation of local detail. Experiments show that disruption of MA degrades local information. The authors propose Detail Guidance (DG) to enhance CFG on local details. The manuscript is clearly written. The analysis of MA is comprehension. The experimental design is legit. Overall, it is a high-quality study. Though, I note several unclear parts below and suggest several controls to support the effectiveness of DG.

### Strengths
Massive activation is a common observation in transformer models across modalities. Its functional role is less known in DiT models. The analysis provided by the study is comprehensive (Figure 3-5). The finding of MA’s role of synthesizing is novel and supported. Leveraging this finding, Detail Guidance seems to be an effective, though simple, way to enhance local details.

### Weaknesses
**DG control**

As the authors mentioned, Karras et al 2024 proposed using an under-trained version to contrast the condition path. DG works in a similar way. Method in Karras et al 2024 should be used as a control. Or, if MA-disruption is really only about disrupting local detail features, there should be another control that uses blurred conditional path as the unconditional signal.

**MA-disruption control**

In “Non-MA disruption”, it is hard to say that zeroing non-MA dims is a fair control. The amount of perturbation in “Non-MA disruption” is smaller than in “MA disruption”. To be a fair control, the same total magnitude of disruption should be the same.

**Missing information, or not emphasized in the main text**

Are MA studied in the conditional path (D_/theta(z,t,c) in eq 1) or unconditional path (D_/theta(z,t) in eq 1)? From eq. 2, it seems the state is taken from the unconditional path, but in eq 4, the MA is studied/manipulated in the conditional path. The interpretation of the results heavily depends on where MA are found and manipulated. Authors should clarify where the hidden states are taken from in Figure 2-5.

Figure 2 misses information on which layer and time step the activations are taken from.

### Questions
**Spatial map of MA**

In the ViT studies, the MA often appears in only several background tokens in the image. If the role of MA in DiT is to synthesize details, then does it mean the MA is supposed to be in all the tokens, or only in tokens with plenty of details? If only disrupting several tokens’ MA, is the details only missing locally? The special activation map of MA dim is not shown. Showing an activation map of MA dim would be helpful to understand its actual role.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper systematically studies the massive activations in the DiT (Diffusion Transformer) image generation framework. The authors find that these activations are highly correlated with detail synthesis. Based on this observation, they propose a guidance (DG) strategy to steer the generation process toward better texture synthesis. Experiments demonstrate its effectiveness.

### Strengths
1. The presentation is well-organized and easy to follow.
2. The study on massive activations is detailed and systematic.
3. The proposed DG strategy is effective and novel.

### Weaknesses
1. The paper lacks comparison with advanced CFG strategies, such as PAG [1], FA-CFG [2], and Semantic-CFG [3]. While such comparisons are not essential, including them would strengthen the experimental validation and enhance the robustness of the evaluation if avaialbe.

2. In Figure 2, the authors claim that massive activations appear at fixed dimensions across all patch tokens. Is this dimension consistent across all layers?

[1] Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance.
[2] FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling.
[3] Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance.

### Questions
See weakness.

### Soundness
3

### Presentation
4

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
This paper systematically studies the role of Massive Activations in image generation and discovers that they play a crucial role in generating fine-grained local details. Building on this observation, the authors propose a simple, training-free method that constructs a “detail-deficient” model by deliberately disrupting these Massive Activations. This degraded model is then used as a negative reference to guide the original network (similar to CFG) to generating images with better detail fidelity.

### Strengths
- The paper is well-motivated and well-structured. It first provide a clear analysis of Massive Activations in Diffusion Transformers, and then proposes a simple and effective method based on this observation to improve image generation quality.

- The proposed approach is simple and training-free. It only involves disrupting the massive activations within a pretrained model to construct a degraded variant, which is then used as a negative reference through a CFG-style guidance to enhance fine-grained visual details.

- Strong results. The proposed method shows strong performance gains and visual improvements

### Weaknesses
- DG relies on fixed-dimension activation patterns and the AdaLN scaling mechanism specific to DiTs. It is unclear whether the approach generalizes to non-transformer or hybrid architectures. Moreover, several recent works have suggested that AdaLN may not be the most optimal solution due to its parameter overhead and have proposed lighter alternatives. It would be valuable to discuss whether DG can be adapted to other schemes.

- No ablation on computational overhead or inference latency. Since the proposed method requires constructing and utilizing a degraded model, it would be helpful to provide a comparison of GPU memory usage and inference time with and without DG to better understand its efficiency trade-offs.


- It is good to also discuss the failure cases and limitations for the method. For example, while the finding that disrupting MAs mainly affects local details rather than semantics is compelling, such disruptions could also introduce unintended side effects. For exmaple I am gussing over-sharpening or loss of texture consistency. Additional experiments or qualitative examples discussing limitations would make the paper more comprehensive.

### Questions
- Can it be adapted to non-DiT or hybrid architectures?
- What is the computational overhead and inference latency of DG?

### Soundness
3

### Presentation
3

### Contribution
3
