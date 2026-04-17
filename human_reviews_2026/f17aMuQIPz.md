# INPO: Image-based Negative Preference Optimization for Concept Erasure in Text-to-Image Diffusion Models

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Text-to-image diffusion models have achieved remarkable generative performance, yet they are susceptible to memorizing and reproducing undesirable concepts, such as NSFW content or copyrighted material. While concept erasure has emerged as a promising approach to remove undesirable concepts from pre-trained models, existing methods still suffer from prompt-dependence, architecture-dependence, and unstable training dynamics, which limit their effectiveness and generalization. In this work, we propose Image-based Negative Preference Optimization (INPO), a novel model-agnostic framework for concept erasure that unifies joint image–text supervision under a principled preference optimization paradigm. By formulating the target concept as a negative preference, INPO inherits the stable optimization dynamics of Negative Preference Optimization (NPO), thereby mitigating the instability of prior gradient-ascent-based methods. To achieve precise and controllable erasure, INPO further incorporates a concept mask for localized suppression and an adaptive negative scaling strategy that dynamically modulates optimization strength according to erasure progress. Extensive experiments on the latest FLUX model demonstrate that INPO achieves precise and consistent erasure across a variety of tasks, including object, IP, style and NSFW content, while preserving the model’s overall generative capabilities, highlighting the robustness, reliability and practical applicability of INPO for safe and controllable image generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed a method to erase concepts from Diffusion based generative models using an adaptation of negative preference optimization. The report results on nudity, objects and style erasure on FLUX.1-Dev.

### Strengths
- "Adaptive Erasure Trajectories" in section 4.3 is very interesting and makes the approach easily useable without defining a target concept. 
- The results on applying NPO for concept erasure are promising and can be beneficial for future research.  
- The paper in general is well written and is easy to follow.

### Weaknesses
- The authors talk about "Architecture-dependence" in the introduction as motivation behind their approach but proceed to only show results on FLUX. 
- The experiments evaluations are lacking. I'm in particular interested in how the approach benchmarks on SDv1.4 on adversarial prompt datasets (Ring-A-Bell, MMA Diffusion, etc). Especially in comparison with newer approaches such as AdvUnlearn, AGE. 
- The paper talks about prompt-dependence but fails to cite or acknowledge [1].
- There are many relevant works that have not been cited. While I am not listing all here, please add relevant works. 

[1] Pham, Minh, et al. "Prompt-Agnostic Erasure for Diffusion Models Using Task Vectors."

### Questions
- How do you find the concept masks M? 
- What is the performance on multi-concept erasure such as on celebrity 100 dataset from MACE [1]? 
- Have you re-run red-teaming attacks on FLUX or have you reused prompt sets that were optimized for SDv1.4? I'm surprised to see such low numbers for the baseline nudity generation on FLUX. 

[1] Lu, Shilin, et al. "Mace: Mass concept erasure in diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

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
This paper proposes a preference optimization approach for concept erasure in text-to-image diffusion models. This paper formulates the unwanted concept as a negative preference, and then applies negative preference optimization to steer the model to remove the unwanted concepts. The proposed method also designs concept masks and an adaptive negative scaling strategy to improve the unlearning quality. The experiments are conducted on several types of concept erasure tasks.

### Strengths
1. Concept erasure is crucial and practical for real-world trustworthy generative model developments.
2. The proposed method is reasonable to formulate the concept erasure as a negative preference optimization problem.
3. The overall paper is easy to follow and well structured.

### Weaknesses
1. While the proposed method can erase the unwanted concepts described by the target prompts, it remains unclear whether the proposed method is able to handle the rephrased prompts that can be used to recover the target concepts. It would be beneficial to discuss or clarify this potential robustness concern. 
2. The proposed approach relies on Eq.14 to preserve the unrelated concepts that are not affected. However, how to decide the preservation set? It seems impractical to cover every concept in this preservation set. More clarifications or explanations for this concern are helpful.

### Questions
1. Can this proposed method be applicable to handle multi-concept erasure scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces INPO, a framework for concept removal in text-to-image diffusion models. It extends Negative Preference Optimization (NPO), originally proposed for LLM unlearning, to the text-to-image diffusion setting, where the target concept to remove is treated as the negative preference.  The method further incorporates (1) a concept mask to spatially focus the loss on relevant regions and (2) an adaptive negative scaling strategy that reduces gradient strength once the target concept is sufficiently suppressed.

### Strengths
* The proposed adaptation of NPO to text-to-image diffusion models is intuitive and empirically effective. It achieves strong performance across diverse erasure types (object, IP, style, NSFW) and remains stable on modern architectures like FLUX.
* Evaluation against red-teaming attacks (MMA-Diffusion, P4D, etc.) is impressive and shows practical robustness.

### Weaknesses
1. It would be great to have an evaluation to include prior benchmarks (as used in ESD, CA, or EA) with a broader set of prompts across multiple concepts. In addition, listing the exact prompts used for current experiments would further enhance reproducibility.
2. An ablation on the role of eta, gamma, and tau (Eq. 12) would be useful, especially since Table 5 in the Appendix shows different settings (e.g., η=3 for style removal vs. 1 for other types). This would help assess the robustness and interpretability of the adaptive scaling.
3. Section 3.2 defines a prior loss over concepts, c′,  to preserve, but it is not specified which concepts or datasets are used for this (COCO? random prompts?). Adding more details regarding this would help improve the reproducibility of the method. 
4. Reporting some quantitative measure (e.g., CLIP similarity to neighboring concepts before/after erasure) would clarify whether INPO doesn’t affect any nearby concepts, e.g., Monet style when removing Van Gogh. 

 Minor points:
1. It's unclear what the mask should be for abstract or global attributes such as artistic style or NSFW tone. A clarification and any ablation regarding this would help.

### Questions
Please look at the weakness section. Specifically, if the evaluation checks the generation on a wide variety of prompts for each target concept. Or maybe including the performance on the same set of target concepts and prompts as done in previous works. In addition, clarifying some of the implementation details would help strengthen the paper and improve its reproducibility.

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
4

### Summary
This paper builds upon Negative Preference Optimization (NPO) and introduces Image-based Negative Preference Optimization (INPO), a model-agnostic framework for concept erasure in text-to-image diffusion models. To achieve precise concept erasure, it introduces a concept mask and an adaptive negative scaling strategy that dynamically adjusts the erasure strength based on the model’s learning state. In addition, it includes a prior preservation loss to retain the model's generative capabilities for non-target concepts.
Experiments are conducted across various domains, including objects, copyrighted content, artistic styles, and inappropriate content.

### Strengths
- The idea of using relative score difference as an indicator to adaptively control the erasure strength is novel.
- The paper provides comprehensive comparison with prior concept erasure methods under diverse red-teaming settings, showing consistent results.

### Weaknesses
- The overall contribution and novelty are limited. The work primarily adapts NPO’s objective to diffusion models for concept erasure, and the use of the concept mask is relatively straightforward.
- The paper lacks theoretical or empirical validation for the claimed instability of gradient-ascent-based unlearning methods in diffusion models. In the original NPO paper, a toy experiment was conducted to compare forget quality, model utility, and divergence rate between gradient-ascent- and NPO-trained LLMs, but similar analyses are missing here.
- The evaluation scope for object, IP, and identity erasure is narrow. For object erasure, only 3 objects are tested, while prior works (e.g., MACE, ESD, UCE, RECE, EAP) evaluated 10 objects from either CIFAR-10 or Imagenette and reported per-object and overall results. Similarly, IP and identity erasure are evaluated on only 1 or 2 concepts, which is insufficient to demonstrate effectiveness. Moreover, IP and identity erasure should be treated as different domains and evaluated separately.
- Although the paper claims that the proposed method is model-agnostic, it excludes many major methods (e.g., MACE, RECE) that have been tested on U-Net-based diffusion models (e.g., Stable Diffusion v1.4). Including experiments on such architectures would strengthen the claim of generality. In addition, these methods are not restricted to U-Net-based models. They can still be applied to DiT-based diffusion models as long as the model architecture has a cross-attention mechanism between image and text features.
- How long does the proposed method take to erase a target concept from a diffusion model?
- (minor) How are the hyper-parameters $\eta$, $\gamma$, and $\tau$ chosen?
- (minor) Typos:
  - Line 173: “acent” -> “ascent”
  - Line 178: “defined the as NPO” -> “defined the same as NPO”
  - Line 186: “As Eq. 3” -> “As shown in Eq. 3”

### Questions
See the weaknesses above

### Soundness
2

### Presentation
2

### Contribution
1
