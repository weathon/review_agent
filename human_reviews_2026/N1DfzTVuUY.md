# Training-Free Text-Guided Color Editing with Multi-Modal Diffusion Transformer

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Text-guided color editing in images and videos is a fundamental yet unsolved problem, requiring fine-grained manipulation of color attributes, including albedo, light source color, and ambient lighting, while preserving physical consistency in geometry, material properties, and light-matter interactions. Existing training-free approaches provide broad applicability across editing tasks but struggle with precise color control and often introduce visual inconsistency in both edited and non-edited regions. In this work, we present ColorCtrl, a training-free color editing method that leverages the attention mechanisms of modern Multi-Modal Diffusion Transformers (MM-DiT). By disentangling structure and color through targeted manipulation of attention maps and value tokens, our method enables accurate and consistent color editing, along with word-level control of attribute intensity. Our method modifies only the intended regions specified by the prompt, leaving unrelated areas untouched. Extensive experiments on both SD3 and FLUX.1-dev demonstrate that ColorCtrl outperforms existing training-free approaches and achieves state-of-the-art performances in both edit quality and consistency. Furthermore, our method surpasses strong commercial models such as FLUX.1 Kontext Max and GPT-4o Image Generation in terms of consistency. When extended to video models like CogVideoX, our approach exhibits greater advantages, particularly in maintaining temporal coherence and editing stability. Finally, our method generalizes to instruction-based editing diffusion models such as Step1X-Edit and FLUX.1 Kontext dev, further demonstrating its versatility. Here is the website.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents ColorCtrl, a training-free, text-guided color editing method for images and videos built on Multi-Modal Diffusion Transformers (MM-DiT). The core idea is to disentangle “what should stay” (structure, materials, lighting geometry) from “what should change” (albedo or light color) by (i) swapping the vision-to-vision quadrant of the attention map from a source branch to a target branch to preserve geometry and view, (ii) extracting an edit mask from vision-to-text attention and copying vision value tokens outside the mask to preserve non-edited colors, and (iii) enabling word-level attribute intensity control by pre-softmax scaling in the text-to-vision attention region. Extensive comparisons on SD3 and FLUX.1-dev, plus CogVideoX for video, indicate state-of-the-art training-free performance on PIE-Bench and a new ColorCtrl-Bench; the method reportedly surpasses strong commercial systems (FLUX.1 Kontext Max, GPT-4o Image Generation) on consistency while remaining model-agnostic and compatible with instruction-based editors (Step1X-Edit, FLUX.1 Kontext dev).

### Strengths
* Originality: The paper adapts attention-control editing to MM-DiT with a clear decomposition of attention quadrants: vision-to-vision for structure preservation, vision-to-text for mask extraction, and text-to-vision for controllable attribute strength. This differs from U-Net cross-attention methods and prior MM-DiT controls (e.g., DiTCtrl) by operating directly on attention maps and value-token routing without training.   

* Quality: The mechanism is well specified: two-branch unrolling with cached source features; mask from vision-to-text attention; copying only vision value tokens for non-edited areas; and pre-softmax token-specific scaling for attribute re-weighting. Ablations show that structure preservation and color preservation cumulatively improve consistency metrics with minimal loss in CLIP alignment.   

* Clarity: The paper provides a task formulation grounded in a rendering-style decomposition (G, L, A, S, C), detailed pipeline figures, and explicit implementation/inference settings (steps, CFG, mask threshold, fixed seeds). These details aid reproducibility.

### Weaknesses
* Masking and subject detection reliance: The evaluation and parts of the pipeline hinge on subject keywords and a fixed attention-threshold ($\epsilon=0.1$) for mask extraction; robustness to threshold choice, ambiguous subject words, or multi-object scenes is not deeply analyzed.  

* Claims versus limitations: The paper acknowledges failures when the base model mislocalizes targets or confuses attributes (e.g., trees or lipstick casing). More systematic characterization of such failure modes—especially under crowded scenes, glossy materials, or colored lighting—would be valuable.

### Questions
* How sensitive is performance to the mask threshold ϵ and to the choice of the “blended word” subject token? Please provide a small sensitivity analysis (e.g., ±0.05 around 0.1) and results for ambiguous subjects or multi-instance scenes. 

* The benchmark uses noise-to-image generation to isolate editing. Can you complement this with a real-image inversion benchmark that reports reconstruction error and edit success jointly, to reflect common editing workflows?

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
This paper introduces ColorCtrl, a training-free framework for text-guided color editing using Multi-Modal Diffusion Transformers (MM-DiT). The method leverages attention maps from MM-DiT to extract semantic masks and reweight attention for localized color edits. The authors claim improved controllability and consistency without requiring model fine-tuning.

### Strengths
1. Originality: The idea of using MM-DiT attention maps for color-specific editing is novel in the context of training-free pipelines.The integration of word-level control adds granularity not commonly seen in prior works.
2. Quality: The method is well-implemented and evaluated across multiple domains (images, videos, instructions). Results show high semantic consistency and localized edits, outperforming several baselines.
3. Clarity: The paper is generally well-written, with clear motivation and structured methodology. Visual examples and comparisons are effective in illustrating the method’s capabilities.
4. Significance: The approach is impactful for real-world applications where retraining is costly or infeasible. It contributes to the growing field of training-free multimodal editing, pushing the boundaries of what can be achieved with pre-trained models.

### Weaknesses
1. Limited Novelty: While the use of MM-DiT is novel for color editing, similar pipelines have applied attention-based editing in other contexts. TextCrafter also leverages attention maps from MM-DiT to extract semantic masks and reweight attention for image editing. Add-It also leverages attention maps from MM-DiT to extract semantic masks. 
2. Insufficient Analysis of Key Components: 
(1) The mask extraction process is underexplained: How are attention maps selected? What thresholding strategy is used? How robust is the mask across different prompts?
(2) The attention reweighting mechanism lacks detailed discussion: How is the reweighting computed?
(3) A more thorough ablation study would help isolate the contribution of each component (e.g., mask quality, reweighting strategy).

### Questions
1. Could you elaborate on how the semantic mask is extracted from MM-DiT attention maps? Which layers are used? What criteria or thresholds are used to binarize or localize the mask?
2. How is the attention reweighting applied during inference? Is it uniform across all heads/layers or selectively applied?
3. How sensitive is the editing quality to the mask threshold and prompt?

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
This paper proposes a training-free color editing method for images and videos based on pretrained T2I/T2V MMDiT models. It disentangles the color and structure within the edited region, and achieves accurate and natural color editing without touching unrelated regions or attributes. The proposed method is evaluated on extensive examples with various base models, surpassing strong commercial models. A new benchmarking protocol is proposed by adopting the PIE-bench prompts on generated images and adjusted metrics. The proposed method also applies to videos and instructional image editing models seamlessly.

### Strengths
- The proposed method is dedicatedly designed for color editing and exclude other factors, achieving clean color editing results with faithfully maintained content.

- The proposed method and benchmark highlights the importance of producing reasonable color edit with similar lighting etc. environmental conditions, over the traditional standard semantic CLIP alignment, making the output results more realistic and natural.

### Weaknesses
- [Major] The only editing quality metric is still CLIP similarity, which doesn't align with the claim that CLIP usually leads to over saturated preference as it lacks details. The major upgrade of metrics only focus on the structure preservation. Given the focus of the paper, some improved editing metrics are necessary to solidate the evaluation. For example, at least aesthetics/harmony can be tested to show the edited colors fit in the environment well. Maybe other color spaces like HSV could also help to decompose? Texture preservation can also be considered (e.g. semantic not impacted), while Canny is relatively rough especially when the threshold is not low enough. For CLIP similarity, it could also potentially help if it is calculated separately between color words and object words etc.

- [Major] The proposed method is tested mainly on generated images instead of real images, and the proposed benchmark also emphasizes this. Although a dedicated section is provided for application on real images, the major quantitative results are compared on synthetic data. It is claimed that testing on generated images can calibrate the impact of inversion quality etc., but in practice it's still important to evaluate on real data, in terms of both performance and methodology. The performance of all methods will be impacted by inversion, while some methods might cooperate better or worse with inversion, and the difference matters for real usage.

- UNet-based models sometimes feature their tighter spatial correspondence over transformers. One or two strong baselines could be involved.

### Questions
- For the failure cases described in Sec. B.5, would there be additional approach to improve? For example, would it be feasible to locate the editing region with the full object in the source image caption, i.e. use "red trees" or "red lipstick" instead of "trees" or "lipstick" to locate the desired region?

- Would the proposed method be applied for other attribute editing like color that are uniform and doesn't change structures, e.g. textures, reflection/transparency? If so the scope of this paper could be largely extended.

### Soundness
2

### Presentation
3

### Contribution
2
