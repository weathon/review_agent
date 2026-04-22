# MVP: Multi-scale Visual Prompt for Visual AutoRegressive Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Prompt tuning, especially perturbation-based prompt tuning, encounters obstacles in visual generation. On the one hand, the autoregressive paradigm, which provides the most ideal environment for prompt tuning, struggles to model planar concept: traditional autoregressive methods employ raster-scan for image modeling, disrupting the spatial structure of images. On the other hand, perturbation-based prompts work as learnable perturbations in pixel space, and their effectiveness comes at quite a little computational cost, making it difficult to balance performance and efficiency. To address these challenges, we propose Multi-scale Visual Prompt (MVP), a perturbation-based prompt tuning method tailored for visual autoregressive generation with planar concept and efficient information propagation. MVP builds on Visual AutoRegressive (VAR) models with next-scale prediction for capturing planar concept, and introduces prompt tokens in the outermost token frame at each scale for efficient signal control and information propagation. During training, we use increasingly detailed tuning text to facilitate prompt learning. Moreover, MVP extends VAR's capability for text-to-image generation. Extensive experiments validate the effectiveness of MVP. Code is available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MVP, a perturbation-based prompt tuning method, for the VAR models. Instead of adding embeddings, the authors propose to add learnable perturbation tokens on the outermost square frame of the feature map, which they claim it can minimize impact on the image center, and thereby avoiding model feature corruption (line 184). To resolve the lacks of semantics which traditional perturbation prompt tuning has, the authors use multi-level semantic refinement (line 271) with multi-level CLIP contrastive loss (line 287-291), which enables the class-to-image improvements (Table 1) and extending VAR to text-to-image with much less trainable parameters (Table 3) and GPU computation time (Table 4). For the experiments, author has shown MVP improving FID/IS over VAR at 256 x 256 (line 84) and 512 x 512 (line 326) on ImageNet.

### Strengths
$\bullet$ Simple and Effective method: only adding prompts on the outermost frame is simple, which requires minimal architectural modifications. 

$\bullet$ Convincing statistics: MVP achieves better performance than VAR's with less than one percent extra parameters (line 325) and 0.54% of VAR-CLIP training GPU-hours (line 86). Plus, MVP presents better performance over VAR at 256 x 256 and 512 x 512 on ImageNet.

$\bullet$ Interesting Mutli-level semantic refinement: class $\to$ sentence  $\to$ caption CLIP supervision aligned to different generation stages is well-motivated.

### Weaknesses
$\bullet$ Lack of Design Justification: Section 3.1.2 seems to claim that the prompt should live on the outermost frame because a perturbation’s effect on the center decays with distance. Then, the authors uses an attenuation factor $\alpha$ to derive that if a perturbation is placed far away (frame 0), it will impact the center less than the perturbation placed closer (frame c). $Impact(n \to N) = \delta \cdot \alpha^{N-n} \Rightarrow I_0 < I_c$. This is a overly simplified model of propagation which depends only on distance, and I think authors need to come up a full causal analysis of the actual VAR stack in order to validate the claim. Current result lacks broad ablations against other spatial layouts.

$\bullet$ On Line 394, authors claim first-scale prompting hurts class-to-image but helps text-to-image. I think authors need to explain why it hurts class-to-image and providing some diagnostics. 

$\bullet$ The current text-to-image baselines and metrics are limited, which mainly focused on IS/FID, sparse human or robustness checks. Moreover, the generality beyond VAR backbones is under-tested. 

$\bullet$ Lack of Clarity in the paper presentation and organization:
In figure 1, there is a typo, and it should be "framework". In Section 3.1.2, authors defines $\alpha$ as the signal attenuation factor, but on Line 277, $\alpha$ is redefined as a hyperparameter in $(0,1)$. This is an example of abuse of notations. On Line 182, do authors mean to use "denoted" instead of "donated"? I recommend the authors to check the rest of the paper to enhance clarity. 


There are some chances that I misunderstood some parts of this paper, and I welcome corrections and an active discussion from the authors.

### Questions
$\bullet$ Author's current attenuation model $Impact(n \to N)$ assumes distance-only decay. Can author provide me more empirical evidence that shows the real VAR stacks follow this decay? And the center corruption is minimized by the outermost frame? 

$\bullet$ Does the optimal layout change with scale depth, token budget $\tau$, or resolution? 

$\bullet$ Why is the outermost frame optimize versus other spatial layouts? Please provide ablations.

$\bullet$ Regarding the Weakness 2, Can you explain why first-scale prompts degrade class-to-image but helps text-to-image? Please provide the diagnostics that shows the degradation.

$\bullet$ You claim that VAR and "VAR-like" models serve as the target model for MVP (line 142). Can you port MVP to at least one masked-AR and one encoder-decoder image generator to show the architecture agnostic benefits?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MVP (Multi-scale Visual Prompt), a perturbation-based prompt tuning method designed for visual autoregressive (VAR) models that utilize a next-scale prediction mechanism. The core idea is the "square frame prompt," which introduces learnable prompt tokens exclusively to the outermost border of the feature map at each generative scale. This design aims to achieve efficient information propagation while minimizing feature corruption in the image's center. The authors propose a multi-level semantic refinement strategy for training, using increasingly detailed texts (class labels, sentences, captions) and a CLIP-based contrastive loss to supervise prompt learning. The paper demonstrates that MVP improves the class-to-image generation quality of VAR models and, with minimal computational cost, extends their capability to text-to-image synthesis.

### Strengths
- The method is exceptionally efficient. For text-to-image generation, MVP achieves competitive performance using only 0.54% of the training GPU-hours and 0.46% of the trainable parameters compared to a fully trained VAR-CLIP model, making it highly practical.
- MVP demonstrates consistent and notable improvements in generation quality over strong VAR baselines. On ImageNet, it improves both FID and IS scores for class-to-image generation at $256 \times 256$ and $512 \times 512$ resolutions.

### Weaknesses
- The primary weakness is the insufficient justification for the core design choice of the "square frame prompt." The paper lacks a crucial ablation study comparing this specific spatial arrangement to other prompt distributions with an equal number of parameters (e.g., random placement, a sparse grid, or prompts concentrated in the center). Without this, it is unclear if the benefits come from the specific "frame" structure or simply from adding learnable parameters at the periphery.
- The theoretical motivation, particularly the concept of the "planar concept," is vague and not rigorously defined. The connection between this abstract idea and the concrete implementation of a square frame prompt feels tenuous and more like a post-hoc rationalization than a guiding design principle.
- The method introduces a key hyperparameter, the threshold $\tau$, which determines when the prompt shape changes to preserve efficiency. The ablation study in Table 6 shows that the optimal value for $\tau$ is sensitive to model depth and lacks a clear pattern, suggesting it requires expensive, model-specific tuning.
- Using VAR to do T2I seems not proper and lack of convenience.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MVP (Multi-scale Visual Prompt), a multi-scale visual prompting method with a planar concept, specifically designed for visual autoregressive (VAR) models. Through an analysis of information propagation, the authors introduce prompts within the outermost square frame. Combined with increasingly detailed tuning text, MVP enables efficient prompt learning at a low computational cost, enhancing the class-to-image capability of VAR and extending its text-to-image generation ability.

### Strengths
1. The paper proposes a perturbation-based visual prompt tuning method featuring a planar concept and an efficient information propagation mechanism, achieving competitive performance with relatively low training cost.
2. Through theoretical derivation and empirical analysis, the authors show that incorporating prompt features into the outermost frame of the feature map enables more efficient information propagation, and a scale threshold mechanism is introduced to balance performance and computational efficiency.
3. A multi-level semantic refinement strategy is employed during prompt training, enhancing the semantic representation and generation quality of VAR.

### Weaknesses
1. The paper lacks experimental comparisons with embedding-based methods.
2. Theoretical explanation is insufficient: although the authors analyze the advantages of introducing prompt features into the outermost frame of the feature map from the perspective of information propagation, the analysis lacks rigorous theoretical grounding. It would be helpful to include experimental comparisons of incorporating prompts at other positions within the feature map.
3. The experimental setup is constrained by the VAR framework, focusing mainly on the class-to-image task, which is neither the mainstream nor the most representative scenario for visual prompt tuning. Since class labels contain limited semantic information, this setup cannot fully demonstrate the potential of prompts in semantic alignment and generative control. Therefore, it remains unclear whether MVP can maintain its advantages in more typical text-conditioned or open-domain generation tasks.

### Questions
- Missing metrics: the paper does not report LPIPS or PSNR scores.

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
5

### Summary
This paper introduces Multi-scale Visual Prompt (MVP), a novel parameter-efficient tuning (PEFT) method designed to enhance and expand the capabilities of Visual AutoRegressive (VAR) models for image generation. The authors argue that traditional autoregressive (AR) models struggle with a "planar concept" due to raster-scan order, which is addressed by adopting the VAR's next-scale prediction paradigm. MVP employs a perturbation-based prompt tuning strategy where learnable prompt tokens are introduced only at the outermost square frame of the feature map at each scale.

### Strengths
The work demonstrates a few promising aspects, particularly in efficiency and the choice of model:
* The paper explores perturbation-based visual prompt tuning in the relatively unexplored domain of visual autoregressive generation. The core idea of a multi-scale prompt restricted to the outermost frame is an original design choice for next-scale AR models
* The most notable strength is the computational efficiency achieved during tuning. MVP achieves T2I performance comparable to the fully trained VAR-CLIP while requiring only a fraction of the GPU-hours for training, highlighting its potential as an efficient fine-tuning mechanism for large VAR models.

### Weaknesses
* Lack of Rigorous Justification for Prompt Design (Planar Concept and Impact): The theoretical justification for selecting the outermost frame is weak and based on highly simplified models.
* The notion that selecting tokens in the outermost frame is sufficient to capture the "planar concept" based on the geometric principle of three non-collinear points is an oversimplification. It is unclear why three points are sufficient to control the plane of a high-dimensional feature map
* Insufficient Quantitative Validation on Text-to-Image (T2I) Task: While T2I generation is a key claimed contribution, the experimental validation is inadequate and does not use standard practices.
* The multi-scale nature and the multi-stage training are central to MVP, but the ablation studies (Tables 5, 6, 7) are too shallow.

### Questions
* Please provide a rigorous ablation study that compares the performance (FID/IS and efficiency) of the outermost square frame prompt against an equally-sized prompt budget (i.e., the same number of trainable tokens) placed at different strategic locations, such as a compact block in the center of the feature map, and a set of randomly distributed tokens. This is crucial to validate the claims about minimizing central impact and maximizing information propagation efficiency.
*  Provide an ablation to justify the complex, multi-stage tuning text strategy (labels $\rightarrow$ sentences $\rightarrow$ captions). Specifically, what is the performance difference when only the most detailed tuning text (captions) is used across all stages? Does the incremental complexity of the tuning strategy genuinely contribute to the final performance?

### Soundness
3

### Presentation
2

### Contribution
2
