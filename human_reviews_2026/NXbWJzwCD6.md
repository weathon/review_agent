# Mono4DEditor: Text-Driven 4D Scene Editing from Monocular Video via Point-Level Localization of Language-Embedded Gaussians

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Editing 4D scenes reconstructed from monocular videos based on text prompts is a valuable yet challenging task with broad applications in content creation and virtual environments. The key difficulty lies in achieving semantically precise edits in localized regions of complex, dynamic scenes, while preserving the integrity of unedited content. 
To address this, we introduce Mono4DEditor, a novel framework for flexible and accurate text-driven 4D scene editing. Our method augments 3D Gaussians with quantized CLIP features to form a language-embedded dynamic representation, enabling efficient semantic querying of arbitrary spatial regions. We further propose a two-stage point-level localization strategy that first selects candidate Gaussians via CLIP similarity and then refines their spatial extent to improve accuracy. Finally, targeted edits are performed on localized regions using a diffusion-based video editing model, with flow and scribble guidance ensuring spatial fidelity and temporal coherence. 
Extensive experiments demonstrate that Mono4DEditor enables high-quality, text-driven edits across diverse scenes and object types, while preserving the appearance and geometry of unedited areas and surpassing prior approaches in both flexibility and visual fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper tackles the text-driven monocular 4D scene editing task. Its core contribution lies in introducing a point-level localization strategy that integrates language-embedded dynamic representations, enabling precise and spatially localized edits. Through extensive experiments across diverse monocular 4D editing settings, the method achieves state-of-the-art performance, clearly outperforming existing baselines. The experimental results are comprehensive and convincingly support the authors’ claims, suggesting that this work has the potential to make a significant impact on the 4D scene editing community.

### Strengths
Strengths:

1. The paper introduces a **novel point-level localization strategy** that enables **precise and localized 4D editing** guided by text instructions. This approach achieves high-quality edits while effectively preserving the appearance and geometry of unedited regions, demonstrating fine-grained control and robustness.

2. The experimental evaluation is **comprehensive and convincing**. The authors provide extensive qualitative results across DAVIS, DyCheck iPhone, DyNeRF, and in-the-wild video datasets, using diverse textual prompts to validate the feasibility and generality of the proposed framework. Moreover, quantitative comparisons against strong baselines such as IN4D and Ctrl-D clearly highlight the superior performance of the proposed Mono4DEditor.

### Weaknesses
Weaknesses:

1. The proposed language-embedded dynamic scene representation closely follows existing works, which somewhat **limits the novelty** of the approach.

2. The authors are transparent about the limitations of the underlying Gaussian representation, which currently **restricts the framework to monocular video inputs**. While this constraint narrows applicability, it is understandable given that monocular 4D editing remains a highly challenging and valuable problem to investigate.

3. The user study could be improved by providing more **detailed evaluations across distinct dimensions**—such as temporal consistency, editing precision, and visual quality. As it stands, the overall human preference results make it difficult to discern which specific aspect contributes most to the performance gain.

### Questions
1. How does the method handle cases where the edited object’s shape extends beyond the original 3D mask? For instance, in a prompt like “turn the cat into a bear”, the resulting object is significantly larger than the source. How does the framework ensure consistent appearance transfer and spatial coherence when the edited geometry exceeds the original mask boundaries?

### Soundness
3

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
4

### Summary
The paper proposes Mono4DEditor, which reconstructs a dynamic 3D Gaussian field from a single input video and augments each Gaussian with quantized CLIP features to enable text-guided, region-selective edits over space and time. The paper compares qualitatively to IN4D and CTRL-D on the iPhone set (Fig. 3) and reports quantitative CLIP similarity and user-study results (Table 1), showing better performance than baselines. The major contribution lies in the Gaussian Localization part.

### Strengths
* Compared with IN4D/CTRL-D, the method shows better temporal consistency and fewer background artifacts on iPhone scenes (Fig. 3 captions and discussion). The texture details of the edited part are also better.
* The overall localization method seems to successfully detect the target regions in the results shown in the paper.
* “Language-embedded Gaussians” via quantized CLIP with an index map and codebook enable efficient semantic queries. 
* The paper is well written and easy to follow

### Weaknesses
* The major concern I have is the technical novelty of this paper. While proposing "Language-embedded Gaussians" in dynamic scene edits, the paper's contribution lies in the localization and masking part. However, the localization part seems kind of redundant. I didn't quite get why we need the “Language-embedded Gaussians" when we already have the video region-level masks and the CLIP embedding. For 3D/4D Editing, masking in images is already enough to edit the scene. If a Gaussian Mask is really needed, it can be obtained from the rendering mask of each view (although we mostly do not need it). To me, the overall approach seems overcomplicated.
* For the improvements in appearance, especially the texture details of the edited part, they seem to directly come from the better video editing model (Figure 9). Thus, the evaluation should be more focused on the localization part, not the appearance quality (Figure 3)
* Some quantitative evaluation is missing, like the background preservation ability (using PSNR, etc.)
* Although the paper claims to work in complicated scenes where localization is difficult, most of the scenes are simple since the target part is obviously the only dynamic object. It would be better to show some more dynamic cases (e.g., editing one small part in addition to the hat in Figure 5)

### Questions
Since poses, depth, masks, and tracks come from third-party tools (MegaSaM, DROID-SLAM, SAM2, etc.), how does the method behave under perturbed inputs (e.g., adding synthetic noise to tracks), and which stage fails first?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work proposed Mono4DEditor, a method that supports precise text-guided local editing of 4D-GS scenes from a monocular video input. Building upon a point-level language-embedded 4D Gaussian representation, Mono4DEditor leverage diffusion-based video editing to achieve highly-precise, temporally and semantically consistent local editing.

### Strengths
1. The work proposed a "point-level" language-embedded 4D radiance field, which supports more accuracy local editing. 
2. Quantization of CLIP feature can theoretically save memory and improve efficiency. Mono4DEdit applied video model for editing to improve temporal consistency
3. Experiments show advantages of 4D local editing via the proposed language-embedded dynamic gaussians, which helps accurately edit text-related target regions while preserving unrelated regions intact.
4. The implementation description is in detail, and possibly for reproduction.

### Weaknesses
1. The proposed framework demonstrates practical value, though its technical contributions are relatively limited. The Reconstruction-Localization-Editing pipeline, while intuitive, is a relatively general framework. The method relies on off‑the‑shelf preprocessing for pose, masks, depth and tracks. The effect of their errors on the results cannot be excluded. The pipeline can accumulate inaccuracies when camera poses drift or when segmentation and depth contain artifacts, which may shift the selected Gaussians or broaden edit regions. Furthermore, it primarily leverages existing diffusion-based video models to tackle the core challenge of maintaining 4D consistency, while such a solution is already well-established in the field.

2. The evaluation section lacks sufficient details and additional experimental validation. For instance, baseline comparisons are only conducted on a single dataset, which limits the generalizability of the results. For further specifics and targeted concerns, please refer to Questions 1-3.

### Questions
1. In the comparisons and evaluation with baselines, it showed 4 editing cases in qualitative comparisons (Figure 3), and used 9 editing cases for user study.  
1.1. How many editing cased were used for CLIP similarity evaluation?  
1.2. Why only one dataset "DyCheck iPhone" was used for comparing with baselines. 

2. For ablation studies,  
2.1. Please also state how many scenes and edits used in quantitative evaluation.  
2.2. Since baselines are based on InstructPix2Pix backbone, in editing model ablation/selection experiments, Mono4DEdit can use InstructPix2Pix as image model editing for comparison, showing video editing models assist better in 4D editing than image editing models.   
2.3. There lacks ablation study of CLIP features quantization. How much efficiency or memory saving got via quantization of CLIP features? If quantization degraded semantics matching?  

3. There is no runtime comparisons in all experiments.

4. The method is restricted to monocular videos. I wonder if Mono4DEdit techniques can also be applied to multi-view video inputs. How will the framework be extended to multi‑camera systems so that reconstruction, point‑level localization, and editing can exploit cross‑view geometry and spatiotemporal consistency while preserving precise edit regions and stable backgrounds?

### Soundness
3

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
3

### Summary
This paper proposes the approach of text-driven editing framework for 4D scene reconstructed from monocular videos. The authors introduces three contributions: (1) unified framework that integrates language-embedded Gaussian representations with diffusion-based video editing models. (2) point-level localization strategy that aims to identify and refine the editable regions. (3) extensive experiments with user study.

### Strengths
Belows are strong points that this paper has:

1. The paper is clear and enjoyable to read, providing a straightforward explanation of the motivation and methodology.

2. The authors well addressed the limitations of previous works and provided corresponding solutions. For example, quantized clip feature strategy is adopted for efficient 4D editing / point-level localization with recall and precision-oriented refinement.

3. The experiments are thoughtfully designed and effectively demonstrate the strength and validity of the proposed method.

### Weaknesses
Belows are weak points that the paper has or questions that reviewer has:

1. As acknowledged in the paper’s limitation section, I share concerns regarding potential errors or failure cases caused by reliance on multiple external priors (e.g., depth, pose, or optical flow). It would strengthen the paper if the authors could illustrate specific failure cases and provide experiments using different external models to analyze the model’s sensitivity or robustness to inaccuracies in these priors.

2. In addition to the above point, I am also interested in the computational efficiency of the proposed approach. Could the authors provide a comparison of training and inference times against baseline methods?

3. Beyond CLIP similarity, are there additional quantitative metrics that could be used to evaluate the proposed method’s performance more comprehensively?

### Questions
Please check above listed in Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
