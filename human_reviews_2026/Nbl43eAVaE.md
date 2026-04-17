# MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
We tackle the task of any-reference video generation, which aims to synthesize videos conditioned on arbitrary types and combinations of reference subjects, together with textual prompts. This task faces persistent challenges, including identity inconsistency, entanglement among multiple reference subjects, and copy-paste artifacts. To address these issues, we introduce MAGREF, a unified and effective framework for any-reference video generation. Our approach incorporates masked guidance and a subject disentanglement mechanism, enabling flexible synthesis conditioned on diverse reference images and textual prompts. Specifically, masked guidance employs a region-aware masking mechanism combined with pixel-wise channel concatenation to preserve appearance features of multiple subjects along the channel dimension. This design preserves identity consistency and maintains the capabilities of the pre-trained backbone, without requiring any architectural changes. To mitigate subject confusion, we introduce a subject disentanglement mechanism which injects the semantic values of each subject derived from the text condition into its corresponding visual region. Additionally, we establish a four-stage data pipeline to construct diverse training pairs, effectively alleviating copy-paste artifacts. Extensive experiments on a comprehensive benchmark demonstrate that MAGREF consistently outperforms existing state-of-the-art approaches, paving the way for scalable, controllable, and high-fidelity any-reference video synthesis. The code and video demos are available in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes MAGREF, a unified framework for any-reference video generation, addressing challenges such as identity inconsistency, subject entanglement, and copy-paste artifacts. The method integrates masked guidance and subject disentanglement mechanisms, along with a four-stage data pipeline. Experiments show that MAGREF achieves superior performance over existing methods, demonstrating its effectiveness for scalable for  subject-driven video synthesis.

### Strengths
The citations are comprehensive, suggesting that the authors are well-versed in the field of subject-driven generation.


The experiments are extensive, encompassing various open-source and commercial baselines, including both single-subject and multi-subject settings. These results clearly demonstrate the effectiveness of MAGREF.

### Weaknesses
If the base image-to-video model does not use channel concatenation to fuse the reference images, the effectiveness of this method would be limited.

### Questions
(1)  The overall spatial resolution for all subjects is fixed. During inference, how are reasonable spatial resolutions allocated to different subjects? If a subject is very large, such as the background, this implies that the spatial resolution for other subjects will be scaled down, which may lead to information loss.

(2) In Equation (6), is $M^k_{\text{sub}}$  a typographical error? Should it instead be written as $M^i_{\text{sub}} $?

(3) Is Equation (6) also applied during inference? In addition, can Equation (6) be extended to scenarios involving subject interactions, where the subject masks may overlap?

Addressing my concerns would lead me to reconsider and potentially raise my score.

### Soundness
4

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
This paper proposes MARGREF,  a unified masked guidance design by combining region-aware masking with pixel-wise channel concatenation, to inject reference cues at the channel level. It further proposes a subject disentanglement mechanism that maps text semantics to their corresponding visual regions, cleanly separating identities and mitigating cross-reference confusion without extra identity modules. The model achieved SOTA performance in consistent subject-driven video generation.

### Strengths
- This paper proposed a novel structure to condition video generation on multiple images by combining multiple images into one. It also proposed data-pipeline to collect large-scale
- The proposed subject disentanglement mechanism is novel and effective in text-prompt alignment for different subjects in the image.
- The results show that the proposed framework is better than other baselines. Ablation study proves that all the proposed module is meaningful.
- The paper is well-written and easy to follow.

### Weaknesses
The paper is in general good, some minor points:
- The computational power is not mentioned e.g. how many GPUs have been used
- No details about the dataset e.g. source/size and possible privacy problem for human face data

### Questions
- What is the value of C_m?
- What is the base T2V model that MAGREF is fine-tuned on?
- For the composed reference image, does the resolution matter?
- For the composed reference image, does the organization matter?
- In 4.3, what is result in Table 3 / 4 tested on? the score seems different from Table 1 / 2?

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
4

### Summary
To solve the issue of any-reference video generation task, this paper proposes a unified framework, MAGREF. The framework contains region-aware masking mechanism and subject disentanglement mechanism. Experiments show the scalable, controllable, and high-fidelity any-reference video synthesis results.

### Strengths
-  The writing is easy to understand, and the painting is well-drawn.
-  The proposed region-aware masking method preserves subject identity without backbone changes.
-  Experimentally, the paper achieves best single-ID and multi-subject score.

### Weaknesses
- 1.	In this paper, multi-subjects are introduced into the videos through a blank canvas. Several subjects are directly added to the canvas with their pixels values. Although this function is useful, my major concerns are listed below:
 -      a)	The canvas size is limited. How many subjects can be placed on the canvas without harming the model’s generation ability?
 -      b)	The positions of these subjects are randomly shuffled during training, in my opinion, the locations of different subjects may contain implicit relationship, such as decide the distance of two subjects in the generated videos. However, this is not discussed in the paper and related ablations are not considered.
 -      c)	Similar to the absolute locations, the scale of each subject in the canvas is still missing. Because such model relies on the explicit vision cues to catch the details of the reference subjects, the scale is an important factor that needs to be considered.
- 2. In subsection “Pixel-wise channel concatenation”, the composited image I_{comp} is encoded by VAE encoder and then concatenated with noised video latents along the channel dimension. I cannot find technically something new that differs from existing methods. Existing methods also apply the pixel-wise image/video with VAE encoder and concatenate the latent with noised video.
- 3. The paper claims that the methods support arbitrary subject categories in Lines 098-103, but in their methods, it is unclear how the model support such ability.
- 4. The qualitative comparison in Fig. 5 seems cannot demonstrate the superiority of the proposed methods, such as compared with close-source method Kling1.6 or the open-source method VACE. Besides, why does the multi-subject result of Skyreels method shows show a poor first frame?
- 5. The experimental results lack more deep analysis to explain why the proposed methods outperform the previous methods in qualitative and quantitative results.
-6. The failure cases and more analyses should be discussed.

### Questions
See above weaknesses

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
5

### Summary
The paper introduces MAGREF, a framework for any-reference video generation that combines region-aware masked guidance with subject disentanglement to support arbitrary combinations of human, object, and environment references. It achieves superior identity preservation and visual quality compared to open-source and proprietary baselines.

### Strengths
1. Technically sound: Masked guidance and pixel-wise concatenation are simple yet effective extensions of I2V backbones.

2. Comprehensive results: The experiments and ablations clearly support the claimed improvements.

### Weaknesses
1. My major concern is that the pixel-wise channel concatenation may limit scalability when the number of reference subjects grows. It is unclear how the model handles more subjects simultaneously. Would temporal or latent-level concatenation yield more flexible conditioning in such cases?
2. While pixel-wise concatenation effectively preserves subject appearance, it may inherently limit global-level customization. Since it injects spatially grounded features, the model mainly captures concrete subject geometry rather than abstract global styles (e.g., tone, lighting, art style). Temporal or latent-level fusion could offer more flexibility for such style control. Can this framework be extended to global element customization, such as atmosphere, or texture?

### Questions
As seen in weakness

### Soundness
3

### Presentation
3

### Contribution
3
