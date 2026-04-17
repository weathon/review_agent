# Adaptive Projected Guidance for Controllable Instruction-Guided Image Editing

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Instruction-guided diffusion models have demonstrated strong capabilities in generating targeted image edits based on diverse textual prompts. A fundamental challenge in this setting is achieving the right balance between adhering to textual instructions and preserving the original content of the input image. InstructPix2Pix (IP2P) addresses this by applying separate classifier-free guidance (CFG) terms to the text and image conditions, each scaled independently. However, this limited parametrization restricts user control, as increasing one guidance scale often causes the corresponding condition to dominate the output, resulting in imbalanced edits. Independently, Adaptive Projected Guidance (APG) was recently introduced to mitigate inherit limitations of CFG at high guidance scales in text- and class-conditioned diffusion models, reframing CFG as a gradient ascent process with decomposed guidance directions and improved signal control. In this work, we present IP2P-APG, a plug-and-play extension of IP2P that repurposes APG to improve the balance between instruction adherence and content preservation in image editing tasks. IP2P-APG significantly expands the controllable parameter space, allowing users to have more precise control over the editing process. Moreover, by enabling the use of higher guidance scales without introducing artifacts or compromising fidelity to the original content, IP2P-APG achieves a more effective trade-off between textual alignment and content preservation. Extensive experiments across multiple generative backbones and datasets demonstrate that our method consistently produces more realistic and instruction-faithful edits, without additional training and with negligible computational overhead. Code will be released after the review process.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to replace the conventional CFG with adaptive projected guidance, therefore mitigating the inherent limitations of CFG at high guidance scales. It expands the controllable parameters space, and the editing performance is improved across different diffusion backbones according to the quantitative results.

### Strengths
1. It adopts the adaptive guidance into the IP2P framework, which can enable high guidance scale for text condition. The proposed method can better preserve the original layout compared to the original IP2P.

2. The idea is straightforward and intuitive.

### Weaknesses
1. I consider this work as a simple combination of the existing adaptive projected guidance and the IP2P framework. Although it can improve the performance of original IP2P, there is not enough technical novelty or new insight brought by the paper itself. This is the fundamental limitation of this paper as I see it.

2. The editing results showcased by the paper are not strong enough. The example in the teaser under s_i=1.4 does not look good enough. The two examples in Figure 3 are also not competitive enough compared to the baselines. I would also suggest the authors to look for more visually pleasing [input image + instruction] examples to replace several existing ones shown in the paper. Input images with cleaner background and more obvious editing changes are recommended.

### Questions
It looks like that the images in Figure 4 are not scaled to the correct range to visualize, so they have the distorted overflowed colors. Could the authors check the images again?

### Soundness
2

### Presentation
2

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
To address the imbalanced edits on InstructPix2Pix (IP2P) applying separate classifier-free guidance (CFG), the paper present IP2P-APG, a plug-and-play extension of IP2P that repurposes APG to improve the balance between instruction adherence and content preservation in image editing tasks. It expands the controllable parameter space, allowing users to have more precise control over the editing process.

### Strengths
- This paper is of high practical value. The paper repurposes APG to improve the balance between instruction adherence and content preservation in image editing tasks. 
- This paper designs a rich set of experiments to demonstrate the advantages of the method, and compares the effects of different hyperparameters.

### Weaknesses
- The contribution lacks novelty. The proposed method builds entirely upon the existing InstructPix2Pix (IP2P) and Adaptive Projected Guidance (APG) frameworks, amounting to a direct substitution of APG for the original classifier-free guidance component in IP2P without introducing any additional conceptual or algorithmic innovations.
- The paper exhibits a substantive content deficit. Beyond the experimental section, its technical work is confined to Subsection 2.4, where it is further diluted by an excessive number of definitional equations that add little interpretive value. Overall, the paper offers an unduly prolix account of prior work and presents its key derivations with unduly prolix detail, raising the suspicion of padding.

### Questions
- Could the authors please clarify the theoretical innovations that may have been missed by me, especially principles or derivations distinct from the original Adaptive Projected Guidance formulation?
- It is recommended that the paper be restructured to allocate substantially more space to a detailed exposition of the present work. Prior-work summaries should be condensed so that the core methodological design, theoretical justification, and ablative analyses receive adequate emphasis, thereby enabling readers to assess the contribution without wading through disproportionately lengthy background sections.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes IP2P-APG, a training-free extension to InstructPix2Pix (IP2P) that improves controllability in instruction-guided image editing by incorporating Adaptive Projected Guidance (APG). The key idea is to decompose the text and image guidance signals into parallel and orthogonal components, and regulate them separately using momentum, normalization, and projection, which allows users to adjust how strongly the model follows the instruction versus how much it preserves the original image. The proposed method is plug-and-play, requires no retraining, and works across various diffusion backbones Experiments show that IP2P-APG consistently yields promising editing results.

### Strengths
S1. Quantitative results in the MagicBrush and Emuedit test set are promising, and authors have reported extensive experimental results using both DiT-based architecture (SD3) and U-Net based architecture (SD 1.5). 

S2. The authors clearly verified the effects of each component (in terms of hyperparameter isolation) both qualitatively (Fig. 4, Fig C.1. - C.4.) and quantitatively (Table 3). It strengths the proposed thesis’ requiredness.

### Weaknesses
W1. The novelty of the proposed method is limited. The authors proposed the IP2P-APG framework, which combines (the tailored version of) adaptive projected guidance (APG) with instructpix2pix (IP2P). However, two components (IP2P, APG) are already defined and verified in the previous works, and I think it’s hard to find novel components and core contributions within the proposed method. This is my major concern with this paper. If there are any other novel contributions, please emphasize it; otherwise, please re-organize the method section to strongly support the paper’s contributions.

W2. The experiment section only focuses on the object change and addition tasks, such as “Add kids playing …”. I was wondering if the method is applicable to more difficult tasks, such as 1) object duplication, 2) object deletion, and 3) enlarging or shrinking the object size. Extensive experiments with additional tasks is required.

W3. As mentioned in S2, authors provided brief analysis on hyperparameters and each proposed component. I saw that the proposed method contains (at least) 8 hyperparameters ($\beta_T$, $\beta_I$, $r_T$, $r_I$, $\eta_T$, $\eta_I$, $s_T$, $s_I$), and the sensitivity of each hyperparameters should be also analyzed. I was wondering how the result is quantitatively affected by the variation of each hyperparameter, beyond ablation. In addition, from the last column of Figure 4, I think the proposed method is somewhat sensitive to $\beta$ hyperparameter. Is there any plausible strategy to manually tune these hyperparameters?

W4. More qualitative comparison with baseline is required. The main paper shows only four examples (Included in Figure 3 and 5), and more qualitative results that emphasize the strengths and effectiveness of the proposed IP2P-APG across baselines is required. 

W5. The authors reported ablation study results using SD v1.5 checkpoint. Could authors kindly provide the ablation study results with SD v3 (*i.e.* Transformer-based diffusion model) checkpoint?

### Questions
Please check the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper applies Adaptive Projected Guidance (APG) to intruction-guided diffusion model (IP2P) as a plug-and-play framework. The model-agnostic framework shows improvement across benchmarks.

### Strengths
- The work shows that APG works on editing model such as IP2P

### Weaknesses
- Lack of technical novelty since the APG paper already shows generalization across image generation models.
- The method seems fragile qualitatively in Figure 1 as $s_I$ works at 1.1/1.5 but does not work at other values.

### Questions
- Is there technical difference in applying APG to image generation models vs image editing models.

### Soundness
2

### Presentation
3

### Contribution
1
