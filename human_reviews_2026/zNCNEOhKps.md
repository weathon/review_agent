# PICS: Pairwise Image Compositing with Spatial Interactions

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Despite strong single-turn performance, diffusion-based image compositing often struggles to preserve coherent spatial relations in pairwise or sequential edits, where subsequent insertions may overwrite previously generated content and disrupt physical consistency. We introduce PICS, a self-supervised composition-by-decomposition paradigm that composes objects in parallel while explicitly modeling the compositional interactions among (fully-/partially-)visible objects and background. At its core, an Interaction Transformer employs mask-guided Mixture-of-Experts to route background, exclusive, and overlap regions to dedicated experts, with an adaptive $\alpha$-blending strategy that infers a compatibility-aware fusion of overlapping objects while preserving boundary fidelity. To further enhance robustness to geometric variations, we incorporate geometry-aware augmentations covering both out-of-plane and in-plane pose changes of objects. Our method delivers superior pairwise compositing quality and substantially improved stability, with extensive evaluations across virtual try-on, indoor, and street scene settings showing consistent gains over state-of-the-art baselines. Code and data are available at https://github.com/RyanHangZhou/PICS

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the task of composing objects in parallel by explicitly modeling the compositional interactions among visible objects and background. The core is an Interaction Transformer that employs mask-guided Mixture-of-Experts to route background, exclusive, and overlap regions to dedicated experts. The proposed method outperforms baselines in inserting multiple objects.

### Strengths
1. I appreciate the task and the motivation, that sequentially insert objects into an image will overwrite previously generated content and inserting multiple objects is a meaningful task itself.
2. The visualized results seem good.
3. The paper is overall well-written.
4. I appreciate the interaction diffusion network as well as the interaction transformer block, that sounds reasonable.

### Weaknesses
1. It seems that most cases shown in the paper are inserting two objects. I wonder if there's any direct extension to multiple objects since previous methods can perform it sequentially. As the authors also claim in the limitation "restricted to two objects".
2. It seems that during training, the model takes in modal part (visible part) of an object? But I am assuming that during in-the-wild inference, the ideal input would be unoccluded instances like those shown in Fig. 5. Could the authors attach more examples as well as explaining how the proposed method can generalize well to this kind of setting?

### Questions
See above

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
5

### Summary
PICS is a diffusion-based framework for composing two objects into a background while maintaining spatial and visual consistency. The key idea is to perform parallel compositing of two objects rather than sequential editing. This paper introduces an Interaction Transformer with mask-guided experts and adaptive α-blending to handle overlapping regions and preserve boundaries, along with geometry-aware augmentations for robustness to pose changes.

### Strengths
The paper identifies a real issue, multi-turn compositing instability, and addresses it with a structured, parallel formulation. The task definition is well-motivated.

The mask-guided MoE and α-blending are simple but intuitive mechanisms to ensure boundary consistency.

### Weaknesses
It seems that the method is explicitly limited to pairwise (two objects) compositing.  Could PICS perform well on three-object, four object compositing? Please give me several visual examples if available.

Baselines are outdated. To the best of my knowledge, some open-source object insertion models can perform much better than the baselines you have chosen, such FreeCompose [1], OmniPaint [2], and Insert Anything [3]. Could you provide a comparison of these models?

The proposed mask-guided MoE and multiple cross-attention blocks likely increase inference cost. Could you report detailed runtime and parameter counts to quantify the overhead?

Figure 8 briefly shows degradation of compositing, but the causes are not analyzed in detail.

There is no visualization or strong justification for why geometry-aware augmentation improves compositing quality.



[1] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior (ECCV 2024)

[2] OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting (ICCV 2025, release time: 2025.03)

[3] Insert Anything: Image Insertion via In-Context Editing in DiT (arXiv 2025, release time: 2025.04)

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
4

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
This paper proposes PICS, a diffusion-based method for pairwise image compositing that shifts from sequential to parallel composition of two objects into a background, explicitly modeling spatial interactions like support, containment, occlusion, and deformation.
The main technical contribution includes an Interaction Transformer with mask-guided Mixture-of-Experts for region-specific processing (background/exclusive/overlap) and an adaptive alpha-blending mechanism for coherent overlaps, with geometry-aware augmentations for pose variations. The model is trained on ~1M samples from datasets including LVIS, Objects365, and VITON-HD. The experimental validation conducted by the authors claims the gains in compositing tasks on numerical scores, visuals, and user study feedbacks.

### Strengths
This paper offers a novel parallel compositing paradigm and MoE-based transformer that targets pairwise relations, addressing instability in multi-turn diffusion edits. The paper is well-written and easy-to-read. The experimental design is reasonable.

### Weaknesses
1. Despite superior metrics, visual fidelity remains unsatisfactory in some samples, with identity preservation issues such as altered stitch on the bag (2nd row, Figure 4), distorted bottle label (4th row, Figure 4), and changed shoe patterns (last row, Figure 5), suggesting limitations in retaining fine details during interaction modeling.

2. The restriction to exactly two objects may limit the method's broader applicability. Extensions to more than 2 objects are not explored in the paper, leaving scalability of the proposed method unclear. 

3. While failure cases are shown, their analysis is brief, attributing problems to the shape encoder without detailed remedies or further ablations. Also, is this one of the possible reasons that lead to the poor visual fidelity mention in the weakness 1?

### Questions
1. Given observed fidelity losses in textures and labels (e.g., Figure 4 rows 2/4, Figure 5 last row), how will tuning the CFG scale or refining the shape encoder improve identity preservation without sacrificing harmony? Is there any other solution to improve visual fidelity?

2. How does the proposed Interaction Transformer extend to more than two objects? For instance, would the overlap expert generalize to multi-way intersections, or require modifications such as hierarchical routing?

3. In line 283, the authors mentioned they use the off-the-shelf single-view reconstruction model to render K auxiliary views as augmentation. Which model is used here? How does it impact inference time or robustness to extreme viewpoints?

I would be happy to raise my score if the author rebuttal addresses my concerns.

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
3

### Summary
The authors developed a parallel image compositing framework, PICS, to resolve spatial inconsistencies in pairwise edits by explicitly modeling object interactions using a mask-guided Mixture-of-Experts. The paper is well-organized.

### Strengths
1. The paper is well-written, and the proposed methodology is sound.
2. The approach is well-supported by sufficient experimental results, including both quantitative and qualitative comparisons.
3. The core idea of the Mask-guided Mixture-of-Experts (MoE) is interesting.

### Weaknesses
1. Dependency on Mask Quality: The method's performance relies on precise input masks. The paper lacks a sensitivity analysis on mask quality (e.g., comparing results from high-quality vs. coarse masks or different segmentation sources), making it difficult to assess its robustness to imperfect inputs.
2. Background-based Gating: The gating mechanism for occlusion (Eqs. 6-9)  is primarily based on background-object similarity. This may lead to incorrect judgments when objects share similar textures or colors with the background. Incorporating direct object-object interactions might be necessary for more reliable occlusion reasoning.
3. Limited Physical Consistency: The method occasionally degrades the original object's geometry or texture , as seen in the failure cases (Figure 8).
4. High Computational Cost: Compared to existing methods, the model has a significant parameter count (2.74G – 4.66G), which could hinder practical adoption.
5. Scalability Limitation: The approach is restricted to pairwise compositing, making it less flexible for multi-object scenes.

### Questions
See previous.

### Soundness
3

### Presentation
4

### Contribution
3
