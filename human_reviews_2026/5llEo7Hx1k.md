# StreetDiffusion: Street Scenes Generation via Multi-view Stable Diffusion with Structure Prompts

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Multi-view Stable Diffusion has been proposed and applied for indoor or wild scene generation. However, the generation of outdoor scenes, especially urban street scenes, has not yet been well studied, which shall be more complicated than existing indoor or wild scene generation due to containing more objects and structures. In this work, we focus on the generation of street scenes relying on a multi-view stable diffusion model with structure prompts, such as segmentation maps. Thus, we propose StreetDiffusion, which employs a dual-branch architecture to integrate panoramic and local information where structural priors are inserted into two branches to generate highly consistent and realistic multi-view street scene images. To study the street scene generation issue, we also propose a large multi-view street scene dataset, Street 360, which consists of 10K panoramic images from urban streets. Experiments demonstrate that the proposed StreetDiffusion model generates high-quality street scenes, with a clear advantage in the street scene generation task over existing multi-view generation models designed for indoor or wild scenes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents StreetDiffusion, a diffusion-based framework for generating street scenes conditioned on various structural prompts. The authors introduce a PAM to ensure multiview and global consistency. In addition, they propose a new dataset, Street360, which includes panoramic street images and multimodal annotations. However, the proposed task appears to have limited significance, and the overall performance of the method is underwhelming.

### Strengths
1. The proposed Street360 dataset could potentially enrich data diversity for street scene generation and related multimodal research.
2. The proposed PAM module is a reasonably good design, integrating global (panoramic) and local (multi-view) cues through attention mechanisms to maintain consistency.

### Weaknesses
**1. Poor qualitative results**
* In Figure 1, there is no visible geometric or structural correspondence between the structure prompts(left) and the generated results(right).
* In Figure 3, the given prompts (e.g., “a street with a bike parked in front of a building”, “a street with a green line painted on it”) are not reflected in the generated images. The adherence to prompts is significantly weaker compared to baselines such as MVD and PanFusion.

**2. The task itself seems ill-motivated**
* The proposed problem of generating images from texture or structure prompts can already be effectively addressed by existing methods such as ControlNet.
* The authors seem to have transformed an otherwise straightforward texture-conditioned generation task into a more complex panoramic stitching problem, yet the motivation behind this design choice is not clearly articulated.
* To justify the necessity of this new task, the authors should demonstrate that a baseline like ControlNet (conditioned on panoramic structure maps and text) fails to perform well, thereby motivating the proposed formulation.

**3. Overstated contribution in dataset design**

The claimed contribution in constructing a “street scene” dataset is overstated, as existing datasets such as KITTI and VIGOR already include rich street-level scenes that can serve similar purposes.

**4. Limited performance and generalization**
* The proposed method does not achieve competitive results on the Matterport3D benchmark, which raises concerns about generalization.
* The authors are encouraged to include qualitative results on Matterport3D to demonstrate the general applicability and robustness of their approach.

**5. Ambiguous design choice**

Why are panoramic images divided into eight perspective views rather than the conventional six cube-map faces? The motivation and advantage of this specific decomposition should be clarified.

### Questions
The problem corresponds one-to-one with the weakness:
1. How do the authors explain the poor structural and semantic alignment between prompts and generated results? Why does the model fail to capture explicit objects described in the prompt?
2. What is the motivation for defining this task when similar outcomes can be achieved using ControlNet? Can the authors show that existing diffusion-based methods cannot solve this problem effectively?
3. How does Street360 differ fundamentally from existing datasets like KITTI or VIGOR, beyond including panoramic data?
4. Why does the method not outperform existing approaches on Matterport3D? Could the authors provide visualizations on Matterport3D to verify generalization?
5. What is the reasoning behind splitting panoramic images into eight perspective views instead of six? Does this improve consistency or model efficiency?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of generating high-quality, multi-view consistent urban street scenes, a domain where existing text-to-multi-view methods struggle with complex layouts and dynamic elements. The authors identify two key challenges: a lack of complex urban datasets and the tendency of current models to produce structural artifacts.

To address this, they make two core contributions:

1. Street360 Dataset: A large-scale dataset of over 10K real-world street panoramas, complete with text prompts and structural annotations (e.g., segmentation maps).

2. StreetDiffusion Model: A novel framework that synergizes a panoramic branch and a perspective branch. Its key innovation is the Panorama Alignment Module (PAM), which uses structural prompts and geometry-aware attention to enforce global consistency and local detail.

Experiments show the proposed method outperforms existing models in both quality and multi-view consistency for street scene generation.

### Strengths
+ A new framework: The "Panorama–Perspective Synergy Framework," which integrates a global panoramic branch and a local perspective branch, guided by structural prompts (e.g., segmentation maps, sketches), to enhance consistency and realism.

+ A new dataset: Street 360, a large-scale dataset of 10K multi-view and panorama images to support research in this specific domain.

+ Superior performance: Experimental results show that StreetDiffusion outperforms existing multi-view diffusion models on the task of street scene generation.

### Weaknesses
1. The author argues that previous works can only generate images with limited resolution (1024x512) due to their architecture design, while this paper is able to generate high-resolution images (2048x512). 
However, I don't think that this is a resolution of previous works. The resolution can be easily handled by a downstream super-resolution network. Furthermore, a typical panorama image holds a 180-degree vertical FoV and 360-degree horizontal FoV. This is the motivation why the generated panoramic images holds an aspect ratio of 2:1 instead of 4:1. 

2. The authors claim that this is the first work on street scene generation full of various object structures. This might not be true. The following works also address street-view scene generation:

a. Yan, Yunzhi, et al. "Streetcrafter: Street view synthesis with controllable video diffusion models."CVPR 2025.
b. Ze, Xianghui, et al. "Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control." ICLR 2025.
c. Lin, Tao Jun, et al. "Geometry-guided cross-view diffusion for one-to-many cross-view image synthesis." 3DV 2025.

3. The proposed method is of limited novelty. As also stated by the authors, the dual-branch architecture is borrowed from PanFusion, and the CAA is borrowed from MVDiffusion. The differences are that this paper introduces panoramic and multi-view structural prompts, such as segmentation maps and contour maps, and employs a panoramic branch to assist the multi-view branch explicitly. However, the former is just an introduction to additional inputs. The latter is also used in other dual-branch works, using one branch to guide the other branch. For example, PanFusion leverages the multi-view branch to guide the panoramic branch. 

4. I do not really get the key differences between this work and PanFusion. From my understanding, the differences include:

a. This paper employs additional structure conditions, such as segmentation maps and contour maps, which are not used in PanFusion. However, I don't think this is a key difference. The structure conditions can also be easily applied to PanFusion by ControlNet or other related methods. This should not be a key difference or significant contribution of this paper. 

b. This paper leverages the panoramic branch to assist the multi-view branch, while in PanFusion, the two branches exchange information with each other. PanFusion uses the output from the panoramic branch, while this paper stitches the outputs from the multi-view branch. I understand there are different intuitions behind them. However, I still don't think this is a significant contribution.

### Questions
Plz refer to my comments in "Weakness". 

Some detailed comments:

1. There are formatting issues in Lines 179-180. 
2. In Lines 195-196, the authors state that PanFusion generates panoramas and uses them to supervise multi-view results. However, if my understanding is not wrong, the PanFusion is bijective. The multi-view branch also assists the panoramic branch.

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
5

### Summary
This paper presents StreetDiffusion, a panorama–perspective synergy framework for generating multi-view urban street scenes from text and structural prompts. The method pairs a panoramic branch with several perspective branches and connects them through a Panorama Alignment Module that maps local views onto the sphere to enforce cross-view geometric consistency while incorporating structure priors such as semantic segmentation, contours, or sketches via ControlNet-style adapters and attention. A global-to-local prompt design supplies scene-level context and view-specific detail, and training proceeds in stages to first stabilize each branch and then optimize joint alignment. The authors introduce the Street360 dataset with panoramic images and corresponding multi-view crops, and benchmark against strong baselines including PanFusion and MVDiffusion using FID, IS, CLIP-Score, overlapping-region PSNR, and a user study. Results show higher fidelity and stronger cross-view coherence, particularly when structure conditions are provided, advancing controllable street-scene generation.

### Strengths
1. **Elegant and effective framework design**: Structural cues such as segmentation, outline, and sketch are uniformly injected, while panoramic branches and multi-perspective branches are collaboratively modeled, providing each other with context and constraints, and can simultaneously take into account global layout and single-view details.
2. **Robust Structural Control**: is a key advantage of StreetDiffusion, which generates reasonable street scene skeletons even with weakly constrained inputs such as rough sketches or incomplete outlines.
3. **Good writing**: This article has a clear structure and logical flow, making it easy to understand the basic principles.
4. **Extensive and comprehensive experiments**: numerous experiments have demonstrated that StreetDiffusion can effectively generate high-quality and realistic street view images based on various structural cues, and the ablation experiments provide sufficient evidence.

### Weaknesses
1. **Insufficient experimental description**: In the comparative experiments, the text prompt input for the comparative methods needs clarification.
2. **The credibility of the user evaluation results is insufficient**: In the User Study Results, the authors only counted 21 valid questionnaires derived from 20 generated images. Is this representative?
3. **Poor intra-image consistency**: Qualitative results (Fig 3) show that although the street view images generated by StreetDiffusion are more realistic and accurate in structure and texture, there seems to be a significant inconsistency in texture style under different viewpoints (for example, different buildings exhibit different texture styles at every 90° angle). The authors need to provide further explanation.

### Questions
1. In the comparative experiments, the text input for different methods is ambiguous. StreetDiffusion utilizes GPT4 to describe multi-view images and generate local text when generating text prompts. However, does the input in the baseline also use global + local text as input? Will different text input lengths lead to unfair comparisons?
2. Intra-image consistency is crucial for 360° street view panoramic images, especially the consistency between the two sides of the image. For example, if the generated result is rotated 180° along the camera viewpoint, does the resulting image have obvious inconsistencies (e.g., obvious stitching marks)? I would love to see the authors discuss the above issues.

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
This paper introduces StreetDiffusion, a novel multi-view stable diffusion model designed to generate realistic and consistent urban street scenes, a task more complex than indoor or wild scene generation. To support this, the authors created Street360, a new large-scale dataset of 10,000 multi-view panoramic street images with corresponding structural annotations. The StreetDiffusion model employs a Panorama-Perspective Synergy Framework to integrate global panoramic information with local perspective views, using structure prompts (like segmentation maps, contour maps, or user sketches) to guide the generation process and ensure structural accuracy. A Panorama Alignment Module (PAM) is also introduced to enforce geometric consistency between the panoramic and perspective representations. Experiments show that StreetDiffusion outperforms existing multi-view generation methods, producing higher-quality and more consistent street scene images that adhere to the provided structural cues.

### Strengths
The paper introduces Street360, the first large-scale, high-resolution dataset specifically designed for the complex and under-studied task of multi-view urban street scene generation, complete with valuable structural annotations. 

It proposes a novel model, StreetDiffusion, which uses a Panorama-Perspective Synergy Framework to uniquely integrate global panoramic information with local multi-view synthesis, guided by structural prompts like segmentation maps or sketches for enhanced control and realism. 

The inclusion of a Panorama Alignment Module (PAM) ensures geometric consistency between views, allowing the model to demonstrably outperform existing methods in generating high-quality, realistic, and structurally coherent street panoramas.

### Weaknesses
1. The model's optimal performance relies heavily on the availability and quality of detailed structure prompts (like segmentation or contour maps). While the paper demonstrates results using only text and user sketches, the quality and consistency might be reduced without this strong structural guidance.

2.The model employs a complex three-stage training strategy. This multi-stage approach can be more cumbersome and computationally expensive than a single, end-to-end training process.

3.The panoramic module synthesizes at $2048 \times 512$ resolution, while the multi-view branch trains on $512 \times 512$ images. This means the final stitched panorama (approximately $2048 \times 512$) is a significantly lower resolution than the 4K to 8K images available in their own Street360 dataset, indicating a gap in the model's ability to generate truly high-resolution content.

4.The method is, to a large extent, a clever combination of existing technologies. It borrows the dual-branch architecture from PanFusion and the CAA from MVDiffusion , and it integrates standard components like ControlNet and LORA.

5.The paper includes an extra experiment on the Matterport3D indoor dataset (Table 7). The results show that while StreetDiffusion achieves the best FID score, it performs worse than the baseline MVDiffusion on the IS and CS metrics. This suggests that the model's architecture, while effective for street scenes, is not universally applicable and may be less effective for indoor environments.

### Questions
Given the authors' insightful future work direction on exploring BEV representations, it might be valuable to also discuss or reference recent related works, such as CrossViewDiff and SkyDiffusion, as these could serve as relevant points of comparison for that line of inquiry.

### Soundness
3

### Presentation
3

### Contribution
3
