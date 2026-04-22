# GASPACHO: Gaussian Splatting for Controllable Humans and Objects

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
We present GASPACHO, a method for generating photorealistic, controllable renderings of human–object interactions from multi-view RGB video. Unlike prior work that reconstructs only the human and treats objects as background, GASPACHO simultaneously recovers animatable templates for both the human and the interacting object as distinct sets of Gaussians, thereby allowing for controllable renderings of novel human object interactions in different poses from novel-camera viewpoints. We introduce a novel formulation that learns object Gaussians on an underlying 2D surface manifold rather than in 3D volume, yielding sharper, fine-grained object details for dynamic object reconstruction. We further propose a contact constraint in Gaussian space that regularizes human–object relations and enables natural, physically plausible animation. Across three benchmarks—BEHAVE, NeuralDome, and DNA-Rendering—GASPACHO achieves high-quality reconstructions under heavy occlusion and supports controllable synthesis of novel human–object interactions. We also demonstrate that our method allows for composition of humans and objects in 3D scenes and for the first time showcase that neural rendering can be used for the controllable generation of photoreal humans interacting with dynamic objects in diverse scenes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work focuses on modeling human-object interactions from multi-view RGB video with Gaussian Splatting. While previous work often under neglect the dynamic movement of object, this work introduce a coarse-to-fine pipeline for reconstructing dynamic objects. The moded humans and objects can be animated to synthesize novel interactions. It also introduce human-object contact constraints in Gaussian splace to ensure proper concats when 3D Gaussian humans are animated to interact with objects.

### Strengths
1. While the baseline methods cannot accurately reconstruct objects, the proposed method demonstrates much better visual performance than baselines.

2. The introduced components are vaild through ablation study.

3. Methods are introduced with mathematical equations.

### Weaknesses
1. The process of obtaining position maps from input images should also be included in Figure 2 for completeness.

2. The application of reconstructing human interact with novel objects seems fancy, but it can be achieved with Gaussian editing or segmention baselines. The advancement of using the proposed method is not mentioned.

3. The introduction of baselines in ablation studies is confusing. Meanwhile the paper uses a lot of bold font without a clear pattern. The format of this paper need to be further polished.

4. Figure 3 seems to be redundant, given the information are included in figure 2. 

Missing related work:
[1] Wang, Xiaoyuan, et al. "HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis." arXiv preprint arXiv:2506.19291 (2025).

Minor weakness: Some texts are overlap with figures. Meanwhile some labels of figure images are introduced in the caption, which could be clearer if put them directly under the images, like Figure 7. Meanwhile the space between text looks weird. Hope this can be adjusted in the revision.

### Questions
1. Is this method able to model multple humans and objects in one scene?

2. Why traditional 3DGS fails under natrually occurring human-object occlusions, but feature-based planes can manage that? Are other baselines also feature-based? If not, the contribution of feature-based representation seems to be more important.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents GASPACHO, a method for generating photorealistic and controllable renderings of human–object interactions from multi-view RGB video. The approach models both the human and the interacting object as distinct sets of Gaussian primitives. It introduces a novel formulation for objects, learning Gaussians on an underlying 2D surface manifold to capture fine-grained details. The method also proposes a contact constraint in Gaussian space to regularize human-object relations and enable physically plausible animations. Experiments on the BEHAVE, NeuralDome, and DNA-Rendering datasets demonstrate high-quality reconstructions under occlusion and the ability to synthesize novel, controllable interactions.

### Strengths
- The paper simultaneously reconstructs humans and interacting objects under occlusion, advancing beyond human-only Gaussian methods.
- Introducing contact constraints in Gaussian space improves physical plausibility and reduces interpenetration during animation.
- The framework enables photorealistic and controllable synthesis of human–object interactions across diverse scenes and viewpoints.

### Weaknesses
- The quantitative results presented in Table 1 and Table 2  are difficult to parse quickly. The best-performing result in each column is not bolded or otherwise highlighted, forcing the reader to manually scan all numbers to identify the state-of-the-art.
- Although the paper introduces a contact constraint in Gaussian space, the contact quality remains suboptimal. Hands and other body parts often penetrate the object surfaces, suggesting that the constraint is weak or insufficiently enforced during animation.
- The method makes a strong assumption that it only models "one dynamic object" and that this object's motion "can be explained using only a rigid transform". This is a significant limitation, as it excludes a vast category of common human-object interactions involving non-rigid or articulated objects, such as interacting with blankets, clothing, ropes, or laptops.

### Questions
- In Figure 2, what does the gray translucent mask represent?
- The contact constraint (Sec 3.5) is defined by mapping a fixed set of SMPL vertices ("feet, hips, hands")  to the nearest Gaussians. How does the method handle or enforce plausible contact for other body parts not in this list?

### Soundness
2

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
This paper proposes GASPACHO, a 3D Gaussian-based neural rendering framework that jointly reconstructs animatable humans and dynamic objects from multi-view RGB sequences, with the explicit goal of enabling controllable human–object interactions under novel human/object poses and viewpoints. The core ideas are: (i) learn pose-dependent Gaussian maps for humans and pose-independent Gaussian maps for objects, anchored to canonical templates; (ii) use a composition- and occlusion-aware loss during training so that occluded human regions are not penalized improperly; and (iii) introduce a Gaussian-space contact refinement that adjusts a sparse set of “contact Gaussians” to promote physically plausible human–object contact at test time. Compared to prior 3DGS avatar work that typically reconstructs humans in isolation, GASPACHO explicitly separates and animates both entities and demonstrates novel cross-sequence retargeting.

### Strengths
1. Addresses controllable human–object interactions instead of human-only avatars, enabling cross-sequence retargeting and scene composition.
2. Introduces pose-independent Gaussian maps for rigid objects that stabilize pose optimization and sharpen textures; principled occlusion-aware losses mitigate erroneous supervision; contact refinement improves plausibility.
3. Demonstrates consistent quantitative gains over strong baselines and compelling qualitative compositions across multiple datasets.

### Weaknesses
1. Restricts to one rigid object, and there is no support for non-rigid objects or multiple objects, limiting applicability to richer HOI scenes.
2. Requires SMPL poses and object masks; sensitivity to pose/mask errors is not thoroughly analyzed. A robustness study would be informative.
3. While the paper explains why some dynamic 3DGS systems are not controllable, additional qualitative side-by-side or an explicit metric for controllability would strengthen the case. Also, report compute/time/memory for training/inference to contextualize practicality.

### Questions
1. What are training and inference times, memory footprints, and Gaussian counts for typical sequences (by dataset)?
2. How does performance degrade with noisy SMPL, imperfect masks, or fewer cameras? Can the method recover without accurate masks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes GASPACHO, a framework that reconstructs animatable humans and objects from multi-view posed videos. It reconstructs humans and objects separately using distinct sets of 3D Gaussians, enabling controllable rendering of novel human–object interactions under different poses and from novel camera viewpoints. For object reconstruction, it employs pose-independent Object Gaussian Maps for efficient reconstruction. Additionally, it introduces a contact constraint in Gaussian space to regularize human–object relationships. Experimental results demonstrate that the proposed method achieves SOTA performance and produces reasonable novel view renderings.

### Strengths
- The proposed work addresses the problem of reconstructing both humans and objects from multi-view posed videos and rendering novel human–object interactions with new poses, which is interesting.

- GASPACHO reconstructs objects using pose-independent object maps, enabling efficient and robust object reconstruction.

- The proposed method achieves comparable performance to previous SOTA methods on human reconstruction and animation tasks.

### Weaknesses
- The writing quality is poor, making the paper difficult to read and understand. Moreover, many technical details are unclear:
    - Numerous mathematical notations are introduced without proper explanation, making it difficult to interpret the method section.
    - In Line 265, the term “ground-truth images” is confusing, as ground truth usually refers to evaluation images; input images would be more appropriate.
    - In Line 199, since the Gaussians are learned later, how are their locations projected onto the planes?
     - In Line 202, it is unclear how the frame with minimal occlusion is determined.
    - The process of obtaining the object template is poorly described; providing pseudocode for this step would improve clarity.
    - The initialization and training procedure of the StyleUNet network for producing Gaussians are not explained.

- The paper includes only one quantitative comparison with previous methods, which is insufficient to validate the results.

- The paper lacks a discussion of the proposed method’s limitations.

### Questions
- The main issue lies in the clarity of the paper’s writing and presentation.

- How do the training and inference times compare with the baselines?

### Soundness
3

### Presentation
1

### Contribution
2
