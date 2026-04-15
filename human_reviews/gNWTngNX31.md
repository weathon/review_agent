# TimeWalker: Personalized Neural Space for Life-long Head Avatar

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 6

## Abstract
We present TimeWalker, a novel framework that models realistic, full-scale 3D head avatars of a person on lifelong scale. Unlike current human head avatar pipelines that capture a person's identity only at the momentary level (i.e., instant photography, or short videos), TimeWalker constructs a person's comprehensive identity from unstructured data collection over his/her various life stages, offering a paradigm to achieve full reconstruction and animation of that person at different moments of life.  At the heart of TimeWalker's success is a novel neural parametric model that learns personalized representation with the disentanglement of shape, expression, and appearance across ages. Central to our methodology are the concepts of two aspects: (1) We track back to the principle of modeling a person's identity in an additive combination of his/her average head representation in the canonical space, and moment-specific head attribute representations driven from a set of neural head basis. To learn the set of head basis that could represent the comprehensive head variations of the target person in a compact manner, we propose a Dynamic Neural Basis-Blending Module (Dynamo). It dynamically adjusts the number and blend weights of neural head bases, according to both shared and specific traits of the target person over ages. (2) We introduce Dynamic 2D Gaussian Splatting (DNA-2DGS), an extension of Gaussian splatting representation, to model head motion deformations like facial expressions without losing the realism of rendering and reconstruction of full head. DNA-2DGS includes a set of controllable 2D oriented planar Gaussian disks that utilize the priors from a parametric morphable face model, and move/rotate with the change of expression.  Through extensive experimental evaluations, we show TimeWalker's ability to reconstruct and animate avatars across decoupled dimensions with realistic rendering effects, demonstrating a way to achieve personalized ``time traveling'' in a breeze.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a novel framework to model lifelong 3D animatable avatars. To address this avatar reconstruction problem on a lifelong scale, the author first constructs a large-scale dataset comprising lifelong images from different subjects. Then, they propose a head representation using neural head basis and 2D Gaussian Splatting (GS) for modeling all life stages of one person. Extensive experiments demonstrate the proposed approach can effectively reconstruct lifelong 3D animatable avatars.

### Strengths
1. The paper addresses an intriguing and technically significant task, and the constructed dataset will be valuable for future research in this domain.

2. The proposed method is technically robust, and the extensive experiments provide strong justification for its effectiveness.

### Weaknesses
1. The reconstructed lifelong avatars exhibit significant artifacts beyond those associated with the mouth region, as noted by the authors. However, there is a lack of discussion regarding the origins of these artifacts. For instance, could the limited availability of data during certain life stages be a contributing factor?

2. It would be beneficial to explore the advantages of incorporating multiple lifetimes within the representation for this task. However, the paper lacks a discussion on this point, particularly concerning the extent of performance improvement attributed to the inclusion of additional lifelong data.

3. The novelty of the proposed method appears to be somewhat constrained. The use of blendshape concepts within the Gaussian Splatting framework bears similarities to prior work [1]. Additionally, the adoption of the multi-resolution Hashgrid has also been utilized in INSTA [2], further limiting the originality of the approach.


[1] Ma, Shengjie, et al. "3d gaussian blendshapes for head avatar animation." *ACM SIGGRAPH 2024 Conference Papers*. 2024.

[2] Zielonka, Wojciech, Timo Bolkart, and Justus Thies. "Instant volumetric head avatars." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. Regarding the FLAME parameters of the dataset:  
   1. What method was employed to obtain the FLAME parameters for the constructed dataset?  
   2. How accurate are these FLAME parameters? Additionally, how might the accuracy of the preprocessed FLAME parameters impact the fidelity of the final avatars?

2. Is the representation of lifetime spaces continuous?

3. What is the performance of the method from different viewpoints? The results presented primarily focus on frontal views.

4. The specific lifetime avatars exhibit noticeable shadowing. Given the videos captured at different times, could it be possible to mitigate these shadows to enhance the visual quality of the results?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes to reconstruct a 3D head avatar of a person on a lifelong scale. Specifically, unlike previous avatars that are limited to moment-level reconstructions, this paper achieves full reconstruction and animation of a person across different ages. To this end, they introduce a new head parameter model, in which they design a set of head basis. By linearly combining these bases, they can model a person’s face at different life moments. Additionally, they propose a DNA-2DGS to better integrate the head basis with the deformation from the original FLAME model, resulting in realistic head deformation modeling. They also introduce a dataset where each ID consists of 12K -260K frames across diverse age and head distributions.

### Strengths
1.  They present a new avatar setting that focuses on capturing a comprehensive head avatar over a lifelong scale. To achieve this, they introduce a dataset where each ID contains 12K to 260K frames, spanning diverse age groups and head variations.
2.  This paper proposes a Dynamo module, which learns a set of head basis that could represent the comprehensive head variations of the target person over ages. 
3.  They introduce DNA-2DGS to model head motion deformations based both on head basis and FLAME parameters.
4.  Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Weaknesses
1.  The specific role of this residual embedding is unclear. Is it because the expressiveness of the head basis is insufficient? There seems to be a lack of ablation studies. 
2.  Some details are missing. For example, during inference, how are the blend weights for head basis determined? For instance, when the age is 20, how is the age-to-blend weight mapping obtained? Additionally, is the  I_{res}  that was consistent for all age stages?
3.  The results contain several artifacts, especially during eye-blinking and around the head edges.
4.  In the video results, different life stages exhibit varying lighting conditions. That means several age-irrelevant attributes, such as lighting, were not decoupled from age.

**Similar/identical paper published at CVPR 2024 Workshop POETS 2nd Round**
Thanks for Reviewer f3xJ's remainder.
After reading the paper https://openreview.net/forum?id=3cUdmVfRb4,  I found that this paper is very similar to it, even with identical experimental values (Ablation Study). Based on these findings, I decided to reject this paper.

### Questions
1. Why not use a general basis instead of having a specific head basis for each person?
2. How many head bases are learned for each person?
3. What is the inference speed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes TimeWalker: a scheme for modeling lifelong head portraits. The method learns from a sequence of images and videos of the target object.

** Here is the updated review that ignores the workshop paper and incorporates new content from the rebuttal. **

The rebuttal addressed concerns regarding technical details and comparison baselines, allowing for a better evaluation of the work, but some concerns still remain:
1. The proposed method introduces a novel task, enabling reconstruction on unconstrained noisy data while achieving rendering across time. However, artifacts remain a significant limitation to the method's practical applications. (Supp video 10:40~11:30)
2. While the primary focus of this paper is to establish a neural personalized space spanning various life stages, smooth transitions across stages is an integral component. Abrupt changes will impair the task performance.
3. The method leverages images across lifelong times, one-shot approach can reconstruct avatars by making inferences on these images. This is why I think the task in this paper has some overlap with these approaches: although they do not construct all the avatars in one neural space, the results is somewhat similar (especially the time switching of this method is not smooth, which may be similar to inference on different images).

** The end of the updated part. **

### Strengths
- The article can learn head reconstruction from a set of videos and pictures of the target object at different times, and change between the image and shape at different times after learning is completed.
- This paper includes a dataset of 40 celebrities.

### Weaknesses
- The visual effects of this method have many artifacts, which can usually be avoided when using a large amount of data (15k-260k per person). In addition, the transitions between different time periods are not natural enough, which has a significant impact on achieving the main goal of this method: seamlessly altering their appearance within a predefined set of life stages. For example, in the Personalized Space provided by the project website, the texture of the skin and the overall blurriness are unsatisfactory, and the lifestage switching looks like a quick and abrupt switch between videos of several life periods. The authors should also consider showing more seamless switching results.
- Most importantly, the approach has not been thoroughly compared, making it difficult to assess its contribution. For example, the method lacks discussion of recent Gaussian head methods[1,2], but compared to the baseline methods (insta) of these papers[1,2],
- There is also a lack of comparison with one-shot head reconstruction methods [3,4,5,6], which can also easily reconstruct the target person's head of various ages and shapes, given that in our setting, we have data of the target person at different ages. Although these one-shot methods cannot achieve seamless switching between different shapes and life cycles, this method is also not fully evaluated in this regard (see questions item 2,3).
- While there is a similar paper (https://openreview.net/forum?id=3cUdmVfRb4) published in non-archived form and allowed to be resubmitted in archived form, the current submission reuses a lot of text/visualizations/quantitative results, importantly the methodological and data details are different while the experimental results are the same. For example, the dataset TimeWalker-1.0 proposed in the non-archived paper includes 20 celebrities, but in this submission there is 40. But the ablation study table: table 2 in this submission reused all the numbers in table1 of the non-archived paper.

- Reference:

        [1] Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians
        [2] SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting
        [3] Generalizable One-shot Neural Head Avatar
        [4] Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis
        [5] GPAvatar: Generalizable and Precise Head Avatar from Image(s)
        [6] Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer

### Questions
- How fast is the rendering of this method during inference?
- The method does not evaluate the identity preservation of the person when controlling for lifestage and shape.
- When switching between different life stages, do you consider the order of switching time? Or multiple life stages switched randomly?
- It seems that the Gaussian and 3DMM of this method are bound together (during initialization). Why is it that when it is driven by moment (change the expression), it does not directly use 3DMM to deform the Gaussian points, but introduces a warp field instead?
- In Table 5 (in 1 vs N), FlashAvatar reports lower results than INSTA. The experiments (1vsN) has a similar setting to FlashAvatar paper, but the results is inconsistent with the statement in the FlashAvatar paper that FlashAvatar performs better than INSTA. What could be the possible reason?
- Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
(Updated review; it ignores similarities with workshop paper)

The paper presents TimeWalker, a neural parametric representation that learns personalized representations of subjects with disentangled shape, expressions, and dynamic appearances across time.
The main contributions are twofold: a neural module called Dynamo to encode in a compact manner the main modes of variation in identity across time (a.k.a. a life stage), and another module called DNA-2DGS to encode expression deformations embedded into 3D Gaussians.

While the proposed method utilizes ideas from previous work, such as a canonical shape and appearance space, multi-resolution hash grids, and Gaussian surfels, the proposed modules have some merit and extend beyond previous methods. For instance, an adaptive multiresolution hash grid is proposed to encode neural head bases more efficiently. Also, the Gaussian surfels are adapted to a dynamic setup.

However, the proposed approach exhibits notorious artifacts in different facial regions, and it also shows lighting-identity entanglement due to a simple model that neglects lighting information from the underlying FLAME model. As such, the soundness of the approach is marginal. My final rating is borderline, somewhat inclined to acceptance, if an extended limitations section, additional comparisons, and an extended discussion with related work are added to the paper, and if the societal negative impacts and ethically sourced data are adequately addressed in the paper.

### Strengths
- A new module, called Dynamo, which utilizes a shared canonical shape space to encode identity and a deformed shape and appearance space to capture distinct colors and geometry from different videos captured at various times.
- A new module, called DNA-2DGS, which encodes dynamic expression deformation within a lifelong setting.
- The use of an adaptive multi-resolution hash grid to encode shape more efficiently with better generalization capabilities.
- The paper showcases a prompt-based 3D human head editing, demonstrating the Gaussian-based approach's generalization capabilities and flexibility.

### Weaknesses
- Lighting effects are not disentangled from life stages. In fact, it doesn’t leverage the lighting model of the underlying 3DMM-based model, i.e., FLAME. As a result, the lighting is part of the life stage, which is conceptually wrong.
Noticeable artifacts appear across facial feature boundaries, including eyelids and, more notoriously, the face outline. This could be likely ascribed to a lack of 3D Gaussian scaling regularization, especially across mesh boundaries in the UV space. A better dataset curation, e.g., using images only with good segmentation masks, could have handled this problem, at least for the face outline.
- Mouth interior reconstruction is poor, especially for teeth, which might be solved with more data and better modeling of the mouth interior, e.g., using a separate, more regularized appearance model. A better regularization of the surface normal, e.g., small normal deviations across neighboring surfel locations, could also alleviate the problem.
- Protocols 1 and 2 lack a more comprehensive comparison with other recent related work in 3D human head reconstruction at different ages and shapes. See the Questions section below for more details.
- The proposed dataset might raise ethical concerns as celebrities and actors have not consented to using their images and videos to generate and manipulate sensitive content, such as their faces. Discussions around ethically sourced data, e.g., the origin of the data, licensing, and consent, should be addressed and discussed.
- The paper must further address the potential societal negative impact, especially around Deepfakes. The authors should discuss additional ways to mitigate misuse (see Questions section for more details).

### Questions
*Section 3
- The introduction of the method in Section 3 (first two paragraphs) is redundant as most of it is already mentioned in the introduction. Please remove the first two paragraphs to release some space.
- Please move Section 3.1 (preliminaries) to the supplementary material, as more and more researchers are familiar with the basics of Gaussian Splatting.
The two suggestions above will give extra space to add more comparisons in the main paper.

*Comparisons
- Please move the comparisons with SoTA and mesh comparisons from the supplemental to the main paper.
- Additional comparison for protocols 1 and 2. Please add comparisons to the methods mentioned below.

*References and related work
- Please add the following references
1) Closely related work: Caliskan et al. 2024. PAV: Personalized Head Avatar from Unstructured Video Collection. ECCV 2024
The former paper is a dynamic NeRF-based approach that models an actor’s appearance and shape changes from an unstructured video collection. It utilizes a shared canonical shape space to encode identity and a deformed shape and appearance space to capture distinct colors and geometry from different videos captured at various times. Shape and appearance deformation are encoded through a multi-resolution hash grid and dynamic neural textures, respectively. The approach can naturally be adapted to model lifelong head avatars. Please highlight the differences between the approaches above and provide quantitative and qualitative comparisons, mainly for completeness, as this approach is closely related to the proposed method.

2) Approaches for comparison with Protocol 1 (at least one of the two below):
- Xu et al. 2024. Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians. CVPR 2024.
- Shao et al. 2024. SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting. CVPR 2024.
3) One-shot approaches for comparison with protocol 2 (at least one of the two below). These methods just require a reference image at different life stages to be able to create an avatar at a specific age.
- Ye et al. 2024. Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis. ICLR 2024.
- Chu et al. 2024. GPAvatar: Generalizable and Precise Head Avatar from Images. ICLR 2024.

*Societal negative impact
- Please include an extended section on ethical considerations about the dataset and target applications. This could cover ethically sourced data concerns (data origins, licensing, and consent) as well as potential misuse scenarios, e.g., deepfakes. For the latter, please discuss additional mitigation strategies or guidelines for responsible use of the technology and detection of DeepFakes. Please note that actors have not given consent to the use of their data. So, it is important to make sure the data is crawled with a Creative Commons license.

### Soundness
2

### Presentation
3

### Contribution
2
