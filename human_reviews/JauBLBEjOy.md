# AvatarStudio: High-fidelity and Animatable 3D Avatar Creation from Text

- Decision: Reject
- Scores: 5, 8, 5

## Abstract
We study the problem of creating high-fidelity and animatable 3D avatars from only textual descriptions. Existing text-to-avatar methods are either limited to static avatars which cannot be animated or struggle to generate animatable avatars with promising quality and precise pose control. To address these limitations, we propose AvatarStudio, a coarse-to-fine generative model that generates explicit textured 3D meshes for animatable human avatars. Specifically, AvatarStudio begins with a low-resolution NeRF-based representation for coarse generation, followed by incorporating SMPL-guided articulation into the explicit mesh representation to support avatar animation and high-resolution rendering. To ensure view consistency and pose controllability of the resulting avatars, we  introduce a 2D diffusion model conditioned on DensePose for Score Distillation Sampling supervision. By effectively leveraging the synergy between the articulated mesh representation and the DensePose-conditional diffusion model, AvatarStudio can create high-quality avatars from text that are ready for animation, significantly outperforming previous methods.  Moreover, it is competent for many applications, e.g., multimodal avatar animations and style-guided avatar creation. Our project page is https://avatarstudio3d.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a pipeline for creating 3D avatar from text prompts. In contrast to the existing work, this method can animate the avatar using the underlying SMPL structure. This is a coarse to fine pipeline where in the coarse stage a residual prediction scheme over SMPL using NERF is proposed which is used to initiaise in the fine stage. In the fine stage DMTet (deep Marching tetrahedra) is used create a high resolution avatar. Further there is a part aware superresolution. For text conditioning, the method uses score distillation sampling with a control factor from dense poses.

### Strengths
The paper tries to solve an important problem of current times. The strategy seems to be working and not counter intuitive. The results are also very encouraging.

### Weaknesses
The paper stitches together the existing methods and produces an intuitive pipeline used for other 3D asset creation from texts. Nerf with SMPL followed by Score distillation sampling seems to be very intuitive. Hence the novelty is a concern. The text prompts are also simple. The time taking to create the avatar is 2.5 hours which is too much of time. While the ablation shows favorable results but I am not sure why the coarse and fine stages are separately required?

### Questions
There are few important questions apart from the weaknesses which comes to my mind
1. The skinning is borrowed from SMPL which is coarse. The nearest vertices based weight can create a deformation problem.How that is not impacting?
2. What if we only deal with offset of SMPL instead of fine stage? Geometrically Fig 6 (a) is not showing significant advantage.
3. Its not clear why densepose is important than skeleton? Is it because skeleton is noisy? it is not clear from Fig 6(b) why the control net is behaving well with densepose. The argument is not strong. Cant there be other cases where densepose is failing?
4. We need more example and grounding for the need of dual space training.
5. How "p" is used in equation 4?
6.What is the comparison between CFG and CFG rescale trick?
7. Why Avatarverse has produced red belt instead of black belt in second row of Fig 3? has the comparison being done with proper negative prompt too?
8.It is not clear how the method is handing the janus problem.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of text-guided 3D full-body human generation, and proposes several improvements over the previous SOTA method. Experiments demonstrate that the proposed method indeed improves the generation and animation quality, and all components are well-motivated in the ablation study.

### Strengths
- The paper is well-written and easy to follow
- The proposed method achieves SOTA results in text-to-3D avatar generation.
- The paper introduces several well-motivated techniques to improve the generation and animation quality, including using deep marching tetrahedra, densepose-guided ControlNet, part-based super-resolution, and SDS optimization in both canonical and deformed space. While some of these techniques have been used in other related tasks, demonstrating their effectiveness in this specific domain is a valuable contribution.

### Weaknesses
While one main focus of the paper is to improve animation, the animation still lacks realism. The animation is modeled via pure LBS with SMPL skinning weights and topology, thus cannot generate realistic non-linear cloth deformation, and cannot deal with loose clothing with other topologies such as skirts (skirts are split as shown in the animation results on the webpage).

### Questions
This paper proposes a set of simple yet effective techniques to improve text-to-avatar generation. The resulting method has demonstrated great quality improvement over previous SOTA. While the animation quality still needs further improvement, I believe this paper in its current state is already a valuable contribution to the field. I don't have any specific questions at this point.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a new method for generating 3D human avatars from text prompts. To this end, the authors propose to generate 3d avatars in a coarse-to-fine manner. The coarse stage is based on NeRF, while the fine stage takes the coarse result as initialization and refines it with explicit mesh-based representation. In addition, the authors propose to perform SDS sampling with a diffusion model conditioned on DensePose, which allows for better view consitency. Experiments show that the proposed method is able to generate animatable human avatars from only text input.

### Strengths
* The proposed method is able to generate high-quality human avatars from only text input, and the generated avatars have clear appearance details. Experiments show that the proposed method outperforms existing pipelines. Moreover, the authors also demonstrate stylized avatar creation given a style image as an additional condition, which is very impressive. 

* The authors propose to using DensePose-conditioned ControlNet for SDS supervision. Experiments show that it can achieves precise and stable pose control, which may inspire future work on avatar generation or other catogery-specific 3D generation tasks. 

* The paper is well-writen and easy to follow.

### Weaknesses
* In Abstract and Introduction, the authors claim that using ControlNet conditioned on DensePose offers a benefit on view consitency, but I cannot find any experiments to support this claim. In Figure 6(b), the authors conduct an ablation study to evaluate the effects of different SDS supervision, but the results only show that leveraging skeleton-based ControlNet may suffer from leg pose error. Existing methods like DreamHuman and DreamAvatar are typically based on original Stable Diffusion or skeleton-based ControlNet, and I didn't notice any view inconsistency issues. 

* Generating 3D models in a coarse-to-fine manner with two representations is not a new idea. In fact, it has already been proposed in ProlificDreamer [Wang et al, 2023], which also leverage NeRF for initialization and uses DMTet for further refinement. Therefore, I don't think it can be regarded as a technical contribution. 

* Jointly optimizing the textured avatar mesh in both deformed and canonical spaces is also not new and has been proposed in DreamAvatar, which also proposes to perform SDS supervision in both canonical space and posed space.  

* Overall, although this paper demonstrates good results, its technical novelty is not strong enough to me. I feel that the proposed method is more like a combination of existing techniques and tricks.

### Questions
See [Weaknesses]

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
