## Human Reviewer 1

### Summary
This paper proposes a new 3D generation method that introduces a meta-ControlNet and dynamic reference routing, leveraging user-provided 3D reference models and images to produce high-quality, controllable 3D object synthesis.

### Strengths
- The introduction of meta-ControlNet, combined with 3D reference models, provides strong guidance for the generation process, resulting in high-quality 3D outputs with excellent geometric consistency.

- As a reviewer familiar with this field, I believe that the combination of 3D reference models and images for generating 3D results has significant industrial value and could greatly inform future work in 4D generation as well.

- The method is simple, easy to understand, and the paper is clearly written.

### Weaknesses
- Obtaining a reference 3D model can be quite challenging. Although this is valuable in certain application scenarios, it can be difficult to find appropriate reference 3D models in extreme cases. I recommend that the authors add this limitation to the paper.

- While the strategy of combining ControlNet for generation appears straightforward, this could be criticized for lacking novelty. However, in my opinion, simplicity often leads to elegant and effective solutions, and I don’t consider this a reason for rejection.

### Questions
- I strongly support the task of combining reference 3D models with images for 3D generation as proposed in the paper, especially considering its relevance to the gaming industry and asset creation. Some related 3D generation methods discuss a texture editing task using complex images to guide coarse 3D object generation, which is highly relevant to this work. Could the authors provide some discussion or analysis on how to improve the method when there are significant differences between the reference image and the reference 3D model? This could be addressed in future work.

- I am inclined to accept this paper, as I also work on related tasks. Currently, I rate it as 6 but would likely increase the rating to 8 during the rebuttal phase.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces Phidias, a RAG-based framework for high-fidelity 3D content creation from multimodal inputs, including text, images, and 3D references. The framework introduces key innovations include Meta-ControlNet to dynamically modulate reference influence, dynamic reference routing to progressively refine outputs through a coarse-to-fine strategy, and self-reference augmentation to enhance generalization. Phidias enables sparse-view reconstruction, theme-based model variation, and user-interactive generation, demonstrating clear superiority over state-of-the-art approaches in both geometric precision and visual fidelity.

### Strengths
1. This paper presents a well-founded solution for multi-view generation, effectively addressing challenges such as the generation of unseen views and other ill-posed problems.
2. This paper is well-organized and clearly written, making it accessible to readers.

### Weaknesses
1. A significant weakness is the limited analysis of the retrieval mechanism for 3D references, which plays a critical role in guiding generation. The reliance on existing 3D models introduces a potential bottleneck, as Phidias’s performance may degrade if the retrieved references are irrelevant or of poor quality. 
2. The control mechanism introduced via Meta-ControlNet is promising, but the paper does not delve deeply into more challenging settings. For example, scenarios where the input concept conflicts with the reference.

### Questions
1. why the point cloud for retrieval only contains 10K points, is it limitation stems from constraints imposed by Uni3D?
2. Why are the performences in Figure 14 so different from the results of the last row of figures in Figure 4? It looks like that some case's pose in Figure 4 doesn't match the input image, either.
3. The time and resource consumption for retrieval during each inference is not clearly stated

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces a generative model designed for creating 3D content from text, image, and 3D conditions with reference-augmented diffusion. Phidias aims to enhance the quality, generalization, and controllability of 3D generation by leveraging existing 3D models as references. The model integrates three key components: meta-ControlNet for dynamically adjusting conditioning strength, dynamic reference routing to address misalignments between input images and 3D references, and self-reference augmentations for self-supervised training with a progressive curriculum. 

These components collectively improve generative performance over existing methods, offering a unified framework for 3D generation using diverse inputs. Phidias enables various applications, including retrieval-augmented image-to-3D, text-to-3D, theme-aware 3D-to-3D generation, interactive 3D generation with coarse guidance, and high-fidelity 3D completion.

### Strengths
1. This paper proposes a dynamic reference routing approach. Recognizing that reference models do not perfectly correspond to the conceptual image, this paper utilize a feature pyramid to address this discrepancy. The motivation is that, it has been widely observed during the reverse diffusion process that the coarse structure of a target image is established in the early, high-noise timesteps, while finer details emerge progressively as the timesteps advance. This motivates to start with low-resolution reference CCMs at high noise levels.
2. This paper facilitates a range of downstream applications, including retrieval-augmented image-to-3D, text-to-3D, theme-aware 3D-to-3D generation, interactive 3D generation with coarse guidance, and high-fidelity 3D completion.

### Weaknesses
1. Dependence on Retrieval Quality: As stated by the author The performance of Phidias is contingent upon the quality and relevance of the retrieved 3D reference models, which may not always be optimal.
2. Database Limitations: The current 3D database used for retrieval is limited in size, which could restrict the diversity and accuracy of reference models available for augmentation. Moreover, the model necessitates an additional dataset for inference, which may further restrict its applicability.

### Questions
1. In line 186, there is a missing citation regarding the canonical coordinate map. I believe this map is also referred to as the NOCS map, as introduced in reference [a].

[a] Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper presents a new 3D generation approach that utilizes a reference 3D model to enhance the generation process. The method comprises three key components: (1) Meta-ControlNet (2) Dynamic Reference Routing (3) Self-Reference Augmentation. Experimental results demonstrate the effectiveness of this method in enhancing 3D generation quality, generalization ability, and controllability.

### Strengths
- Gathering information from a 3D reference model is straightforward.
- The evaluation of generation quality shows the effectiveness of the approach.
- The paper is well-written and easy to follow.

### Weaknesses
- In the evaluation, only generation quality is assessed, with no evaluation of generation speed. For Phidias, retrieving information from the reference 3D model requires both 3D model retrieval and rendering of the canonical coordinate maps, which introduces additional latency. However, the paper does not report specific latency overheads.
- From Table 3, the effectiveness of dynamic routing appears to be limited.

### Questions
- With the dynamic reference routing mechanism, the meta-ControlNet should be capable of handling inputs at various resolutions. Is the meta-ControlNet trained on images of different resolutions, or is it trained solely on the highest resolution and expected to generalize to lower resolutions?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 5

### Summary
The paper present a reference-augmented method using retrieved 3D objects as auxiliary guidance to reduce the ill-posed nature of the single-view 3D reconstruction task. The method leverages multi-view diffusion zero123++ for the generation of 6 novel view images, and scale-up the LGM for 3D reconstruction. The main contribution of the paper is to use input concept image (typically lack of 3D information of the back or side view, etc.) to refer a similar 3D objects from their 3D database (40K objects selected from Objaverse), the 6 canonical coordinate maps (CCM) from referred 3D object is used to generate better multi-view images. Meta-ControlNet with dynamic resolution of the intake CCM during diffusion sampling contribute to the misalignment dilemma between the concept images and referred 3D object. The results show that the method outperform other SOTA models in image-to-3D tasks given the proper 3D object reference. And it may extend to text-to-3D, and 3D-to-3D applications.

### Strengths
1. The framework extends the controllability of current image-to-3D models. The model retrieves an auxiliary 3D object as guidance to improve the 3D reconstruction quality. The results shows outperformance than current SOTA models.

2. The paper propose meta-controlnet, and self-reference inference to let model learn how to resolve the conflict between retrieved objects and concept front-view image. Given the setting by the paper claimed, the model is able to provide reasonable 3D generation results matches on both sides.

3. The model is able to extend to text-to-3D and 3D-to-3D generation tasks, which may contribute to more broad applications.

### Weaknesses
1. (model behavior) The model is trained with self-augmented ground-truth 3D objects, in order to handle the misalignment dilemma. However, as the A.4 specified, the augmented ones are still very similar to the ground truth. Moreover, all showcases are with no explicit conflicts between concept image and 3D objects. This raises the question of how a bad retrieval influence the model performance, which is actually common in practical case (as line 878-879). More comprehensive analysis or comparisons are necessary.

2. (retrieval robustness) The quality of the retrieval directly influence the improving of the generation quality, especially on the back or side views. I elaborate in detail in Question-1. If the user have either a out-of-retrieval-database front-view concept image, or the retrieval brings an unreasonable objects, the model output may not be as good as claimed. As Fig.7 shows, random reference as an example of non-similar retrieval may provide unreasonable results. This restricts the usage of the method.

3. (restricted/unfair experiment setting) Followed by weakness-1, the paper constrains the input image as only-front-view (line 221-222) to all competitor models. A random view, or 2-4 views as other related works defined, will contain more geometric information of the 3D details of the object. If not restricting to only-front-view input, using other models which do not require an auxiliary 3D object input like OpenLRM may already provide reasonable results. Their results may be better than the results with no reference in Fig.1, and Fig.7, and SOTA results in Fig. 5.

4. (restricted/unfair experiment setting) In the model evaluation section, all showcase examples meet the optimal situation of the framework. (a) The retrieved 3D objects is very similar to the concept image in the front view, (b) there is no/little conflicts on the other views. Given that, the improvement of the quantitative results in Tab.1 is still marginal. The row "Ours (GT Ref.)" may not be a reasonable comparison, since it provides ground truth geometric information in 6-view. I am not surprise it provides better generation results from the unseen view.

5. (model usage) When there is considerable conflict between retrieved 3D objects and concept images (as line 878-879, it is actually very common), which side would the model follow the most? Potential users may want to see more examples of practical cases. And from a point of view as graphics designer, it will be better to provide a controlling nob for user to decide the extent of guidance from either side, rather let the model itself decides the extent. Right now, it is still uncontrollable.

6. (novelty) The novelty of the paper concentrated on the meta-controlnet and reference-based inference. However, given the weakness aforementioned, it may solve an restricted task or with limited contribution to the real problem, and compare to SOTAs in an unfair setting. Moreover, the reason to use diffusion models, and following reconstruction model may need more elaborations.

### Questions
Great work! I elaborate more based on weakness, and provide some suggestions and questions.

1. What is the main research goal of the paper? Is it to improve the quality of the 3D reconstruction on un-seen views (which indicates the output model should faithfully obey the input concept image), or it provides more customization freedom by given auxiliary 3D object reference (which indicates the output should lean to the given 3D objects.). Based on a full reviewing of the paper, my understand is the paper lean to the former one (correct me if I am wrong). So when there is conflict between retrieved 3D objects and concept images, which side would the model follow the most? If I am a graphics designer, a useful one will be a model doing either way, rather than a model-decided interpolation between two sides.
 
2. Is it a correct understand that the meta-controlnet is actually to learn a 3D texture mapping, which utilize the pixel-based texture in the concept image, map it to the retrieved 3D object surface based on 6-view CCMs? In that, the best way to phrase the contribution may like "using the front-view image as 2D texture guidance to generate unseen views given the 6-view CCM priors".

3. Follow Question-2, would you elaborate more about the role and reason to use diffusion model here? My understanding is, because there is conflict if we directly use 3D texture mapping using concept images and 6-view CCMs, so you let the diffusion process handle those conflicts. Any reason to exclude other pixel-based networks or traditional CG methods?

4. There are a lot of necessary details in the supplementary, like details of retrieving, dataset, etc. It will be better to move them into the main paper, even some of details may raise questions from reviewers. There is still a half-page blank now.

 I reserve the change of grades upon authors' responses to my reviewing.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4