# Interpretable 3D Neural Object Volumes for Robust Conceptual Reasoning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
With the rise of deep neural networks, especially in safety-critical applications, robustness and interpretability are crucial to ensure their trustworthiness. Recent advances in 3D-aware classifiers that map image features to volumetric representation of objects, rather than relying solely on 2D appearance, have greatly improved robustness on out-of-distribution (OOD) data. Such classifiers have not yet been studied from the perspective of interpretability. Meanwhile, current concept-based XAI methods often neglect OOD robustness. We aim to address both aspects with CAVE - Concept Aware Volumes for Explanations - a new direction that unifies interpretability and robustness in image classification. We design CAVE as a robust and inherently interpretable classifier that learns sparse concepts from 3D object representation. We further propose 3D Consistency (3D-C), a metric to measure spatial consistency of concepts. Unlike existing metrics that rely on human-annotated parts on images, 3D-C leverages ground-truth object meshes as a common surface to project and compare explanations across concept-based methods. CAVE achieves competitive classification performance while discovering consistent and meaningful concepts across images in various OOD settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper is aim at finding a robust and explainable architecture for classification of images based on 3D representation of the object. The authors propose the CAVE method, which is aware of 3D representation of the object, and create an importance measure through it. The idea is to find the best correlation to a set of representations of given classes when generating the importance map. They also proposed LRP based method for faithfull attribution and 3D-C to evaluate stability.

### Strengths
* XAI is obviously one of the most challenging and important issues in nowadays AI era, in almost all fields, especially vision.
* The approach which the authors provide seems novel to me, and the idea to decompose each image to 3D concepts makes a lot of sense to me, and make it reasonable that the faithfulness of  the explanation will be better.
* The link between ellipsoid and Gaussianity, appear in almost all figures is super reasonable, and make things clearer and simpler. Evidence for the link of Gaussians and Ellipsoids, and how it is appear in CLIP you can find here [1].
* Impressive experimental section with convincing examples.


refs:

[1] Levi, Meir Yossef, and Guy Gilboa. "The Double-Ellipsoid Geometry of CLIP". ICML 25.

### Weaknesses
* One main concern is that this method is limited to rigid motion. Think of human body which can be transformed in a non-rigid motion then it is obvious that fixed set of dictionary representation cannot hold all variation of the body. So I find it as a limitation of the method.
* the 3D-C method is heavily relying on given set of meshes, where in the wild it is not given and may be hard to obtain especially for edge cases of objects (especially object with non-rigid movement abilities). I believe that the authors should elaborate on what are the options for a practitioner when this set is absent, or when he is interested in unknown set of classes.

Minor weaknesses:
* in the main figure on the first page, the left part is very intuitive and understandable, the right plot is much less. I didnt understand what the dashed lines stands for, moreover, it is better to put units or some explanation on the axes. I can get that the Y axis is accuracy in between 0-100 but what is 0.1 in spatial localization means?
* The citation of the CRAFT paper by Thomas Fel et. al is in German, replace it with the appropriate English one.

### Questions
* In line 51-85: "Notably, post-hoc methods... However such methods only approximate the model’s computations, and thus do not provide a faithful explanation". Seems to me like a very general and not accurate statement. Why did you claim that they do not provide faithful explanation? for the extreme case - a pure gradient from output to input of the entire architecture should ideally measure the impact of each input element on the output. Seems like very faithful explanation. I think that it should be clarify or backed somehow with experiments or references to other papers showing something like this. 
* It is not clear in the preliminaries section whether the approach limited to embedders which produce feature vector for each pixel, since I think that most embedders squeeze the input information into much smaller dimension, ViT for example patchify the input. If it is the case, then it is a very profound limitation of your approach (or NOVUM you based on). If not - then I think that the notations of H and W are not clear.
* Im not sure how do the concepts being selected in space? why is bus contains the roof and car not (in Fig 4)?
* In many concept decomposition papers the sparsity is playing a vital role, the most basic example for this is the SAE (Sparse AutoEncoders). In several places in your paper you are trying to cinvince that sparsity is a good characteristic where it is almost trivial in concept decomposition.

Overall I find the ideas of the paper very intuitive, so the results seems reasonable to me. The idea to combine 3d-aware representation with XAI is very important, and the modeling through ellipsoids and Gaussians seems reasonable. I think that the paper is good and worth publishing.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The author introduces a framework for interpretable and robust 3D-aware reasoning for image classification. The method is based on learning a sparse, interpretable dictionary of 3D object concepts and replaces dense 3D feature representations with these high level concepts. The approach is designed to enable model-faithful, localized, and consistent explanations for its predictions, even under OOD scenarios such as occlusion and context shifts. They use a recent method of Want et al (ICML 2025) for orientation estimation form 3D models. In addition, the paper proposes 3D Consistency (3D-C) metric that evaluates the spatial coherence of concepts using either ground-truth or estimated 3D meshes instead of human-defined parts. The paper presents an extensive experiment that benchmark their work against both post-hoc and inherently interpretable baselines, demonstrating advantages in robustness - interpretability trade offs, standard interpretability metrics, and classification accuracy on both in-distribution and OOD datasets.

### Strengths
1. Simplicity – instead of selecting dense 3D feature representations, it uses a sparse high level concepts instead. 
2. Effectiveness - can operate both with and without ground-truth pose supervision leveraging Orient-Anything.
3. High accuracy – the authors demonstrates that CAVE achieves high accuracy and OOD robustness with far fewer parameters than dense NOV-based methods

### Weaknesses
1. 3D-C projects concept attributions onto a class CAD mesh. But if you do not have meshes for a class or the mesh is a poor proxy, you cannot score consistently well. This limits evaluation outside these datasets.
2. If the pose estimator struggles on certain data such as symmetric objects, CAVE accuracy and consistency will drop, making it a manageable weakness.
3. The method is limited to single objects scenes.
4. No reported stability across random seeds or concept count for K-Means.
5. No confidence intervals in the charts. Hard to judge the noise level.
6. Seems to be some missing refs and related work, such as: 
a.	Xue, N., Tan, B., Xiao, Y. (2024): NEAT: Distilling 3D Wireframes from Neural Attraction Fields
b.	Liu, X., Zheng, C., Qian, M. (2024): Multi-View Attentive Contextualization for Multi-View 3D Object Detection
c.	Xue, N., Wu, T., Bai, S. (2023): Holistically-Attracted Wireframe Parsing
d.	Tan, B., Xue, N., Wu, T. (2023): NOPE-SAC: Neural One-Plane RANSAC for Sparse-View Planar 3D Reconstruction
e.	Xiao, Y., Xue, N., Wu, T. (2023): Level-S$^2$fM: Structure from Motion on Neural Level Set of Implicit Surfaces
f.	Grainger, R., Paniagua, T., Song, X. (2023): PaCa-ViT: Learning Patch-to-Cluster Attention in Vision Transformers
g.	Kashyap, P., Ravichandiran, P., Wang, L. (2023): Thermal Estimation for 3D-ICs Through Generative Networks
h.	Savadikar, C., Dai, M., Wu, T. (2023): Learning to Grow Artificial Hippocampi in Vision Transformers for Resilient Lifelong Learning

### Questions
1. Can the authors provide statistical variance of the results?
2. Can the authors present class-wise or attribute-wise breakdowns of accuracy/consistency, particularly in OOD or high-occlusion cases?
3. How can practitioners apply CAVE in unconstrained settings where 3D geometry is sparse or diverse?
4. Can the authors provide an accuracy and 3D Consistency as a function of pose error?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces CAVE, a robust, interpretable image classifier based on 3-D object representations. To visualize pixel importance, the authors also extend LRP for the proposed model. Finally, a new metric called 3D consistency is introduced to measure whether learned concepts consistently match with the same part of the underlying 3D volume. Experiments show that CAVE produces better localization, coverage, and consistency than prior methods while achieving strong accuracy. Notably, in the absence of 3D annotations CAVE outperforms all other methods.

### Strengths
- The problem targeted by this paper is interesting and well motivated, while remaining reasonable in scope.
- The figures throughout the work are clear and well made.
- CAVE is, to my knowledge, the first model to combine neural object volumes with concept based explanations, providing a source of novelty.
- The numerical results achieved by CAVE are strong compared to other robust only or interpretable only methods.

### Weaknesses
- It is unclear to me why CAVE can be considered inherently interpretable/faithful. Each classification is a function of the dot product between each image patch and some concept; this dot product is not bounded, and does not reflect a meaningful distance/similarity measure. Even if it did, these concepts do not seem to have any semantic meaning -- they are freely learned parameters. As such, it may be inappropriate to compare with only interpretable/explainable models. If I am missing something fundamental, clarification would be appreciated. This is my primary concern.
- The clarity of this work suffers a bit from not being self-contained. In particular, it is unclear from this work how spatial location and distributional nature of NOV's actually matter. If a reader does not read NOVUM, each NOV simply seems like a free parameter vector in the embedding space of the backbone. Moreover, there are a couple minor issues with formalism that impact clarity:
    - Line 314: $\sum_{(i,j)} A^+(x, h) = 1$ <- i, j do not appear as indices inside the sum
    - The function $\Omega_y$ is never defined, including in the appendix

### Questions
- How can it be said that standalone NOVUM is not faithful, but CAVE is? The only difference between the two as predictive models seems to be the number of concepts.
- In the appendix, Spatial localisation is defined with respect to a single image-concept pair. How is this aggregated to a single value per model/dataset?
- It may be worth comparing to [1]; this is an extension of ProtoPNet that, while not targeting robustness, learns distributional concepts in 2D space.

[1] Wang, Chong, et al. "Mixture of gaussian-distributed prototypes with generative modelling for interpretable and trustworthy image recognition." IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

### Soundness
2

### Presentation
3

### Contribution
2
