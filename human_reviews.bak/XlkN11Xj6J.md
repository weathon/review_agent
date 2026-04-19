# Generating Images with 3D Annotations Using Diffusion Models

- Decision: Accept (spotlight)
- Scores: 5, 5, 6, 8

## Abstract
Diffusion models have emerged as a powerful generative method, capable of producing stunning photo-realistic images from natural language descriptions. However, these models lack explicit control over the 3D structure in the generated images. Consequently, this hinders our ability to obtain detailed 3D annotations for the generated images or to craft instances with specific poses and distances. In this paper, we propose 3D Diffusion Style Transfer (3D-DST), which incorporates 3D geometry control into diffusion models. Our method exploits ControlNet, which extends diffusion models by using visual prompts in addition to text prompts. We generate images of the 3D objects taken from 3D shape repositories~(e.g., ShapeNet and Objaverse), render them from a variety of poses and viewing directions, compute the edge maps of the rendered images, and use these edge maps as visual prompts to generate realistic images. With explicit 3D geometry control, we can easily change the 3D structures of the objects in the generated images and obtain ground-truth 3D annotations automatically. This allows us to improve a wide range of vision tasks, e.g., classification and 3D pose estimation, in both in-distribution (ID) and out-of-distribution (OOD) settings. We demonstrate the effectiveness of our method through extensive experiments on ImageNet-100/200, ImageNet-R, PASCAL3D+, ObjectNet3D, and OOD-CV. The results show that our method significantly outperforms existing methods, e.g., 3.8 percentage points on ImageNet-100 using DeiT-B. Our code is available at <https://ccvl.jhu.edu/3D-DST/>

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigate how to use generated images from diffusion models to improve discrimitive tasks in both in-distribution and out-of-distribution settings. The authors propose to render 3D data to 2D edge maps, fine-tune the large-scale diffusion model via ControlNet approach with the prompt augmented form LLM. After training, the generated images naturally have all the 3D object annotations. These generated data can be subsequently used as data augmentation for downstream tasks. The paper demonstrate the effectiveness of the proposed method on image classification and 3D pose estimation tasks.

### Strengths
- The proposed framework is simple and straightforward to use. The main technical contribution seems to be how to better use 3D data, ControlNet and LLM for data augmentation.
- Quantitative improvement looks promising. On the evaluated task (image classification and pose estimation), the quantitative improvement seems quite obvious, espcially for the OOD settings.

### Weaknesses
- The main method aims to produce (image, 3D annotation) pairs. Then is 2D image classification a good task for evaluation? The corresponding 3D ground truth is not used anywhere in this task. And you don't need 3D data to create (image, label) pairs. Even though experiments indeed show the improvement, I doubt this could be achieved without using 3D data.
- As the main purpose is to use the generated data for downstream tasks, I think the paper needs to carefully examine the data quality and show the necessity of the proposed approach. From this aspective, some necessary ablations are missing. 
  - One simplest baseline to use 3D data is to just use the rendered image with background (e.g., random environment map). Would this kind of synthetic data improve the evaluated tasks? I think this baseline is needed to prove the necessity of using a image generative model.
  - For image classification, a simpler approach is to just use the imagenet images, extract the edge map and then generate new image conditioned on the LLM prompt. Would this kind of synthetic data also give big improvement? This baseline is needed to show the necessity of using the 3D data, as least for the image classification task.
- The title is a bit misleading. It seems to suggest a method that enable 3D control of the diffusion model (e.g., changing view points), but it's not. The proposed method merely use the 3D data and diffusion model to create (image, 3D annotation) pairs.

### Questions
What is the prompt to LLM for enriching the description?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
- The author introduces a method named 3D-DST, aimed at enhancing the comprehension of 3D objects by diffusion models.
- This method comprises two modules: the "3D Visual Prompt" module, which utilizes edge maps as prompts derived from rendering, and the "Text Prompt" module, which extends prompt words through LLM.
- Through experiments, the author demonstrates that the images generated by the proposed method, along with paired labels, serve as an effective approach for data enhancement or pre-training. This leads to improved performance in tasks such as image classification and 3D pose estimation across multiple baselines.

### Strengths
- The paper is easy to understand.
- The paper presents an approach that incorporates edge maps as additional prompts to enhance the performance of the diffusion-based method.

### Weaknesses
- The framework is mainly inherited from Controlnet, so the technical contributions are limited and incremental.
- The idea of 3D Visual Prompt via CG rendering and LLM Prompt is more like a combination of multiple previous effective techniques.
- The author's excessive focus on introducing background knowledge of known technologies like diffusion or cross-attention is unnecessary if the method utilized in this article relies on off-the-shelf approaches. It is not recommended to extensively discuss these technologies in the main text.
- The second challenge, "simple text prompts," seems to be less directly relevant to the paper's introduction on adding 3D geometry control to diffusion models.
- The experimental part of the paper lacks details on training the network.

### Questions
How to define camera extrinsic matrix and whether to use class-level canonical-space as the extrinsic matrix of identity. If this is the case, there are many symmetric objects whose poses are ambiguous (this issue has been extensively discussed in the work on object 6dof estimation). How to define objects with multiple symmetry axes such as round tables? In addition, how to align the definition of extrinsic coordinate systems between different classes?
It is counterintuitive to claim that edge maps are superior to depth maps because depth maps provide more 3D information, such as occlusion relationships, which goes beyond the 2D representation of edge maps. The conclusions presented in the author's paper are difficult to support with only a few selectively chosen qualitative examples.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
A method to add 3D geometry control into the image generation process. To achieve the goal, three key techniques are leveraged, including 2D edge maps generation with 3D annotations via rendering, text prompt generation for improving the diversity, and conditional image generation from edge maps and text descriptions. The method can be utilized as a data augmentation strategy for many downstream tasks such as image classification. Experiments can show its promising application potential.

### Strengths
- Adding 3D geometry control via 2D edge maps and text descriptions is interesting and reasonable. This way the generative model only needs to deal with controlling information represented in 2D images and texts. Then many existing powerful techniques can be leveraged for controllable generation. 
- The proposed method is reasonable. It can achieve plausible controllable and diverse generation results. Generated images are of good quality and well-related to edge prompts and text conditions. 
- The method can serve as a promising data augmentation strategy for many downstream tasks. It is a promising way to generate diverse 2D images with 3D information annotation.

### Weaknesses
- The technical significance is relatively limited. The problem of generating 2D edge maps from 3D models and generating text prompts from 3D CAD models can be solved by existing techniques. Though the idea is interesting, no new techniques are proposed. The overall method is rather like an application-guided strategy. Though with promising application potential, it is hard to say what general principles that can guide the research in other domains can be distilled from the paper. 
- It is not sure whether the generated images are very faithful to the edge maps conditions. For example, there is no good guarantee that the objects in the generated images are consistent with the geometry described via the edge maps.

### Questions
- Evaluations on whether the generated images are faithful to images and text conditions. 
- It would be better if more potential applications could be discussed.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a novel method for adding 3D geometry control to diffusion models such that recognition models pre-trained on diffusion models generated synthetic data and then trained on target datasets have performance gains on classic tasks like 2D image classification and 3D pose estimation.

### Strengths
- The idea in this paper is neat, simple yet effective.
- The idea is also very novel.
- The empirical improvements on ImageNet classification and pose estimations are solid, significant, and surprising.

### Weaknesses
- In table 4, why the baseline result NeMo w/ AugMix is missing?
- Could you discuss or ablate using other rendering types other than canny edges? Does canny edges work the best and why?
- There is no discussion for limitations.

### Questions
see weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
