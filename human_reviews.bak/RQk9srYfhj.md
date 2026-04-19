# Repositioning the Subject within Image

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 5

## Abstract
Current image manipulation primarily centers on static manipulation, such as replacing specific regions within an image or altering its overall style. In this paper, we introduce an innovative dynamic manipulation task, subject repositioning. This task involves relocating a user-specified subject to a desired position while preserving the image's fidelity. Our research reveals that the fundamental sub-tasks of subject repositioning, which include filling the void left by the moved subject and reconstructing obscured portions of the subject, can be effectively reformulated as a unified, prompt-guided inpainting task. Consequently, we can employ a single diffusion generative model to address these sub-tasks using various task prompts learned through our proposed task inversion technique. Additionally, we integrate pre-processing and post-processing techniques to further enhance the quality of subject repositioning. These elements together form the foundation of our SEgment-gEnerate-and-bLEnd (SEELE) framework. To assess SEELE's effectiveness in subject repositioning, we assemble a real-world subject repositioning dataset called ReS. Our results on ReS demonstrate the exceptional quality of repositioned image generation.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to reformulate the task of repositioning the subject within an image as a unified prompt-guided inpainting task. Specifically, different sub-tasks are achieved by the proposed task inversion technique, as well as pre/post-processing techniques. To verify the effectiveness, the authors also propose a real-world subject repositioning dataset.

### Strengths
1. The authors provide a detailed analysis and a formulation for the target of repositioning the subject within an image, the proposal is interesting.
    
2. This paper shows some promising results in terms of repositioning the subject.
    
3. The authors collect 100 real image pairs for object repositioning. Specifically, the masks are claimed refined by experts. This dataset can be useful to the community for further applications.

### Weaknesses
1. The presentation of this paper needs grand improvements. For example,
    
    1. The organization of this paper is confusing. The authors formulate the repositioning task as three sub-tasks, i.e., local inpainting, subject completion, and local harmonization, and try to solve them by task inversion. However, They only introduce subject moving and completion in Section 2.3 and consider harmonization as post-processing in Section 2.4. Besides, the authors introduce the task in Section 2.3 but leave the whole picture until Section 2.4 which introduces both SEELE framework and harmonization. The organization can make readers hard to follow.
        
    2. I think Figure 2, the pipeline overview is important for readers to catch up with the whole picture. However, there are multiple input and output lines in the Manipulation Model, which also confuses me a lot, what input and output for which sub-task? Besides, in Figure 2, harmonization is in the post-processing step beyond the scope of the manipulation model, which conflicts with the claim as ‘a unified single model‘ in the Introduction.
        
    3. Many important details are missing in the paper. For every qualitative result, the authors should denote the moving mask, and completion mask in the inputs.
        

2\. The motivation/novelty is not that convincing to me. Here are some justifications,

1. The authors claim that they “introduce an innovative dynamic manipulation task”, however, Zhan et al. [1] have introduced the same task and a complete pipeline (also contains local inpainting and subject completion) for the same target. What’s more, I didn’t see much superiority of SEELE over Zhan’s solution given that the proposed SEELE requires many manual efforts. The authors should discuss and compare their solutions.
    
2. local inpainting and subject completion are actually can be unified as completing the region according to context (background or object). Why do we need to explicitly split them into two distinct sub-tasks? What’s the effect if we use opposite task prompts for different sub-tasks?
    
3. The optimization objectives in EQ(2) is confusing. Why do we need an extra term, i.e., x - x\\*?
    
4. In section 2.3, the authors claim that they learn move-prompt for subject-moving tasks. however, the optimization target is more like local inpainting instead of the subject moving, which also involves moving and blending the subject in a new place.
    

3\. I’m afraid the experimental results are not strong enough to support the paper’s claim. Specifically,

1. Since the authors use both task inversion and LoRA adapters for local harmonization, we suggest the authors conduct an ablation study to verify the necessity of using LoRA and the effectiveness of task inversion here.
    
2. The inpainting results of SEELE seem not strong enough. For example, the wood texture looks blurry and flattened in Fig. 1. The perspective issue remains in the SEELE in the 4th case in Figure 6 and most results in Figure 6 look similar. Besides, the results of LAMA and MAT in the 5th case in Figure 6 look weird. It seems that LAMA and MAT didn’t perform subject completion as they have very straight boundaries of the subject.
    
3. The authors introduce that they use SAM to segment the entire image into distinct subjects in the pre-processing step. We suggest the authors show the SAM results for readers as a reference.
    
4. The comparison in Figure 11 is not fair. Specifically, the authors should also show the results of SD by using object prompts given that SEELE already requires many user efforts. Besides, it is not clear how to get the results of SEELE in Figure 11, by local inpainting or subject completion.

### Questions
See more details in the Weaknesses. I'm looking forward to the authors' response that addresses my concerns listed in the Weaknesses and am willing to raise my rating upon the response.

### Soundness
3 good

### Presentation
2 fair

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
This paper proposes a segment-generate-and-blend framework with a preprocessing, manipulation, and post-processing pipeline to address the challenging object reposition task in a single image. In addition, the authors also assembled a real-world subject repositioning dataset called ReS which consists of 100 real image pairs featuring a repositioned subject. The main contribution of this work is to convert task instructions to a text embedding in the text-condition space. Besides, it employs other SOTA preprocessing (image segmentation and occlusion relationship estimation) and post-processing algorithms to optimize the whole object reposition performance.

### Strengths
1. This is the first framework after the deep learning era that deals with the object reposition issue in a still image.
2. This paper defines the object reposition task as a comprehensive image manipulation task that has several sub-tasks, including local inpainting, subject completion, and local harmonization, which is a new perspective.

### Weaknesses
1. There is an object reposition paper [1] before the deep learning era, which should be mentioned by you.
2. I think the most important part and contribution is the definition of task inversion in Sec. 2.2. But I still don't understand how to "translate" users' intentions into the so-called task embedding (task-specific prompts). How do you represent the "arrows" in figures 6 and 7 as task embeddings? The key step is not very clear in your paper.
3. You introduce a LoRA adapter when learning local hormonization, but how do you integrate it with the manipulation process? The statement is not very clear.


[1] Iizuka, Satoshi, et al. "Object repositioning based on the perspective in a single image" Computer Graphics Forum, Vol. 33, No. 8, 2014.

### Questions
1. In Figure 2, I don't identify the difference between the completed image and the output image after postprocessing. Can you provide a further explanation?
2. In Figure 4, how to obtain the mask for object completion as shown on the right side? The description is not very clear.
3. In the example shown in Figure 1 above, how can you determine whether to move the little girl only or the ballon and chair together?

### Soundness
3 good

### Presentation
2 fair

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
This paper provides a subject repositioning task that reposition subjects within their images, which was recently introduced by Magic Editor in Google Photos (https://blog.google/products/photos/google-photos-magic-editor-pixel-io-2023/). This work combines many existing techniques, including pre-trained SAM, diffusion model, as well as image inpainting and harmonization, to build the Segment-generate-and-bLEnd (SEELE) framework for dealing with the subject repositioning task. Inspired by the textual inversion, the authors propose the task inversion for providing text prompts as input instructions of diffusion model to move or complete the subjects in images.

### Strengths
- This work can be considered a detailed technical document of the Magic Editor in Google Photos.
- The provided dataset ReS is helpful to evaluate the performance of subject repositioning task.

### Weaknesses
About the contributions:
- First, the authors claim that this work introduces the concept of subject repositioning, but obviously, it is one of the important features released by the Google Photos in May 2023, named Magic Editor in Google Photos (https://blog.google/products/photos/google-photos-magic-editor-pixel-io-2023/).
- Second, the proposed framework, namely SEELE, combines many existing techniques, and can be considered as an engineering work (like Magic Editor in Google Photos). Moreover, the authors claims that the SEELE addresses multiple generative sub-tasks in subject repositioning using a single diffusion model, on one hand, SEELE actually contains many components not only the diffusion model for generative sub-tasks, on the other hand, it is confusing what is the exact meaning of using a single diffusion model since it seems that different generative sub-tasks are tackled with different datasets at least.
- Third, about the task inversion, from Figure 3, it seems that the differences are small since it's not difficult to change the textual inversion to task inversion by using different training datasets.
- At last, ReS dataset is helpful for the evaluation of subject repositioning task, however, the comparison of this work with Magic Editor of Google Photos is not well addressed in this paper.

About the reproducibility:
- This work contains many different components with different training datasets as well as different training strategies, thus it is not easy to understand and reproduce the details.

### Questions
- What about the comprehensive comparison between this work and the Magic Editor in Google Photos?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the subject repositioning task, and decouples this task into filling the void left by moving, reconstructing obscured portions, and harmonious blending. They address these subjects by the diffusion model based on task prompts. Furthermore, they propose the RES dataset for testing the performance of the model.

### Strengths
1.	The paper proposes the SEgment-gEnerate-and-bLEnd (SEELE) framework to solve subject repositioning problems uniformly.
2.	The paper proposes task inversion technique to generate task-specific prompts to replace text embedding in traditional Stable Diffusion.
3.	The paper proposes the RES dataset for testing the performance of the subject repositioning method.

### Weaknesses
1. The main concern lies in the effectiveness of the proposed task inversion method on subject repositioning. According to the quantitative comparison provided, task inversion seems to improve specific single tasks, but the improvement in subject repositioning is small (see Table 1). In the overall complex process of subject repositioning, task reversal methods may play a weak role.
2. The size of the proposed dataset may be a bit small, limiting the evaluation of the method's effectiveness.
3. It would be better to give the prompt when using SD in Table 1.

### Questions
Please see 'Weaknesses'.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
