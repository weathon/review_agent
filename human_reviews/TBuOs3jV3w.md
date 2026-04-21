# Text-driven Editing of 3D Scenes without Retraining

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 1, 5

## Abstract
Numerous diffusion models have recently been applied to image synthesis and editing. However, editing 3D scenes is still in its early stages. It poses various challenges, such as the requirement to design specific methods for different editing types, retraining new models for various 3D scenes, and the absence of convenient human interaction during editing. To tackle these issues, we introduce a text-driven editing method, termed DN2N, which allows for the direct acquisition of a NeRF model with universal editing capabilities, eliminating the requirement for retraining. Our method employs off-the-shelf text-based editing models of 2D images to modify the 3D scene images, followed by a filtering process to discard poorly edited images that disrupt 3D consistency. We then consider the remaining inconsistency as a problem of removing noise perturbation, which can be solved by generating data with similar perturbation characteristics for training. We propose cross-view regularization terms to help the DN2N model mitigate these perturbations. Our text-driven method allows users to edit a 3D scene with their desired description, which is more friendly, intuitive, and practical than prior works. Empirical results show that our method achieves multiple editing types, including but not limited to appearance editing, weather transition, object changing, and style transfer. Most significantly, our method exhibits strong generalization of editing capabilities, eliminating the need to customize or retrain editing models for specific scenes or editing types. It realizes visual outcomes on par with or exceeding previous techniques needing iterative optimization while reducing editing time and memory overhead.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To tackle the issues of specific design for various editing types, this work proposes framework that allows for the direct acquisition of a NeRF model with universal editing capabilities, eliminating the requirement for retraining. After modify the 3D scene images, a filtering process to
discard poorly edited images and use generalizable nerf to get consistent views. Cross-view regularization terms are used here.

### Strengths
extensive experiments of different editing types are conducted;

### Weaknesses
1. Generalizable nerf model can be limited to novel view generation in very limited view range;
2. The robustness of fliter technique is questionable to me
3. Abalation study cases are very limited.

### Questions
1. when using nearby view supervision, how to get depth is not well explained;
2. how to deduce from eq3 to eq4 is not clear to me
3. how many images used for generalizable nerf synthesis?

### Soundness
2 fair

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
The paper proposes a framework for text-driven 3D scene editing tailored with image-based rendering. The author claims to achieve generalizable editing capability which facilitates training-free editing. The key contribution is to generate lots of multiview training data by adding perturbations to 3D scenes using BLIP→GPT→Null-text inversion pipeline. Then, a generalizable image-based rendering model is trained to get rid of scene-specific training. The experiments show promising results on 3D editing. However, the generalization to novel text/scene is not validated.

### Strengths
- The idea to combine generalizable image-based rendering with 3D scene editing is interesting.
- The pipline BLIP→GPT→Null-text inversion to generate edited images on caption-free multiview dataset is promising.
- The visual result is compelling with good demonstration on various types of scenes, including object-centric scene, face-forward scene, and unbounded scene.

### Weaknesses
Given the lack of validation on the generalizability, I lean to rejection at this point. I’m happy to be convinced by the authors’ response.

- My major concern is the generalizability. According to the paper and supplementary material, I do not find any descriptions on the train/test split for the experiments. If all the results presented in the paper are seen during training, the paper definitely overclaims the generalizability. Thus, the title is also misleading as “without retraining”.
- I hope to see more details about the setting of experiments and corresponding evaluation protocols:
    - The train/test split of text captions. If all captions used in the qualitative results are seen during training, please present the result of editing the scene with unseen captions
    - The train/test split of 3D scenes. If all scenes presented in the qualitative results are seen during training, please exclude some of them and retrain the model. Then test on those unseen scenes. Otherwise, the method still needs retraining given user-specific 3D scenes, which does not support the main claim.

### Questions
- Please see the weakness section for my concern about generalizability.
- What is the motivation of src_a and src_b. Please clarify the motivation for these two separate models.

### Soundness
1 poor

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper uses diffusion model for 2D edits by text and consolidates the edits in NeRF. It claimed to work across several appearance editing and style transfer and also reduce the time overhead.

### Strengths
1. The paper contributes to quite a relevant NeRF editing topic with broad interest. 
2. Extensive results are presented.

### Weaknesses
1.  As Introduction mentioned, existing methods typically rely on known editing types in advance with limited modification capabilities. Please elaborate on this claim and raise what editing types or some works who heavily rely on editing types. As of my knowledge and experience, Instruct-Nerf2Nerf or Clip-based methods are capable of any text input. They are not confined in specific editing types demonstrated in their paper. 

"These techniques are often less user-friendly" seems semantically overlap with prior two claimed challenges. It does not add new information. Suggest to take off. 

2. 2D editing does not guarantee multi-view consistency for each input. However, each edited single frame is well-structured and jumpy between multiview may not be simply noisy perturbations. For example, changing the selfie into Fauvism can change the selfie into multiple (colorful/ high contrast) possible output and each output is well-structured but quite different. The difference between diffusion model outputs may not be noisy perturbation. 

3. The writing is complex and need much more clarity. For example, what is the purpose of input caption? Does it aims to provide a description and replace nouns or subjects to create a target caption for editing? What is the generalizable mean in the context? 

4. There is not enough information for the inference stage. It only mentioned content filter in the paragraph. In Fig. 2 inference time, how can NeRF generate closed eye image if it has not seen examples in the training time? From the figure, after the volume rendering it shows an open-eye image, but the next step in G's output, it abruptly show closed eye image.

How to use the filtered image at the inference time to avoid re-training NeRF is also not clear to me.

5. The abstract claims doing appearance editing, weather transition, object changing, and style transfer. However, it seems the results only serve style transfer and appearance editing. It is not convincing to say changing to snow-covered roads is weather changes. The pineapple and strawberry examples seem just to change appearance only. It does not create new geometry.

### Questions
1. What exactly the "generalizable" term mean in NeRF? This term appears many times in the paragraph but is still vague. For example, what this NeRF model can attain that a vanilla NeRF model cannot.

2. In Eq. 6, how to get M for overlapping areas? Is it provided in the data?

3. In Fig.7, it mentioned a total of 6 methods for comparison but why in the exemplar Fig. 18 has only a total of 3 videos in comparison? (or is it just an example). How does one compute the ratios of one against the others in Fig. 7? It's hard to understand how the subjective test result in Fig. 7 is calculated. Besides, using only 4 scenes for a subjective test is far insufficient in my opinion. Also, please add a statistical significance p-value to validate the significance of the study.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a way to perform editing of 3D scenes represented using NeRFs using text description of the target. They use BLIP for image captioning, GPT to generate target captions, image editing using text, etc., in the training step and train a generalisable NeRF model. During inference, text-based image editing is performed and the images that are 3D inconsistent are removed using a filtering step. The remaining images are used to generate new views using a network derived from IBRNet. The paper presents different editing results. The paper presents many ideas which are mostly explained in the long supplementary/appendix section.

### Strengths
Text-based editing of 3D scenes is an important problem and provides opportunities to create many variations of 3D representations if it can be done well. The paper leverages many recent methods to achieve this task in some way. They train the NeRF model using many more losses to achieve "generalisability". There are many good results shown.

### Weaknesses
Presentation/Writing: The presentation needs to improve as it is not clear what their training achieves. It is impossible to understand the method to a reasonable extent without reading the appendix or supplementary that runs into 15 pages. See my questions below.

Significance: While text-based editing will be useful, what is the predictability and controllability of such a method? How does one know if the generated model is according to the textual targets given? It seems there are few ways to do that beyond basic qualitative or visual assessments. This is a serious limitation of many methods that rely on a generative tool as there is little controllability or predictability on what such a tool generates. The DN2N method presented also has that problem also: what does the text-based-editing block generate at inference time? The method can only filter out inconsistent images. That is, it can only discard the images generated; it can't generate a better image by influencing the editing module. How do we know if selected good results only are shown here? The discussion in Section 4.5 needs to be far more elaborate.

### Questions
I have several doubts/questions about the inference or editing stage. These should be in the main paper clearly. 
- What is the "effort" involved during inference? How many 2D edited images are generated? How many are found to be inconsistent on an average and discarded?  What is the time taken?
- Were there failure cases when sufficient # of consistent images couldn’t be generated? Is the whole process iterated on if that happens?
- Are edited images generated for the same camera poses of the input images used for training? Can other viewpoints be used?
- Is there any way to ensure or control what we want of the text-based image editing module? Your method is very critically dependent on this module generating good ones. Your editing capability is limited seriously by this aspect. Please see my comments under "weakness".
- The filtering step is rather too simple. How are the 4 measures used in the tuple prioritised or weighted? The details of the sorting could not be found anywhere. Why do you eliminate the top 10% of the matches? Aren't they the best?

Also, why is DN2N's output falling short in some methods in the user study?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
