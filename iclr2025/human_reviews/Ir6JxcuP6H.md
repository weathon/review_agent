## Human Reviewer 1

### Summary
The paper aims to segment a wide range of concepts in video through a natural language. The paper proposes to rely on the powerful representation capabilities of the video diffusion model. Besides, to facilitate the research community, the paper introduces a new benchmark for Referral Video Process Segmentation (RVPS), which captures dynamic phenomena that exist at the intersection of video and language.

### Strengths
1. The focus on segmenting video processes presents a new research direction, which is valuable for exploring and evaluating model performance in this context.
2. The application of video diffusion to Referring Video Object Segmentation (RVOS) tasks is an interesting and promising approach.

### Weaknesses
1. The integration of video diffusion into referring segmentation appears incremental and lacks novelty, as it primarily adapts an existing video diffusion model to RVOS without specific tailoring for the task.
2. A detailed computational complexity analysis is needed to better understand the efficiency of the proposed method.
3. Both the figures and the overall writing quality require further refinement for better clarity and presentation.

### Questions
The definition of RVOS in the paper is **Referral** Video Object Segmentation in the introduction section, but it should be **Referring** Video Object Segmentation.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper explores using a pre-trained video diffusion model in referring segmentation. The paper proposes a new setting: Referral Video Process Segmentation, which is interesting. The model provides valuable insights on using generative prior knowledge to perform segmentation better.

### Strengths
1. The paper is well-written and easy to follow. 

2. The task of Referral Video Process Segmentation is interesting. 

3. The REM achieves comparable performance with the state-of-the-art method UNINEXT on the RES task and exhibits stronger performance on out-of-domain benchmarks.

### Weaknesses
REM uses a one-step approach to predict the segmentation mask during training. As a result, REM is more like resuming the pre-trained weights from the video diffusion model and training a U-Net segmentation model. I believe the model could achieve comparable performance without using the VAE Encoder to process the mask and instead using a few simple MLP layers. If this is the case, REM's performance gain may not be due to the authors' claim ``preserving as much of the generative model's original representation." Therefore, I think the authors need to add more detailed ablation studies to analyze the key factors leading to performance improvement.

### Questions
My biggest concern is listed in the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors present a diffusion-model based method for the task of Referring Video Segmentation, called REM, and a new dataset for the task called Ref-VPS. The motivation for this work stems from the observation that current datasets for the task of Referring Video Segmentation are typically extensions of object-centric datasets for Referring Video Object Segmentation, which has prevented development of general video region segmentation methods. To address this, the authors create their carefully curated dataset, Ref-VPS, from TikTok videos, and label it with "concepts" that are not necessarily object-centric. They create REM from diffusion models that, due to their pretraining on Internet-scale data, have generalizable representations. The authors show that on Ref-VPS and two Referral Video Object Segmentation, REM either outperforms or performs favorably compared to current Referring Video Segmentation baselines. On an ablation study, the authors validate their claim that preserving the architecture of diffusion models during fine-tuning for Referring Video Segmentation leads to better performance.

### Strengths
1. REM has a straightforward design for training and inference that is presented clearly. In general the writing is clear.
2. REM results are strong on Ref-VPS and Ref-YTB and the ablation study effectively validates the author's claim that retaining the architecture of diffusion models is important for taking advantage of their generalizable representations for referring video segmentation. The finding that improving video generation models leads to enhanced video segmentation performance is interesting.

### Weaknesses
1. The motivation of the paper is not very clear. Specifically, it is not clear whether the authors are advocating for distinguishing the Referring Video Object Segmentation benchmarks from the Referring Video Process Segmentation benchmark that they propose. If the authors are advocating for this (and thus for two different tasks), then the final output of the model should be evaluated differently for each task, but this doesn't seem to be the case.
2. The Ref-VPS dataset was created from heavily curated TikTok videos and thus may not be representative of real-world scenes and videos. A more in-depth comparison of Ref-VPS to a dataset comprised of random videos (such as YouTube-BoundingBoxes [1]) would strengthen the paper if it is found that Ref-VPS adequately represents a wide variety of scenes, especially since the authors claim that the generalization ability of their method is a contribution.
3. The details of the REM method's computational efficiency and a comparison to that of the baselines are missing. Specifically, the training time, inference time and amount of memory use on the GPU should be included and compared to the baselines. The ease of use of the method is important in determining its contribution.

[1] "YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video". Esteban Real, Jonathon Shlens, Stefano Mazzocchi, Xin Pan and Vincent Vanhoucke. CVPR, 2017.

### Questions
Questions:
1. Can you clarify whether you are proposing Referring Video Process Segmentation as a new/separate task?
2. What is the definition of "concept" and how is it different from a "class"? Is there high inter-annotator agreement on Ref-VPS? My concern is that the challenges of Referring Video Process Segmentation are due to the ambiguity of the task and the arbitrary nature of the label, rather than the lack of generalizable representations in the model.
3. In l. 157 can you clarify what you mean by "universal"? Do you mean generalizing to many domains?
4. In l. 184-188, what are the advantages of using such models? One can imagine that they are more lightweight or less computationally demanding than diffusion models.
5. In l. 275 what is t usually set to?
6. In l. 314 can you clarify what is meant by the "physical nature of the event"?
7. What architecture if REM specifically built on top of in the experiments in Section 5.1?
8. In Section 5.2 are the methods trained on the datasets as reported in Table 1?
9. In the "Fine-tuning strategy" section on page 10, out of curiosity, have you tried freezing the encoder and fine-tuning the decoder to preserve more of the representation?

Minor Typos/Writing Suggestions:
- l. 011 include what REM stands for
- l. 018 "crushing"->"crashing"
- "Referral" is used in l. 036, 039 but "Referring" is used in l. 134. It would be good to stay consistent to avoid confusion.
- l. 122 "approaches,"->"approaches"
- l. 160 "de-fact" -> "de-facto"
- l. 172 "highly-effective"-> "highly effective"
- l. 179 "RVOS,"->"RVOS"
- l. 260 "Figure 3 REM,"-> "Figure 3, REM"
- l. 263 "below"->"below."
- l. 313 "queries however" -> "queries; however"
- l. 331 "expression"->"expressions"
- l. 376 "video-segmentation"->"video segmentation"
- l. 377 "and we"->"and as we"
- l. 377 "by up to 46% out-of-domain in terms of Region Similarity" awkward wording
- l. 495 "StableDiffsuion"-> "StableDiffusion"

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper presents a method to adapt a video diffusion model (a text-to-video generative model) into a referral video segmenter (a text+video to masklet model).
Instead of segmenting clearly defined objects as conventionally done in image and video object segmentation, the authors focus on segmenting dynamic concepts such as a wave or a cloud of smoke. In order to evaluate their model on this task, the authors introduce the Referral Video Process Segmentation (Ref-VPS) benchmark, consisting of 111 annotated Tiktok videos.
Because of the lack of training data for segmenting such dynamic "entities", the authors propose to use a video diffusion model, which has learned the mapping from linguistic descriptions to video on massive amounts of text-video pairs. The model is very minimally adapted to the Referral Video Segmentation task in order to conserve its capabilities acquired during pretraining.

### Strengths
The paper tackles an interesting problem in video segmentation.
The approach is straightforward and well-motivated, leveraging existing diffusion models.
The contributions are clearly articulated, and the paper is well-structured.
The authors commit to releasing code, models, and data, which supports reproducibility of their results by the community.

### Weaknesses
Evaluation of Ref-VPS:
Dynamic concepts such as a wave or a cloud of smoke don't have clearly defined edges. Providing a single ground-truth contour for them is necessarily arbitrary. How then to evaluate? 
For example, in Figure 1, the mask for “the smoke dissipating” arguably does not cover all the smoke. Similarly, for ”the wave crashing in the ocean”, the ground-mask does not include the wave foam on the left. A model prediction that includes these elements would be unjustly penalised.
Segmentations on this kind of "dynamic concepts" are notoriously hard to evaluate but this issue is mostly not covered in this paper. The authors exclude the contour accuracy F metric but the J metric also suffers from this issue. There are possible solutions to alleviate this problem and their discussion is expected for a paper introducing a benchmark on segmenting "Dynamic concepts".

Runtime:
In order to be practical, the approach presented in this paper should have a reasonable running time. The paper does not give details on this subject.
For example, how long does it take for the model to segment a video of 256 frames, using an A100 GPU?

### Questions
In order to match the pre-training input distribution of the model, the authors add noise to the input (as performed during pre-training).
However, it corrupts the input image and reduces the information amount accessed by the model.
Have the authors experimented with not adding noise to the input? That would simplify the approach further.

How many frames does the model process at once? (What is the value of T ?)

Minor typos:
L309: "enities"
L386: "UINNEXT"
L495: "StableDiffsuion"
L523: "changing a little as possible"

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 5

### Summary
The REM paper introduces a novel framework for Referral Video Segmentation (RVS) that uses a video-text diffusion model to generalize segmentation to a wide range of visual concepts.REM effectively segments both common and rare objects, as well as non-object dynamic concepts like smoke, water, or changing light. The paper also introduces Ref-VPS, a new segmentation benchmark on dynamic processes that require both spatial and temporal modeling.

### Strengths
* REM achieves good generalization to out-of-domain concepts and non-object entities by leveraging Internet-scale pretraining on video-language data, outperforming specialized methods.

* The Ref-VPS benchmark provides a way to evaluate models on dynamic processes, filling a gap in existing RVS datasets by focusing on temporal events and continuous appearance changes.

### Weaknesses
* The REM method performs comparably to MUTR  and only slightly better then VLMO on Ref-YTB.

* Given use of a high-capacity diffusion model, there is limited discussion on its computational efficiency or inference speed. A comparison of REM's computational requirements with prior methods would be beneficial.

* Analysis on the impact of the quantity and quality of synthetic data would be useful - whether reducing/increasing the amount of synthetic data affect REM's performance.

* Since REM is applied to video segmentation tasks, especially focusing on continuous changes, it would be great to have some evaluation of temporal consistency.

### Questions
* How does REM handle ambiguous or overlapping concepts (e.g., scenes where multiple dynamic elements, like water and smoke, interact)?

* How critical is the quality of synthetic data for REM’s generalization across diverse concepts?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3