# SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 5

## Abstract
The landscape of publicly available vision foundation models (VFMs), such as CLIP and Segment Anything Model (SAM), is expanding rapidly. VFMs are endowed with distinct capabilities stemming from their pre-training objectives. For instance, CLIP excels in semantic understanding, while SAM specializes in spatial understanding for segmentation. In this work, we introduce a simple recipe to efficiently merge VFMs into a unified model that assimilates their expertise. Our proposed method integrates multi-task learning, continual learning techniques, and teacher-student distillation. This strategy entails significantly less computational cost compared to traditional multi-task training from scratch. Additionally, it only demands a small fraction of the pre-training datasets that were initially used to train individual models. By applying our method to SAM and CLIP, we derive SAM-CLIP : a unified model that amalgamates the strengths of SAM and CLIP into a single backbone, making it apt for edge device applications. We show that SAM-CLIP learns richer visual representations, equipped with both localization and semantic features, suitable for a broad range of vision tasks. SAM-CLIP obtains improved performance on several head probing tasks when compared with SAM and CLIP. We further show that SAM-CLIP not only retains the foundational strengths of its precursor models but also introduces synergistic functionalities, most notably in zero-shot semantic segmentation, where SAM-CLIP establishes new state-of-the-art results on 5 benchmarks. It outperforms previous models that are specifically designed for this task by a large margin, including +6.8% and +5.9% mean IoU improvement on Pascal-VOC and COCO-Stuff datasets, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work aims to unify CLIP and SAM – two powerful vision foundation models (VFMs) – to enable a single set of parameters that are capable of retraining the advantages of both VFMs. The authors treat such model merging as a continual learning problem, where, given a pretrained VFM, the knowledge of a second VFM is merged without forgetting the initial knowledge.

The proposed model, SAM-CLIP, assumes access to a small part of pretraining data or its surrogates to be replayed during the merging process. The SAM model is used as the base VFM during the distillation, where CLIP is regarded as the auxiliary VFM and its knowledge is distilled via a cosine distillation loss. To avoid the catastrophic forgetting of SAM’s original capabilities, the authors propose a rehearsal-based multi-task distillation loss to gradually distill the external knowledge to the base VFM.

The resulting trained SAM-CLIP is able to perform zero-shot classification, image-text retrieval, instance segmentation, and semantic segmentation. Across several benchmark datasets, the authors show that SAM-CLIP can achieve state-of-the-art performance in a single-stage inference setup.

### Strengths
- The paper endeavors to create a comprehensive model by merging pre-trained vision foundation models, aligning with the contemporary trends in computer vision.
- The contributed SAM-CLIP stems from a continual learning perspective, which is intuitive. As a result, SAM-CLIP is capable of conducting multiple visual understanding tasks in a zero-shot manner.

### Weaknesses
- A glaring omission in the paper is the technical detail surrounding the cross-VFM distillation. A deeper dive into the methodology, choices of operations, and potential effects of the framework is necessary.
- The paper's structure and presentation could use refinement. The disproportionate emphasis on background and literature review, coupled with scant technical details, detracts from its overall coherence and depth.
- Benchmarking SAM-CLIP against prior models, particularly those based on SAM, would offer a more rounded perspective on its performance and advantages.

### Questions
- **Q1:** The efficiency of SAM-CLIP on edge devices is emphasized multiple times throughout the manuscript, particularly in the “Abstract” and “Introduction” sections. However, the empirical evidence supporting SAM-CLIP's performance on such devices seems absent. Could the authors elucidate the specifics of the claim regarding SAM-CLIP’s suitability for edge devices? The reviewer would like to know what the claim means by “apt for edge device applications”.

---

- **Q2:** When assessing zero-shot semantic segmentation, SAM-CLIP is exclusively juxtaposed with CLIP-based models. How does SAM-CLIP fare when contrasted with SAM-centric models, notably Semantic-SAM [R1] and SEEM [R2]?

---

- **Q3:** The “Proposed Approach” section might benefit from more detailed explanations regarding the design and implementation. In particular, how do you perform the joint training between head probing and multi-task distillation?  how is the balance between head probing and multi-task distillation maintained during joint training? What metrics or criteria guide the selection of appropriate stopping points for training?

---

- **Q4:** The “Background” section contains a profusion of general literature introductions. A more succinct and discerning review that delves into comparative analyses would greatly enhance its value.

---

- **Q5:** Notable typos appeared in the current illustration of this paper, which should be revised accordingly. For example:
  - Page 1, the last paragraph: there should be a space between “tasks” and “Fifty et al., 2021”.
  - Page 2, the first paragraph: “consuming massive amount …” should be “consuming a massive amount …”.
  - Page 2, the first paragraph: “how to access to …” should be how to access …”.
  - Page 2, the second paragraph: “generalization to diverse set of tasks” should be “generalization to diverse sets of tasks”.
  - Page 2, the third paragraph: “we allow access to small part of …” should be “we allow access to a small part of …”.
  - Page 3, the first paragraph: “With compromise of a negligible drop …” should be “With a compromise of a negligible drop …”.
  - Page 3, the second paragraph: “enable additional zero-shot capabilities” should be “enabled additional zero-shot capabilities”.
  - Page 3, the second paragraph: “on-top of …” should be “on top of …”.
  - Page 3, the third paragraph: “a model, and training recipe …” should be “a model, and a training recipe …”.
  - Page 3, the third paragraph: “and produce high-resolution segmentation mask” should be “and produces a high-resolution segmentation mask”
  - Page 3, the third paragraph: “but has not released …” should be “but have not released …”.
  - Page 3, the fourth paragraph: “They show transfer of the same …” should be “They show the transfer of the same …”.
  - Page 3, the fourth paragraph: “and demonstrate transfer of different zero-shot capabilities” should be “and demonstrate the transfer of different zero-shot capabilities”.
  - Page 3, the fourth paragraph: “as well as emergence of new zero-shot capability” should be “as well as the emergence of new zero-shot capability”.
  - Page 3, the fifth paragraph: “referring to loss of previously learned knowledge due to …” should be “referring to a loss of previously learned knowledge due to …”.
  - Page 4, the third paragraph: “to obtain segmentation mask” should be “to obtain segmentation masks”.
  - Page 4, the third paragraph: “and many forward passes, make their deployment …” should be “and many forward passes, making their deployment …”.
  - Page 4, the fourth paragraph: “the optimization algorithm is exploring the parameter space …” should be “the optimization algorithm explores the parameter space …”.
  - Page 5, the second paragraph: “and inherits its …” should be “and inherit its …”.
  - Page 5, the sixth paragraph: “which is the case of our experiment of …” should be “which is the case in our experiment of …”.

---

- **Q6:** How does SAM-CLIP perform under out-of-distribution or data corruption cases?

---

References

- [R1] F. Li, et al. “Semantic-SAM: Segment and Recognize Anything at Any Granularity.” arXiv preprint arXiv 2307.04767.

- [R2] X. Zou, et al. “Segment Everything Everywhere All at Once.” arXiv preprint arXiv  2304.06718.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes SAM-CLIP to build a unified model with both the strengths of SAM and CLIP.  SAM and CLIP is employed to share the same image encoder with two separate heads. Two phased are adopted during the KD process: 1) Head probing 2) Multi-task distillation. Also 40.8M images are used in the distillation process. The results are validated on zero-shot instance segmentation, semantic segmentation and classification benchmarks.

### Strengths
1. The paper has a good motivation on merging two visual foundation models, i.e., SAM and CLIP, into a unified model, such that the distilled model can obtain both semantic and spatial understanding.

2. The paper is well organized and easy to understand.

3. The experiments in Figure 1 and the experiment section show the distilled model retains both good zero-shot ability from SAM and CLIP.

### Weaknesses
1. When evaluating zero-shot semantic segmentation, as in Figure 3, the paper proposes a two-stage process to first using clip head for coarse masks predictions and taking it as input to SAM for refinement. Is the predicted masks by SegCLIP in Table 2 also refined by SAM? Can the authors also provide the zero-shot semantic segmentation without using geometric prompts?

2. When evaluating zero-shot instance segmentation, the performance decrease on LVIS is not negligible. This suggests that the ability of SAM is decreasing after the distillation process. Can the authors also provide comparison to HQ-SAM on zero-shot instance segmentation with the same bounding box as prompt? HQ-SAM [a] is also designed for minimal forgetting and efficient tuning for SAM but without discussion in related works or results comparison. Also, the paper misses MobileSAM in the related work section, which also uses knowledge distillation.

[a] Segment Anything in High Quality. NeurIPS, 2023.
[b] Faster Segment Anything: Towards Lightweight SAM for Mobile Applications. arXiv:2306.14289.

3. Since the paper mentions edge device applications in the abstract, what are the model size, speed and memory consumption of the proposed sam-clip comparing to SAM/CLIP?

4. What is the influence of the dataset scale in Merged-41M, for example reducing images by half or further increasing the image number? How does the paper decide the respective data percentage for CLIP and SAM training? Also, how to decide the distillation loss value scales for the sam head and clip head, like 1:10?

### Questions
Can the method deal with the instance segmentation not using bbox as prompt but using the semantics from CLIP? Overall I am positive about this paper and willing to raise scores if my concerns in the weakness can be well addressed.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper merges CLIP and SAM, the two foundation models, into a single one that assimilates both knowledge and expertise learned separately. Specifically, the technical contributions include a reasonable finetuning design and integration of the two distillation losses. The resulting model supports language-driven prompts and enjoys a high-quality segmentation result.

### Strengths
1. This paper presents a simple yet effective way to merge two foundation models into a single one, and it inherits both advantages and demonstrates a significant performance boost;
2. The paper is well-organized, clearly written, and easy to follow;
3. The resulting model is promising and helpful for future research.

### Weaknesses
1. The resulting model inherits the zero-shot capability of CLIP, as demonstrated in Table 1-5. However, it seems that there is no evidence showing the resulting model does not suffer from catastrophic forgetting. Even though the segmentation performance is better than CLIP-head prediction, it still doesn't compare with the segmentation result of SAM and it is unclear how much performance is degraded compared to the original SAM. The demo in Figure 3 shows that the SAM-head refined output is still filled with some artifacts and seems to have a large performance gap with the original SAM.
2. The proposed method is limited to the sizes of released SAM models. Since the vision encoder must be initialized SAM vision encoder, we cannot obtain a resulting model with an arbitrary size.

### Questions
1. The authors should explain more clearly the performance gap with the original SAM in terms of segmentation quality.
2. The authors should also give the output of the original SAM, with the same examples shown in Figure 3.
3. The authors should discuss more limitations with the resulting model and the proposed method.

If the above concerns are addressed, I am willing to improve the rating.

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a distillation paradigm to incorporate SAM and CLIP, combing their instance segmentation and semantic recognition capabilities. SAM-CLIP uses extensive pre-training data from original models and learns a unified encoder along with two task-specific heads. SAM-CLIP showcases good performance across zero-shot classification and segmentation tasks.

### Strengths
1. The motivation is reasonable to combine SAM and CLIP to infuse their own advantages.

2. SAM-CLIP shows good performance on zero-shot semantic segmentation tasks.

3. The writing is clear and easy to follow.

### Weaknesses
1. The contribution is a little overclaimed as *'we introduce a simple recipe to efficiently merge VFMs into a unified model that assimilates their expertise.'*. I think this method is specifically designed for CLIP and SAM, and cannot be simply generalized to other VFMs.

2. The cost of training SAM-CLIP is expensive. The training data includes many sources up to 41M. Considering CLIP and SAM have already cost large-scale pre-training resources, continually tuning them as SAM-CLIP is not cost-effective. Although SAM-CLIP achieves good results for semantic segmentation, it hurts the original performance of both SAM and CLIP. I think simply cascading SAM and CLIP in a training-free way (CLIP generates prompt by vision-language alignment and then SAM segments or SAM segments all objects and then CLIP classifies) can obtain even comparable results to SAM-CLIP, which is more practical in real-world applications.

### Questions
SAM itself can also be prompted by texts (semantics), though not open-sourced. What's the advantage of SAM-CLIP compared to SAM with text prompt?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
