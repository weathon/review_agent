# PixelGaze: Toward Pixel-Level Gaze Target Prediction in Natural Scenes

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Following the gaze of other people and analyzing the target they are looking at can help us understand what they are thinking, and doing, and predict the actions that may follow. Existing methods for gaze following primarily focus on gaze points or heatmaps rather than objects, making it difficult to deliver clear semantics and an accurate scope of the gaze-at targets. To address this shortcoming, we propose a novel gaze target prediction method named PixelGaze, that can effectively leverage the spatial visual field of the person as guidance, enabling a progressive coarse-to-fine process for gaze target segmentation and recognition. Specifically, a prompt-based visual foundation model serves as the encoder, working in conjunction with three distinct decoding modules (e.g. FoV perception, heatmap generation, and segmentation) to form the framework for gaze target prediction. Then, with the head bounding box performed as an initial prompt, PixelGaze obtains the FoV map, heatmap, and segmentation map progressively, leading to a unified framework for multiple tasks (e.g. direction estimation, gaze target segmentation, and recognition). In particular, to facilitate this research, we construct and release a new dataset, comprising 72k images with pixel-level annotations and 270 categories of gaze targets, built upon the GazeFollow dataset. The quantitative evaluation shows that our approach achieves the mIoU of 34.9% in gaze target segmentation and 45.1% recognition accuracy. Meanwhile, our approach also achieves state-of-the-art performance on the gaze-following task.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses gaze target prediction in natural scenes and introduces the novel task of gaze target segmentation. To enable this, the authors augment the GazeFollow dataset with pixel-level annotations, creating the first third-person, pixel-level gaze target segmentation dataset. Building on this dataset, they propose PixelGaze, a unified multi-task framework that performs gaze following, segmentation, and recognition simultaneously. The framework consists of three key modules: a) 3D Field of View (FoV) Perception that generates a 3D gaze cone using head bounding boxes and depth information to provide a precise foundation for gaze prediction.
b) FoV-aware Heatmap Generation that integrates spatial FoV and scene context to predict gaze-at locations via a heatmap.
c) Segmentation and Recognition that combines heatmap cues with mask prompts for pixel-level segmentation and recognition, bridging heatmap and mask predictions using differentiable coordinate regression.
The framework is optimized with task-specific losses as well as novel FoV supervision and mask heatmap matching losses, achieving more accurate gaze target prediction.

### Strengths
- The paper presents a unified framework for gaze target detection and segmentation. The segmentation task is novel in gaze behavior analysis and could be useful for providing more fine-grained detections.

- It also introduces new annotations, consisting of carefully curated segmentation masks based on samples from the GazeFollow dataset (but also see the weaknesses section)

### Weaknesses
1) The comparisons do not seem entirely fair. For example, in Table 1, the proposed method uses the head bounding box as a prior, whereas many of the methods listed in the table simultaneously detect head locations (e.g., Tu et al., Tonini et al.). Please confirm this and justify how this relates to the results.

2) One of the main contributions of this study is the new annotations. However, without understanding how they were collected, reviewers cannot assess whether the results are reliable. Unfortunately, there is no explanation regarding the annotations in the main paper, appendix, or supplementary video. Key details are missing, such as (a) how you annotated it, (b) how many people participated in the annotation process, (c) whether the same image was annotated by more than one person, and (d) whether additional verification of the annotations was performed. These details are crucial in gaze research, as subjectivity plays a significant role in annotating gaze images, which is reflected in the high variance of gaze point locations in the GazeFollow test set (already shown by many prior papers). It is also unclear whether the entire GazeFollow dataset was annotated, which seems unlikely given its size and the multiple gaze annotations per image, or just a portion of it. From Fig. 3c, it seems a portion. Can you clarify? 

3) Not clear why there are no comparisons on the GOO dataset, which in fact supplies pixel-level annotations. Could you please explain?

4) The paper includes a “w/o DSNT” ablation, showing minor performance drops when replacing the DSNT layer with a non-differentiable argmax. However, the motivation for choosing DSNT over direct coordinate regression or simple heatmap sampling is not conceptually discussed, and the reported improvement is modest without clear justification.

5) The paper lacks a comparison with a simple yet informative baseline where the segmentation branch is replaced by a frozen semantic segmentation model (e.g., SAM) that generates a mask using the gaze point predicted by the heatmap module. Such a “naïve” setup would provide a clear reference for assessing the real contribution of the proposed segmentation and recognition branches. It would help determine whether the performance gains stem from the proposed design or could be similarly achieved by combining standard gaze prediction with an off-the-shelf segmenter.

6) It is not clear what the failure cases are or why they occur (can be given in appendix). Overall, the paper provides very little discussion or explanation of the design choices, and most of the parameters appear to have been set empirically. It also seems that no separate validation split was used for parameter tuning across the datasets. Can you explain?

7) [Minor] It seems that the Supp. video was prepared for CVPR 2025 submission, but including the name of the paper, should be changed.

### Questions
In addition to the above questions:

1) During progressive training, the model is first trained (PixelGaze w/o Seg) and then frozen while adding the segmentation module. The paper doesn’t explain why freezing is beneficial. For example, was joint fine-tuning unstable? It’s unclear whether end-to-end fine-tuning might improve performance. Such info can be included in an appendix, which is nowadays extensive in top conferences.

2) The loss weights $(\alpha_{1}, \alpha_{2}, \beta_{1}, \beta_{2}, \lambda_{1}, \lambda_{2})$ are said to be ``empirically set,'' but there is no description of how these values were chosen, whether they were tuned per dataset, and how sensitive the performance is to these parameters. Without this information, it is difficult to reproduce the results or assess fairness in comparison.
 
3) What is the significance of the segmentation and recognition tasks' results between your method and the others?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PixelGaze, a new method for gaze target prediction that goes beyond traditional gaze point or heatmap estimation. Instead of only predicting where a person is looking, PixelGaze identifies which object they are looking at and segments it at the pixel level. The method uses a coarse-to-fine pipeline guided by a visual foundation model. Starting from the person's head bounding box as a prompt, the model progressively generates a field-of-view (FoV) map, a gaze heatmap, a pixel-level segmentation map of the gaze target, along with its object category. To support this task, the authors create a new dataset of 72k images with pixel-level gaze target annotations and 270 object categories, derived from GazeFollow. Experiments show that PixelGaze achieves 34.9% mIoU for segmentation and 45.1% recognition accuracy, outperforming previous gaze-following methods.

### Strengths
1. This paper go further from gaze target detection to gaze target segmentation.
2. The proposed method is effective for doing this problem.
3. A dataset and benchmark GazeSeg is proposed.

### Weaknesses
1. No detailed description of GazeSeg. Moreover, the construction process of the benchmark is missing. The labels depend heavily on GazeFollow dataset, making the contribution limited.
2. The novelty of the proposed method is limited. The proposed method is just a concatenation of FOV perception, heatmap generation, segmentation modules.
3. The segmentation function is significantly affected by the localization performance. What is the motivation of training a new segmentation module? The three modules are trained one by one or together?

### Questions
1. How was the ground truth made?
2. The three modules are trained one by one or together?
3. What is the performance if use your backbone + SAM + CLIP

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors explore pixel-level gaze target prediction through their proposed PixelGaze method which performs multiple tasks such as direction estimation, gaze target segmentation and recognition. PixelGaze consists of several modules dedicated to 3D FoV perception, heatmap generation and segmentation. To facilitate their multi-task method on fine-grained gaze target following and recognition, the authors also extend Gazefollow dataset by including segmentation mask ground truth annotations for the gaze-target object. The method achieves impressive results on existing gaze following datasets using traditional gaze following metrics. Additionally, they achieve SOTA on the new gaze target prediction tasks such as segmentation and recognition.

### Strengths
1. The extension of gaze following to pixel-level understanding that the paper is trying to solve is interesting and relevant in real-world applications.
2. The PixelGaze method has been designed well and makes sense intuitively.
3.  PixelGaze achieves good performance on the traditional gaze following task, and also on their proposed segmentation and recognition tasks.

### Weaknesses
1. Localization Results for Gazefollow are worse for PixelGaze, while that for  VideoAttentionTarget and ChildPlay datasets are only slightly better for some metrics (note that AP value [0.930] for PixelGaze is in bold signifying best result, but Tonnini et al. achieve higher AP value [0.934]). 
2. There are several missing portions in the experimental results in Tables 1 and 2: (a) Ours (w/o Seg) for VideoAttentionTarget dataset; (b) Params column for several previous methods which have publicly available code (such as Tafasca et al. (2024), Tafasca et al. (2023), Ryan et al. (2024)). Why were these not reported?

### Questions
Please respond to my question in Weakness point 2. Additionally, I have a couple of questions and a comment: 

1. Have you tried using mobileSAM and CLIP instead of the segmentation and recognition module, to have an apple-to-apple comparison with your baselines in Table 2? 
2. What do you think are the limitations of your method?
3. Lastly, a comment: the way you have cited papers without enclosing parentheses (for instance, see lines 130-136) looks off to me. This can be rectified using a different LaTeX command.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces PixelGaze, a unified model designed to predict not only where a person is looking within an image but also to produce a fine-grained, pixel-level segmentation map of the gaze target. To enable this capability, the authors also present a new  dataset, GazeSeg, an extension of the GazeFollow dataset. GazeSeg contains 72,000 images with pixel-level gaze target annotations across 270 object categories, providing a comprehensive resource for training and evaluating gaze target segmentation models in natural scenes.

### Strengths
GaseSeg is a dataset contribution with pixel-level masks, and semantic labels, likely to stimulate further research.

### Weaknesses
he novelty of the paper is somewhat limited. While the overall framework integrates multiple modules, several components are not clearly explained or insufficiently justified. The training procedure involves a two-stage setup, but it is unclear why this design is necessary or why the model cannot be trained end-to-end. As a result, the work feels more like an engineering effort that combines existing modules rather than a fundamentally novel methodological contribution. In addition, there are inconsistencies and typographical errors in the paper—for example, at lines 176 and 180, the dimension of  Pw init is described once as  R2 and once as
R1 . The description of the encoder architecture is also vague, leaving it unclear which backbone or features are used. Similarly, the section on person-specific FoV generation lacks sufficient implementation and mathematical detail to be fully reproducible.

there are also typos in related works such as line 123-124.

### Questions
- Why is the bounding box prediction not sufficient for this task? Can you explain more clearly why it is not semantically meaningful compared to your segmentation-based approach?
- How often is the predicted label from the bounding box actually different from the label obtained from the segmentation map? Some quantitative comparison between these two outputs would help justify the need for pixel-level segmentation.
- what is the baseline exactly for 34% in gaze target segmentation and 45% accuracy? to what baseline is this being compared to?
- in line 175 what is the encoder model?
- in table 1 where the human values are coming from -reference is missing. 
- how the annotation were done, was a SAM model used for annotation? there is also a big portion of others, what is included in that, how the categories were selected for annotation?

### Soundness
2

### Presentation
1

### Contribution
2
