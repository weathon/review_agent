# What, when, and where? -- Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
Spatio-temporal grounding describes the task of localizing events in space and time, e.g., in video data, based on verbal descriptions only. Models for this task are usually trained with human-annotated sentences and bounding box supervision. This work addresses this task from a multimodal supervision perspective, proposing a framework for spatio-temporal action grounding trained on loose video and subtitle supervision only, without human annotation. To this end, we combine local representation learning, which focuses on leveraging fine-grained spatial information, with a global representation encoding that captures higher-level representations and incorporates both in a joint approach. To evaluate this challenging task in a real-life setting, a new benchmark dataset is proposed providing dense spatio-temporal grounding annotations in long, untrimmed, multi-action instructional videos for over 5K events. We evaluate the proposed approach and other methods on the proposed and standard downstream tasks showing that our method improves over current baselines in various settings, including spatial, temporal, and untrimmed multi-action spatio-temporal grounding.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an approach for learning spatio-temporal grounding with self-supervision and a new dataset. It proposes frame-wise alignment with local-global learning. Experiments show great results.

### Strengths
1. This paper is well-written and easy to follow.
2. The proposed dataset is interesting.

### Weaknesses
1. Novelty is limited. The proposed frame-wise matching is not new and has been well investigated in the temporal grounding field. Besides, the local-global learning is just borrowed from other similar tasks. The authors should carefully cite these previous works and provide discussion.

2. The experiments are not convincing. There is no statistics comparison between the proposed dataset and the existing datasets.

3. The compared methods are out-of-date. The authors should add more latest methods for comparison.

### Questions
1. Novelty is limited. The proposed frame-wise matching is not new and has been well investigated in the temporal grounding field. Besides, the local-global learning is just borrowed from other similar tasks. The authors should carefully cite these previous works and provide discussion.

2. The experiments are not convincing. There is no statistics comparison between the proposed dataset and the existing datasets.

3. The compared methods are out-of-date. The authors should add more latest methods for comparison.

### Soundness
2 fair

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel dataset called Grounding YouTube for spatial-temporal grounding in untrimmed videos. It further proposes a self-supervised learning approach for spatio-temporal action grounding trained on loose video and subtitle supervision only, outperforming a few self-supervised baseline approaches.

### Strengths
1. The paper proposes a novel dataset that combines both spatial and temporal grounding. Annotation details and evaluation protocols are clearly presented. Quality control indicates that the dataset is likely to have a high quality.

2. The paper compares a few self-supervised baseline methods on the proposed dataset and show the proposed method outperforms all of them.

### Weaknesses
1. ActivityNet-Entities [1] is an extension of ActivityNet-Captions that contains grounding annotations. This dataset seems quite relevant to what this submission is tackling. The submission might need to discuss ActivityNet-Entities and provide experimental comparison on this dataset if necessary.

2. "Pointing game accuracy" is not a commonly known metric. To make the submission more self-contained, a citation might be needed - firstly proposed in [2] if I recall correctly.

3. The description for "Mining: MLP" (1st row in Table 4) is missing, though it is referring to the same paper with MIL-NCE.

4. How is IoU+Pointing game computed? Is it simply an addition between IoU and the Pointing game accuracy? The authors may need to provide more details about how the metric is combined and the rationale.

5. Recent progresses on spatial grounding and temporal grounding are missing, to name a few: [3,4,5]. Particularly, my hunch is that the recent advances of open-vocabulary object detector may push the limits of spatial grounding by quite a lot since the objects in videos may appear often existing large-scale image grounding/detection datasets.

[1] Zhou, et al. "Grounded video description." CVPR 2019

[2] Zhang, et al. "Top-Down Neural Attention by Excitation Backprop". ECCV 2016

[3] Zhang, et al. "Glipv2: Unifying localization and vision-language understanding." NeurIPS 2022

[4] Yao, et al. "Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection." NeurIPS 2022.

[5] Chen, et al. "End-to-end Multi-modal Video Temporal Grounding". NeurIPS 2021

### Questions
My questions are already listed in the weakness section (see above).

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper addresses the task of spatio-temporal grounding, which involves localizing events in space and time within video data based on verbal descriptions. The authors propose a new approach that uses multimodal supervision, specifically video and subtitle data, without the need for human annotation. Specifically, they introduce a combination of local representation learning (for fine-grained spatial information) and global representation encoding (for higher-level representations). They also present a new benchmark dataset for evaluating spatio-temporal grounding in long, untrimmed, multi-action instructional videos. The proposed method outperforms current baselines in various settings.

### Strengths
1. The paper introduces a novel method for spatio-temporal grounding using multimodal supervision. This approach, which leverages both video and subtitle data without the need for human annotations, is a step closer to real-world senarios than the traditional methods that rely heavily on human-annotated sentences and bounding box supervision.
2. It is interesting to apply Sinkhorn-Knopp Optimal transport algorithm for key frame selection. If would be good if the authors could give some visualization results to show the effectiveness of this strategy. For example, showing the key frames selected by the algorithm. 
3. The authors have thoroughly evaluated their proposed method against other state-of-the-art techniques on both the new dataset and standard tasks. Their method's ability to outperform current baselines in various settings adds credibility to their approach.

### Weaknesses
1. The paper's approach to the global feature seems to primarily focus on aggregating frame-level appearance features. However, it does not appear to adequately model the temporal relation between key frames. This might pose a challenge in distinguishing actions driven by hand motions, such as 'pick up' and 'put down'. Could you provide insights into this aspect?
2. The choice to use key points instead of bounding boxes for the proposed dataset raises concerns. Key points can be inherently subjective, leading to inconsistencies in annotations. This subjectivity might affect the reliability and generalizability of any conclusions drawn from this dataset.
3. The rationale provided for certain choices, such as the preference for key points over bounding boxes due to distractions by object outlines, is not entirely convincing. A more robust justification or empirical evidence supporting such decisions would strengthen the paper's arguments.

### Questions
1. Could you provide visualization results to demonstrate the effectiveness of the Sinkhorn-Knopp Optimal transport algorithm in key frame selection? It would be particularly insightful to see the key frames selected by the algorithm.

2. The paper's approach to the global feature seems to focus on aggregating frame-level appearance features without adequately modeling the temporal relation between key frames. How does the model address the challenge of distinguishing actions driven by hand motions, such as 'pick up' and 'put down'?

3. Could you provide a more detailed justification or empirical evidence for the preference of key points over bounding boxes, especially in light of potential distractions by object outlines?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work tackles the challenging problem of spatio-temporal video grounding (STVG). They propose a framework trained on loose "annotations", where the ASRs, extracted from the videos, are used to align the multiple modalities. Although transcripts could provide a "free" source of information for STVG, they come with a significant challenge. ASR is noisy for various reasons, and not all the comments in the video are necessarily related to what is happening at that moment or even in the video itself. In this framework, they propose to combine global and local representations to learn STVG. They also present a frame selection strategy with optimal transport to deal with the loose annotations, in addition to a new benchmark dataset, which provides dense annotations in long, untrimmed, multi-action instructional videos.

### Strengths
- Challenging problem that is very important for the video analysis community.
- Although the Global and local representations have been explore in other works, the method is reasonable.
- In general the paper is well written.

### Weaknesses
- ActivityNet has a variant called ActivityNet entities (Grounded Video Description, Zhou, L. et. al.), which contains temporal and spatial annotations. It would be great to see the performance of this method on that dataset since it is a well-study benchmark for temporal grounding.

-  Since ActivityNet-Entities exists, I found the claim in section 4 "current downstream datasets either provide spatial, temporal annotations or short video clip with spatio-temporal annotations. These datasets do not provide the opportunity to evaluate both aspects, spatial and temporal grounding, in an untrimmed long video manner." not true.

- I found the notations confusing. Please see question 4.

### Questions
1. Why can we not use ActivityNet as a benchmark for STVG?

2. How long are the videos in the GoundingYoutube benchmark? Since I don't know how long the videos are, it is difficult to suggest how precise the method is. From Table 2, we can infer that the method has good recall and low precision. But what is the nature of the dataset? It could be that all the temporal groundings are happening in a certain portion of the video or by just predicting the beginning and endinding of the video as the temporal location is good enough for a 0.1 threshold (we have seen this in other datasets). Please look at Figures 2 and 4 of  "A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric" 

3. In the temporal grounding literature it is well established the tIoU@thresholds metrics and the mean_tIoU to evaluate the predictions, other spatial-temporal grounding works like HC-STVG use tIoU @ thresholds to evaluate the performance of the predictions. Can you evaluate the performance of your method using that metric? and take a look a  "A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric" which weighs the low IoU thresholds. I would like to see the bands 0.1, 0.3, 0.5, 0.7 and 0.9.

4. Section 3.1 "t \in {1,...,T} represents the number of frames in the video". Section 3.2, "our goal is to find T frames out of the U frames". can you please clarify this? is U or T the total amount of frames in the video? are those frames sampled from the video at what fps? 5? 

5. About the frame sampling strategy, I recommend looking at "DORi: Discovering Object Relationships for Moment Localization of a Natural
Language Query in a Video" where they use the sharpest frame in a small clip. It could help to get better frames for Sinkhorn-knopp alignment. I also think is missed related work since they don't use Bbox annotations.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
