# Smoothing Slot Attention Iterations and Recurrences

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Slot Attention (SA) and its variants lie at the heart of mainstream Object-Centric Learning (OCL).
Objects in an image can be aggregated into corresponding slot vectors, by \textit{iteratively} refining cold-start query vectors, typically three times, via SA on image features.
For video, this aggregation is \textit{recurrently} shared across frames, with queries cold-started on the first frame while transitioned from the previous frame's slots on non-first frames.
However, cold-start queries lack sample-specific cues thus hindering precise aggregation on an image or a video's first frame;
Also, non-first frames' queries are already sample-specific thus requiring aggregation transforms different from the first frame.
We address these issues for the first time with our \textit{SmoothSA}:
(1) To smooth SA iterations on the image or video's first frame, we \textit{preheat} the cold-start queries with rich information of input features, via a tiny module self-distilled inside OCL;
(2) To smooth SA recurrences across all video frames, we \textit{differentiate} the homogeneous transforms on the first and non-first frames, by using full and single iterations respectively.
Comprehensive experiments on object discovery, recognition and downstream benchmarks validate our method's effectiveness.
Further analyses illuminate how our method smooths SA iterations and recurrences.
Our source code and training logs are provided in the supplement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SmoothSA, a novel method to enhance Slot Attention (SA) in object-centric learning for images and videos.  Its main contributions are: 1. addressing the query cold-start issue by preheating queries with informative features, and 2. tackling the transform homogeneity issue across video frames by differentiating transform operations for first and subsequent frames.

### Strengths
1. The paper addresses two long-overlooked limitations in mainstream Object-Centric Learning with Slot Attention and introduces two simple yet impactful techniques with minimal computational overhead to solve the problems: 1). Query Preheating; 2). Differentiated Transforms. 
2. The method is extensively evaluated across multiple tasks, datasets, and baselines, and achieves consistent performance.

### Weaknesses
1. The choice of 3 iterations for the first frame and 1 for non-first frames is empirical, not adaptive. The paper acknowledges this may not generalize to all scenarios, but it does not explore dynamic iteration adjustment.
2. While ablations show a Transformer decoder block (with swapped attention) is effective for preheating, the paper does not compare it to other lightweight architectures (e.g., MLPs) or analyze in detail why swapped attention outperforms standard attention (considering that in Table 5 the performance gap of swapping is huge).
3. Paper should provide a more detailed comparison of the computation overhead of each designed component.
4. The experiments are mainly conducted on standard and rather easy scenarios, and I wonder the model's effectiveness on more challenging scenarios. In VIS for example, author should test on OVIS, which is more occluded than YTVIS.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an essential method to smoothen slot attention iterations SmoothSA which aims to improvise on existing slot attention and provides a solutions to smoothen the predictions. Core contributions include addressing cold start issue as in noted in the paper.

### Strengths
The papers to claim that they are the first in addressing the query cold start issue in SA. While the problem sounds interesting I still have reservations.

The authors claim to solve for the first time addressing the transform homogeneity issue in SA recurrences across the first and non-first frames

Results seems competitive in comparison to previous methods

### Weaknesses
1. c1 needs more basis on why this issue needs to be looked at. While the authors say best of their knowledge. I am not convinced by this, is there any basis based on experiments, notes from other papers, any other proof documented?

2. I believe working on query initializations has been active as in indicated in BO-QSA, meta slot, https://arxiv.org/abs/2404.19654 also uses multiple queries for inference which boosts performance. It will be therefore interesting and a requirement on how the current strategy acts against the current method

3. c3 and c4 are virtually the same I do not see why they should be separate points

4. The experiments lack comparison to SOTA methods in SA, every table is almost compared to 1 method which I feel is not enough

5. The discussions lack references, it seems like this paper pretty much solves a lot of things even though no one else noted anything in the past

6. Comment 2 say as heavy RNN modules, in practice slot attention uses GRU which is not that heavy in theory, I do not get this point. Also Slot attention is independent of any overall encoding and decoding. More experiments and concrete numbers with results are needed to show this point

7. The analysis lacks any ablations and concrete discussions. The current discussions are too generic to show any point. This gives me a sense of unfinished work and early submission than intended

### Questions
See weaknesses

### Soundness
2

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
4

### Summary
The paper targets two frictions in Slot Attention (SA)–based object-centric learning (OCL): (1) cold-start queries on the first image/video frame, and (2) homogeneous aggregation transforms applied identically across all video frames despite their differing query conditions. The authors propose SmoothSA with two simple modifications: a tiny preheater that “preheats” first-frame queries using input features via self-distillation; and a differentiated recurrence policy using full three SA iterations on the first frame and a single iteration on non-first frames. Experiments on image (CLEVRtex, COCO, VOC) and video (YTVIS, VideoSAUR) OCL, plus downstream object recognition and VQA, show consistent gains over strong baselines such as SPOT and SlotContrast.

### Strengths
Clear identification of two practical bottlenecks in SA pipelines (first-frame cold start; transform homogeneity across frames), with a minimal, easy-to-adopt solution. 

Method simplicity: the preheater is a light Transformer decoder–style module trained via self-distillation inside the OCL model; the recurrence rule (3 iterations on the first frame, 1 on others) is plug-and-play. 

Broad empirical coverage: improvements reported on synthetic and real-world image datasets (ARI/ARI_fg/mIoU/mBO) and video datasets, with qualitative masks and downstream boosts in object recognition and VQA (GQA, CLEVRER). 

Positioning vs prior work is knowledgeable (e.g., BO-QSA, MetaSlot, SAVi/++, SlotContrast, STATM, SlotPi, RandSF.Q). The paper argues SmoothSA addresses issues orthogonal to prior query-initialization or query-prediction lines.

### Weaknesses
1. Unclear baseline performances:
The reported results for VideoSAUR and SlotContrast are substantially higher than those in the original papers, particularly in terms of FG-ARI. This discrepancy raises questions about the experimental comparability—whether the authors re-implemented these methods under different settings, used stronger backbones, or applied additional training tricks. A clear explanation is needed to ensure that the reported improvements are attributable to the proposed method rather than to differences in baseline implementations.

2. Inconsistent evaluation metrics:
The paper evaluates SmoothSA on both image and video datasets but seems to use  same metrics for them. In image datasets, the image FG-ARI measures spatial clustering quality, while in video datasets the video FG-ARI captures both spatial segmentation and temporal consistency. However, the paper interprets these results uniformly, without clarifying the differences. This makes it difficult to fairly assess the improvements and understand whether the gains arise from better object segmentation, temporal stability, or both.

3. Unclear experimental setup and sequence length generalization.
The paper does not specify the segment lengths used during training and testing. In video object-centric learning, training typically uses short clips while testing involves longer sequences—making sequence length generalization a central challenge. The absence of this information obscures how SmoothSA performs under distribution shifts in sequence length, and whether its recurrence design truly enhances temporal robustness.

### Questions
The main questions are already reflected in the Weaknesses section above, concerning (1) the discrepancy between reported and original baseline performances, (2) the unclear interpretation of evaluation metrics across image and video datasets, and (3) the lack of clarity regarding training/testing segment lengths and sequence length generalization.

### Soundness
3

### Presentation
3

### Contribution
3
