# VideoMolmo: Spatio-Temporal Grounding meets Pointing

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Spatio-temporal localization—the ability to identify both the position and temporal evolution of objects—is essential for applications from cell tracking to autonomous navigation. Recent Video Large Multimodal Models (Video-LMMs) show promise but remain limited by coarse predictions, heavy reliance on dense mask optimization, and limited interpretability. We introduce VideoMolmo, a two-stage framework that grounds objects through point-based localization. Rather than directly predicting dense masks, VideoMolmo first produces precise points as lightweight, interpretable anchors, which are then used for downstream tasks including referring segmentation, video object segmentation, and counting. By decoupling localization from task execution, our approach provides more robust and transparent reasoning. Built on Molmo, our framework incorporates a temporal attention module for cross-frame reasoning and introduces a novel bidirectional temporal mask fusion strategy, enabling coherent point propagation and accurate segmentation. To facilitate training and evaluation, we release a large-scale spatio-temporal pointing dataset of 72k video–caption pairs with 100k annotated points and curate VPoS-Bench, a challenging benchmark spanning five real-world domains. Experiments show that VideoMolmo outperforms existing approaches, with gains of $5.4$ percentage points (pp) on VPoS-Bench and $9.5$ pp on MeViS. This highlights the effectiveness of point-based representations as a foundation for interpretable, fine-grained reasoning in dynamic visual environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper addresses the challenge of fine-grained spatio-temporal localization by replacing dense mask prediction with point-based grounding, enabling interpretable and efficient localization across frames. Built on Molmo, it introduces 1) a temporal attention module for cross-frame reasoning and 2) a bidirectional temporal mask fusion mechanism for coherent point propagation for video understanding tasks.

### Strengths
- The proposed approach outperforms previous approaches on multiple datasets at multiple metrics.
- The evaluation benchmark is exhaustive on multiple aspects of spatio-temporal evaluation.

### Weaknesses
- Architecture Novelty
   - Section 4.1: Temporal Module: Aggregating information using past frames feature aggregation is a very common aspect of trivial video understanding. For video segmentation or dense tasks the understanding from frames to patch level aggregation. Earlier works utilize temporal feature aggregation or memory module - the idea is a base setup not a novelty. If there’s something missing I would like authors to clarify. Specifically, if there’s some previous work used as a baseline and then made changes for this paper.
   - Section 4.2: BiDirectional Temporal Mask Fusion: In the bidirectional propagation, are the masks interpolated for intermediate frames (between i and i+n)? The fusion strategy looks more like a heuristic based on masks - a hyperparameter value dependent. If the overlap is significant enough then take intersection otherwise union. This heuristic might be generalizable enough across datasets, but that doesn’t make a novel contribution. It can’t be a contribution to the ICLR level main conference paper.
- Dataset
   - VPoS-Bench: Since dataset generation is a contribution of the paper, I was expecting more stats or details about the dataset. The datasets combined all have different properties in terms of video length and query aspect. Can authors please go more in detail about the dataset? The range contains from simple scenarios to complex datasets such as MeVis. I tried to look at appendix, however, I couldn’t find the details.
- Result
    - The paper should include more approaches for fair comparison -> Table 3 [1].

[1] Ding, H., Tang, S., He, S., Liu, C., Wu, Z., & Jiang, Y. G. (2025). Multimodal referring segmentation: A survey. arXiv preprint arXiv:2508.00265.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces VideoMolmo as a two-stage framework that grounds objects through point-based localization and a large-scale spatio-temporal pointing dataset of 72k video–caption pairs with 100k annotated points, and VPoS-Bench, a challenging benchmark spanning five real-world domains. Extensive experiments verify the effectiveness of the proposed framework and the benchmark.

### Strengths
1.The paper is well-written and easy to follow for readers.

2.A lot of quantitative experiments have been conducted to verify that VideoMolmo outperforms most previous state-of-the-arts across various downstream tasks.

### Weaknesses
1.One concern for this work is the motivation of the point-based grounding formulation. As mentioned in the manuscript, the point-level supervision is constructed from mask-level data, and it would be unclear why the point-based formulation would be better compared to other data formats such masks for visual grounding in videos? I think mask-based annotations can also transfer to various other forms like points, bounding boxes, etc. So the rationality of the point-based formulation of this work needs further clarifications and justifications.

2.There seem to be some missing details for the experimental comparisons. When comparing on the proposed VPoS-Bench, are the baselines like some video mllms retrained on the proposed spatio-temporal pointing dataset or just evaluated in a zero-shot manner? When comparing on other downstream tasks, is the proposed VideoMolmo trained from stratch on these downstream data or inherited with the pretrained knowledge of the proposed point-based dataset?

### Questions
Please refer to the weaknesses.

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
3

### Summary
The paper introduces VideoMolmo, a two-stage video–language model:
It separates the task of identifying an object (by points) from other downstream tasks like segmenting the object.
The first stage predicts points that represent object identity.
The second stage uses these points to guide downstream models.

### Strengths
- Leveraging of pretrained segmentation models to generate instance masks based on point prompts

- Strong performance against baselines

### Weaknesses
- The assumption that points make it clear what the model means is wrong. The Segment Anything Model already shows that if one points at an eye, the pointing is ambiguous: it could be the eye, the head, or the whole body that is pointed at.

- Limited novelty: The paper uses known object-centric principles, such as disentangling object appearance from position. In this case, position is disentangled from the downstream task of mask generation. Additionally, the temporal averaging is a simple mean computation. Overall the model is a straightforward extension of the Molmo model to the video domain with on clever object centric trick.

- Use of synthetic point labels generated with SAM-V2 that are tuned specifically to generate high IoU when used with SAM-V2 but are not evaluated otherwise.

### Questions
- Why is the temporal aggregation just a simple average? Doesn’t that smooth away important temporal details?

- Are there any human-based evaluations showing that the generated annotations of the training sets are really meaningful?

### Soundness
3

### Presentation
3

### Contribution
2
