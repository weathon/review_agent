# SAM 2: Segment Anything in Images and Videos

- Avg Score: 9.00
- Decision: Accept (Oral)
- Scores: 10, 8, 8, 10

## Abstract
We present Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos. We build a data engine, which improves model and data via user interaction, to collect the largest video segmentation dataset to date. Our model is a simple transformer architecture with streaming memory for real-time video processing. SAM 2 trained on our data provides strong performance across a wide range of tasks. In video segmentation, we observe better accuracy, using 3x fewer interactions than prior approaches. In image segmentation, our model is more accurate and 6x faster than the Segment Anything Model (SAM). We believe that our data, model, and insights will serve as a significant milestone for video segmentation and related perception tasks. We are releasing our main model, the dataset, an interactive demo and code.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper extends SAM to video, which can segment anything in images and videos. This paper has three significant contributions:
1. The paper expands SAM to video, enabling the segmentation of anything in video.
2. The paper develops a data engine for promptable video segmentation and constructs a large-scale video segmentation dataset, SA-V.
3. The paper designs a more efficient architecture for promptable image and video segmentation, demonstrating significant acceleration.

### Strengths
1. The paper is well-written and easy to follow. 
2. The paper has significant contributions, including a more efficient model architecture, a large-scale SA-V dataset, and fantastic performance.
3. The paper conducts comprehensive experiments and provides some valuable insights.

### Weaknesses
The paper does not have significant weaknesses. My concerns and suggestions are listed in the ``Questions" section.

### Questions
1. How would existing SOTA VOS methods perform after finetuning on the SA-V dataset? The paper uses SAM+Cutie as a baseline; however, Cutie has not been trained on large-scale data like SA-V. Therefore, I am interested in Cutie's performance after training with the SA-V dataset. After finetuning with SA-V, would SAM+Cutie surpass SAM2? This would not affect the paper's significant contributions, regardless of the results, as SAM2 is an end-to-end model.

2. I suggest the authors should cite some works from VIS [1, 2, 3, 4], VSS [5], and VPS [6, 7, 8] fields in the related works section, as these areas are also highly relevant to this paper.

[1] Video instance segmentation

[2] End-to-end video instance segmentation with transformers

[3] Tube-Link: A flexible cross tube framework for universal video segmentation

[4] DVIS: Decoupled video instance segmentation framework

[5] Vspw: A large-scale dataset for video scene parsing in the wild

[6] Video-kmax: A simple unified approach for online and near-online video panoptic segmentation

[7] Video k-net: A simple, strong, and unified baseline for video segmentation

[8] OMG-Seg: Is one model good enough for all segmentation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work addresses promptable visual segmentation in both images and videos. The primary contributions include a new task that generalizes image segmentation to the video domain, a new unified model for video and image segmentation, and a new dataset consisting of 35.5M masks across 50.9K videos. EExtensive evaluations across diverse benchmarks demonstrate that SAM 2 achieves state-of-the-art performance, highlighting its potential to enable "segment anything in videos."

### Strengths
* Compared to the original SAM model, SAM 2 improves segmentation accuracy, enabling more precise identification and segmentation of objects in images and videos.
* The processing speed is approximately six times faster than its predecessor. This allows SAM 2 to generate segmentation masks more quickly, making it suitable for real-time applications.
* SAM 2 exhibits strong zero-shot transfer capability.
* The training dataset includes 11 million images and 11 billion masks, providing a robust foundation for new video segmentation tasks for the community.
* The model and dataset are open-sourced.

### Weaknesses
From my perspective, there is no obvious weakness in this work. If must to say:
1. The claimed improvement in running speed is mainly due to the usage of the Hiera image encoder, which may not be viewed as a unique contribution of this study.
2. The primary contribution lies in a large-scale dataset and pre-trained models, while the technical contribution is relatively limited.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a strong foundation model for promptable visual segmentation in images and videos. It proposed a new data engine  that enhances model and data through user interaction, creating the largest video segmentation dataset to date.  A streaming memory augmented transformer is proposed  for real-time video processing.

### Strengths
1. This paper proposed a strong foundation model for the video and image segmentation. The data, model, and insights will serve
as a significant milestone for video segmentation.

2. The writing of the paper is good and the paper is easy to understand.

### Weaknesses
1. More experiments should be conducted. For example, more interactive VOS methods should be compared.
[1*] Modular interactive video object segmentation: Interaction-to-mask, propagation and difference-aware fusion. CVPR 2021
[2*] Memory aggregation networks for efficient interactive video object segmentation. CVPR 2020

2. More VOS datasets (e.g., VIPOSeg[4*]) should be included in this paper. 
[4*] Video Object Segmentation in Panoptic Wild Scenes. IJCAI 2023

### Questions
1. How is the annotation quality of the SA-V dataset, and how do you ensure the quality of the annotations? What is the difficulty level of this dataset (such as the movement of objects in the video, occlusions, etc.) compared to previous datasets?

2. Has SAM 2 attempted stability testing for results on ultra-long videos? Is object tracking in long videos more prone to errors?


3. If the memory bank is of fixed size, will it lead to forgetting when dealing with long videos?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
In this paper, the authors build a data engine to generate a large-scale video segmentation dataset. Using the datasets, they train a strong yet efficient model.

### Strengths
1. With the data engine pipeline, the paper provides an extremely large-scale video segmentation dataset compared to previous datasets. This will allow the researchers to tackle much more challenging tasks in video segmentation.

2. Based on the experimental results, the trained SAM2 model outperforms the combination of SAM and existing state-of-the-art trackers by a large margin. Therefore, the data scaling-up with the data engine is effective, as described by the authors. The results also imply the potential of further data scaling up with the data engine.

3. For image segmentation, the SAM2 model can also perform better, even with a much smaller computational cost. This will facilitate the applications of the SAM2 model. In a constrained platform, it is always better to have a good and efficient model.

4. The paper provides detailed information about the implementations of the data engine and data distribution. It is also good that the authors release their data and models. It would be even better if the training code and data engine could be publicly available.

### Weaknesses
1. Although this paper uses a simpler structure and performs well, it is still possible to use previous structures, such as Cutie [R1], to achieve even better performance with SAM2 data. It would be better if the structure could be explored.

2. SAM2 cannot recognize segmented objects like previous models [R2, R3]. It would be better to discuss this since it may limit the application of this paper. It would also be better to discuss the difference with [R4], which supports image and video segmentation and can recognize the objects.


[R1] Putting the Object Back into Video Object Segmentation.

[R2] Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively.

[R3] Semantic-SAM: Segment and Recognize Anything at Any Granularity.

[R4] OMG-Seg: Is One Model Good Enough For All Segmentation?

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
3
