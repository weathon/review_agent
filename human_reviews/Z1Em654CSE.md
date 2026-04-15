# RGB-Event MOT: A Cross-Modal Benchmark for Multi-Object Tracking

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
Leveraging the power of contemporary deep learning techniques, it has become increasingly convenient for methodologies to recognize, detect, and track objects in real-world scenarios. Nonetheless, challenges persist, particularly regarding the robustness of these models in recognizing small objects, operating in low-illumination conditions, or dealing with occlusions. Recognizing the unique advantages offered by Event-based vision - including superior temporal resolution, vast dynamic range, and minimal latency - it is quickly becoming a coveted tool among computer vision researchers. To bolster foundational research in areas such as object detection and tracking, we present the first cross-modal RGB-Event multi-object tracking benchmark dataset. This expansive repository encompasses nearly one million carefully annotated ground-truth bounding boxes, offering an extensive data resource for research endeavors. Designed to augment the practical implementation of Event-based vision technology, this dataset proves particularly beneficial in intricate and challenging environments, including low-light situations, scenarios marked by occlusions, and contexts involving diminutive objects. The utility and potency of cross-modal detection and tracking models have been extensively tested and confirmed through our experimental studies. The encouraging results not only affirm the necessity of these models but also highlight their efficacy, thus emphasizing the benchmark’s potential to significantly propel the advancement of Event-based vision technology. We have included the code in the supplementary material and will make the dataset publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a novel cross-modal RGB-Event dataset for Multi-Object Tracking (MOT), aimed at addressing challenges in object tracking in complex real-world scenarios such as low-illumination conditions, small object detection, and occlusions. Utilizing the advantages of Event-based vision, known for its superior temporal resolution, vast dynamic range, and low latency, alongside conventional RGB data, the authors strive to advance the field of MOT. The newly developed dataset comprises nearly one million annotated ground-truth bounding boxes and is tested using state-of-the-art MOT algorithms, revealing a significant enhancement in performance with the integration of event data. The paper also explores the efficacy of different data fusion techniques, highlighting the potential of mask modeling over simple averaging. Through rigorous assessment and comparison with existing methods and datasets, the authors underline the potential of their proposed benchmark in driving further research and improving the robustness and versatility of detection and tracking systems, particularly in challenging visual scenarios. Besides, the authors acknowledge certain limitations of their dataset including static viewpoints and isolated hard cases, and suggest future directions for refining fusion techniques, embedding methods for event data, and development of specialized box association algorithms to better utilize the unique attributes of event data in MOT.

### Strengths
Here are some strengths of the paper:

	1. The paper is well-written and easy to understand. The authors provide clear explanations of the proposed algorithm and its components, as well as the motivation behind their approach.
	2. The paper introduces a unique cross-modal RGB-Event dataset for Multi-Object Tracking (MOT), significantly enriching the resources available for research in this field.
	3. The focus on overcoming practical challenges such as low-illumination conditions, occlusions, and small object detection aligns the paper with real-world needs in computer vision.
	4. Through thorough evaluation using state-of-the-art MOT algorithms, the paper substantiates the benefits of integrating event data with traditional RGB data.
	5. The authors intend to make the source code and the dataset publicly available upon acceptance, which fosters reproducibility and allows other researchers to build upon their work.

### Weaknesses
Here are some potential weaknesses:

	1. The exploration of data fusion techniques is somewhat limited with the utilization of simplistic averaging and mask modeling, which might not fully exploit the potential of cross-modal data fusion.
	2. The paper seems to focus on early fusion strategies, where RGB and Event data are fused at the input level. However, it does not explore or discuss middle or late fusion strategies, which could provide different perspectives and potentially better performance.
	3. The paper could have delved deeper into proficient embedding methods for event data, which is essential for leveraging the high temporal resolution of event data effectively.
	4. The paper does not delve into the discussion or evaluation of transformer-based methods for Multi-Object Tracking (MOT), which have been emerging as powerful tools for handling sequences and spatial relationships in data. 
	5. The paper does not provide information or discussion on the frame rate (FPS) of the tracker after incorporating event data. This is crucial as the processing speed is a vital aspect of real-time multi-object tracking applications.
	6. The paper aims to optimize detection performance through the integration of RGB and Event data, yet lacks discussion or specification on the particular detector used. This omission can lead to a lack of clarity and could hinder the reproducibility of the proposed methods.

### Questions
1. Could the authors elaborate on why only simplistic averaging and mask modeling were chosen for data fusion over more sophisticated techniques?
	2. Why were middle or late fusion strategies not explored, and do the authors anticipate different outcomes with these alternative fusion strategies?
	3. Could the authors provide more details on the embedding methods explored for event data and their impact on the system's performance?
	4. Have the authors considered integrating transformer-based methods for multi-object tracking, given their promise in sequence processing tasks?
	5. Can the authors provide the frame rate of the tracker post event data integration, and discuss its implications for real-time application?
	6. Could the authors specify the detector used, its integration with RGB and Event data, and the influence of the choice of detector on the results?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper first proposes the rgb-event multi-object tracking task which is new and interesting. It handles the low illumination, occlusion, and low-latency issues in the traditional rgb-based MOT task. It proposes a dataset that contains 12 videos for evaluation and also provides some baselines for future works to compare. For the baseline approach, the authors propose to fuse the dual modalities using concatenate or masking technique. This paper is well-written and the organization is good.

### Strengths
The paper first proposes the rgb-event multi-object tracking task which is new and interesting.

### Weaknesses
For the issues of this work:

the dataset is relatively small, 12 videos is not large-scale enough for current tracking tasks, especially in the big model era;
the baseline method is not novel, only simple fusion strategies are exploited; no novel fusion modules are proposed;
Therefore, I tend to reject this paper and encourage the authors to collect a larger rgb-event mot dataset or a more novel mot tracking framework.

### Questions
1. the dataset is relatively small, 12 videos is not large-scale enough for current tracking tasks, especially in the big model era; 

2. the baseline method is not novel, only simple fusion strategies are exploited; no novel fusion modules are proposed;

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a dataset for combined RGB and Event camera tracking. The initial focus of the paper is to motivate the use of Event cameras for the task, considering challenges like low-illumination, occlusions etc. The dataset is then described in detail. The paper then applies existing algorithms post merging the RGB and Event camera data in the feature space. The results are presented on the proposed dataset.

### Strengths
- The dataset with combined and calibrated RGB and Event camera data is valuable. And the data collection + annotation is the major strength of the paper.

- The need for using Event camera is well motivated 

- The paper is easy to read and understand

### Weaknesses
1. The method and experiments sections are vaguely presented. Several crucial details are missing:

(a) It is not clear, if a separate backbone (Figure 4) is used for both Event and RGB cameras. If yes, how were they trained?

(b) Was the proposed dataset used for training the detector?

(c) Was anything beyond the detector was trained or updated in the method? Was the retrained detector used in all the baseline methods?

(d) How was the Re-ID network trained? If not, which network was used for computing the Re-ID features?

(e) A common observation in several prior MOT paper is that Re_ID does not really play a significant role. The performance largely depends on the detection proposals and the motion model. An ablation without using the Re-ID features would be useful. 

(f) The paper does not talk about the motion model at all. An ablation with and without using any motion model would add value to the paper.  

(g) How exactly is averaging or masking done. Corresponding equations are warranted. It is extremely vague in the current form. 



2. The description is unclear at several places

(a) What is e in Eqn1?

(b) If \delta is a scalar why does it vary with time (Eqn1 \delta_t)



3. If one uses consecutive frame differences instead of the frames from the event camera, will that achieve similar gains?

### Questions
Please address the concerns raised in the weaknesses section. The method section is completely unclear in the current form.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
