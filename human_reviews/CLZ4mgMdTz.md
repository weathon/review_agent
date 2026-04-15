# Sequence-SOD: Sequence-aware Spiking Object Detection for Event Cameras

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 3

## Abstract
Due to the asynchronous sensing of changes in illumination by event cameras, they are highly energy-efficient and therefore exhibit great potential especially in mobile, low power scenarios. Moreover, they are able to acquire sparse data with a high temporal resolution in the order of milliseconds and achieve a large dynamic range. This enables the recording of reliable data with minimal motion blur even during rapid movements and in low light scenarios. SNNs are particularly suitable for the processing of event data due to their asynchronous and spike-based functionality while their low energy consumption enables their deployment in automotive embedded applications. However, recent spiking object detectors do not leverage the full temporal information and only consider a single, fixed-size sample of the event data. In this paper, we propose the first sequence-aware SNN, which processes long sequences of the event stream data and predicts bounding boxes with a frequency of 40 Hz. In combination with a SSD network design, we are able to reach 26.88 mAP on the Gen1 Automotive Detection Dataset.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a Spiking Neural Network (SNN) based detector for event camera data called S-SOD. To leverage the temporal information of event sequences, S-SOD reuses the inner states of SNNs between time bins. Experimental results on Gen1 dataset show that S-SOD outperforms the SOTA SNN-based event detector, while still lagging behind RNN-based counterparts by a large margin.

### Strengths
- Temporal information is important for event-based object detection. The use of SNN's inner states is an intuitive way to leverage it
- Beating SOTA SNN-based detectors on Gen1

### Weaknesses
### Novelty
- The novelty of this work is limited. Both model architecture and training configs are the same as ODSNN. The only difference is the SNN module which reuses the inner states. Yet, this is just a minor change in my opinion. I would like to see more improvement, such as better SNN-based dense blocks instead of the naive one from ODSNN, or some SNN-based Transformer modules to better fuse temporal information

### Experiments
- The authors should conduct more experiments to validate their design. Only experimenting on one dataset is not enough. I would suggest the authors to test their method on 1MPx as well. Since the code of ODSNN has been released, it won't be very hard to compare with them on new datasets

### Questions
- The paper writing is unclear about how S-SOD reuses the inner states of SNNs. Does it reuse it across time bins (25ms) within a single time interval (125ms), or if it also reuse it across time intervals, i.e., keep reusing throughout the entire event sequence? Since objects in an event sequence might stop moving for more than 125ms, we need temporal information longer than a time interval to detect these objects. Therefore, I think only the latter case is reasonable
- It is still unclear to me how can S-SOD produce detections every 25ms instead of 125ms. The paper claims to "... by accumulating the network output over T time steps and dividing it by T", what does this mean? Are you running the SSD head to predict bboxes at the end of each time bin? Then what does accumlate+divide mean?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes sequence-aware SNN for object detection, which processes long-term event streams and predicts bounding boxes wit a frequency of 40 Hz. Besides, the authors design an event augmentation strategy for SNNs in object detection task.

### Strengths
1) The relevant background knowledge of this paper is clearly explained.

2) The topic of sequence-aware spiking object detection for event cameras is very interesting topic.

3) This paper processes long sequences of the event stream data and predicts bounding boxes with a frequency of 40 Hz. In combination with a SSD network design, which are able to reach 26.88 mAP on the Gen1 Automotive Detection Dataset.

### Weaknesses
1) Innovation is only the representation of event data, and there is no innovation in the network structure, which does not fundamentally solve the problem of sequence-aware.

2) Compared with “Cordone et al. (2022)”, in addition to the modification of data representation, what is the innovation of the network structure?

3) The network structure is not explained clearly. Not explained clearly "maintaining the network's inner state within a sequence". For example, How is the red arrow in Figure 1 (b) implemented in the network?

### Questions
See weaknesses

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates the problem of object detection using Spiking Neural Networks (SNN) with event data as input. It analyzes the characteristics of membrane potentials of pulse neurons and uses continuous input data to train the SNN while maintaining the internal state of the SNN during this time to fully utilize the temporal information of events.

### Strengths
1. By combining the characteristics of event data and the advantages of accumulating membrane potential information in SNN, the article does not reset the membrane potential state of pulse neurons during training. It processes multiple consecutive samples and evenly divides the events within a fixed time interval in each sample as sequential input, replacing the time step of SNN, which improves the performance of SNN-based event data object detection methods.
2. The article validates the impact of data augmentation on the model and demonstrates the low power consumption of SNN through calculations.

### Weaknesses
1. The contribution 1 of the paper is not highly innovative as training SNN with continuous data is already common. For example, the method has been used in "Event-based Video Reconstruction via Potential-assisted Spiking Neural Network (CVPR 2022)," and when compared to the method in "Deep Directly-Trained Spiking Neural Networks for Object Detection (CVPR 2023)," the performance does not show a significant difference.
2. The paper claims the ability to process real-world data at 40Hz, but the paper's meaning is limited to events occurring every 25ms, which depends on the input settings. This may not be a significant contribution point, or the paper should test the overall inference speed in FPS to confirm real-time performance.
3. The data augmentation experiments and energy calculations are conventional and lack notable highlights. Additionally, the overall writing quality of the paper is not high, and it is recommended to review and correct numerous grammar errors.

### Questions
1. Clearly articulate how your method differs from existing ones, even if they share some similarities in terms of using continuous data; Highlight where your approach outperforms or provides unique advantages compared to existing methods.

2. To strengthen the claim of real-world data processing at 40Hz, you should clarify the conditions and constraints under which this performance is achieved. Specify the input settings and provide a broader context for this achievement.
Consider conducting experiments to test the overall inference speed in FPS across various real-world scenarios and datasets, not just under specific input settings. This will provide a more comprehensive evaluation of your model's real-time performance.

3. Review the paper for grammar errors.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a full temporal SNN for spiking object detection, which achieves 26.88 mAP on the Gen1 Automotive Detection Dataset. However, their innovativeness is limited and the motivation for the study is not clear enough.

### Strengths
It seems to achieve good performance on the Gen1 Automotive Detection Dataset comparing the SNN-method.

### Weaknesses
1. This paper claims solved the problem of considering only a single, fixed-size sample of the event data in spiking object detection. However, the dataset Gen1 Automotive Detection Dataset used for experiments in the paper has multiple objects at the same time with variable sizes. The authors have not clearly stated the problem to be solved.
2. The main difference between spiking neurons and CNN neurons is the temporal memory through properties such as leakage, so the authors should explain how the "full temporal information" enhances SNNs.
3. Concerning the Sequence-SOD, it just increases the input event time domain length. The authors do not present any theory to support this approach and the method is too simple. As an academic paper, such contribution points are too few.
4. In conclusion, this manuscript needs to be further organized for motivation and to find the theoretical basis of the proposed method to inspire other scholars before it is published as a paper.

### Questions
Please See Weakness

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
