# L4Dog: Towards BEV Perception for Quadruped Robots in Complex Urban Scenes

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8

## Abstract
Embodied intelligence in quadruped robots faces significant challenges in complex urban environments due to the limitations of traditional perception systems and the lack of comprehensive datasets for exteroceptive 3D perception. To address this, we introduce L4Dog, the first large-scale exteroceptive 3D perception dataset tailored for quadruped robots in open urban scenarios. L4Dog provides high-quality 360-degree surround-view sensor data and manual annotations, covering diverse urban scenes such as traffic-light intersections, open roads, subway station, etc. By formulating perception tasks as bird’s-eye-view (BEV) space perception problems, we establish a multi-benchmark framework for BEV detection, tracking, trajectory prediction, and 3D traversable space occupancy estimation. The OmniBEV4D baseline method is proposed to unify multi-task perception (detection, tracking, prediction, and occupancy prediction) through shared temporal BEV features, enabling efficient and robust processing of dynamic urban environments. This work bridges the gap between current research and real-world deployment needs, offering a foundational resource for advancing autonomous navigation and decision-making in complex urban settings. The dataset will be made publicly available upon acceptance of this work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents L4Dog, a large-scale exteroceptive 3D perception dataset designed for quadruped robots in complex urban environments. It offers 360° surround-view sensor data with manual annotations across diverse scenes and introduces a unified BEV-based multi-task perception framework (OmniBEV4D) for detection, tracking, prediction, and occupancy estimation. The work aims to bridge the gap between research and real-world deployment by providing a comprehensive benchmark for embodied intelligence in urban navigation.

### Strengths
•  The dataset is of impressive scale and diversity, covering highly complex urban scenarios.

•  Collecting and annotating such large-scale multimodal data is a major effort and can benefit future research in robotics.

•  The unified BEV-based formulation provides a coherent framework for multiple perception tasks, which may inspire follow-up work.

### Weaknesses
•  Although the paper emphasizes embodied intelligence, it primarily focuses on perception tasks. It would be valuable to include benchmarks or discussions related to Vision-and-Language Navigation (VLN) or Visual Question Answering (VQA) to demonstrate broader embodied reasoning capabilities.

•  The paper presentation could be improved: the use of \citet and \citep is wrong, and table captions are placed awkwardly, affecting readability.

### Questions
How does the proposed OmniBEV4D framework generalize across different robot platforms or sensor configurations?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a dataset designed to evaluate BEV (Bird’s Eye View) perception for quadruped robots. Example BEV perception tasks include object detection, tracking, trajectory prediction, and occupancy prediction. The dataset contains both indoor and outdoor scenes and is substantially larger than those used in related studies.

The primary strength of this dataset lies in its scale. However, the dataset offers limitated coverage for complex terrain, which is a key capability of quadruped robots. Additionally, the paper does not clearly describe the identity anonymization procedures used in the dataset.

### Strengths
1. BEV perception for quadruped robots is an important research direction, and there is a clear need for a more comprehensive dataset in this area. The proposed dataset targests a key research gap in this domain. 
2. In terms of scale, it is significantly larger than those presented in related works.

### Weaknesses
1. Missing complex terrain: If I understand correctly, the proposed dataset primarily contains planar terrain scenarios, both indoor and outdoor. However, it lacks coverage of challenging vertical terrains such as hills, slopes, stairs, and elevators. This omission is significant because one of the key advantages of quadruped robots over wheeled robots is their ability to navigate complex and uneven terrain. As a result, the dataset is limited in its ability to support comprehensive evaluation of future quadruped robot perception research.
2. Additionally, the paper does not clearly describe the identity anonymization procedures used in the dataset.

### Questions
1. It would be valuable to include a comparison of motion statistics with related datasets. For example, presenting pitch, roll, yaw, and acceleration curves from the quadruped robot’s motion sensors could help assess the complexity of the terrains captured in this dataset.
2. Furthermore, the data collection process may involve identity-related information such as children, human faces, or vehicle license plates. The paper should clarify how such sensitive information is handled during data collection and processing to ensure proper anonymization and ethical compliance.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a large-scale BEV perception dataset for quadruped robots in complex urban scenarios, featuring high-quality manual annotations. A benchmark and baseline framework tailored for the evaluation and modeling of simultaneous multiple BEV perception tasks is concomitantly established. The paper is also the first one to contribute 360-degree occupancy annotations with a corresponding network.

### Strengths
1. To address the limitation of perception data for quadruped robotics in complex real-world urban scenarios, a large-scale 3D perception dataset with high-quality manual annotations for multiple BEV tasks is established.
2. The proposed dataset contains high-quality 360-degree 3D occupancy annotations for fine-grained quadruped robotics perception, with a corresponding occupancy prediction network.
3. A multi-task framework with shared temporal BEV space for multiple BEV perception tasks is developed.
4. Experimental results demonstrate the effectiveness of the multi-task framework.

### Weaknesses
1. Concern about the generalization ability. As shown in Fig. 3(b), the number of classes is fixed to several, which may hinder the generalization to objects beyond the predefined categories.
2. What's the detailed architecture of the proposed OmniBEV4D?

### Questions
1. What are the actual computational costs of OmniBEV4D, in terms of memory consumption, latency, etc?

### Soundness
3

### Presentation
3

### Contribution
3
