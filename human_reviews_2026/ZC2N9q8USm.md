# ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
Detecting missing persons in forest environments remains a challenge, as dense canopy cover often conceals individuals from detection in top-down or oblique aerial imagery typically captured by Unmanned Aerial Vehicles (UAVs). While UAVs are effective for covering large, inaccessible areas, their aerial perspectives often miss critical visual cues beneath the forest canopy. This limitation underscores the need for under-canopy perspectives better suited for detecting missing persons in such environments. To address this gap, we introduce ForestPersons, a novel large-scale dataset specifically designed for under-canopy person detection. ForestPersons contains 96,482 images and 204,078 annotations collected under diverse environmental and temporal conditions. Each annotation includes a bounding box, pose, and visibility label for occlusion-aware analysis. ForestPersons provides ground-level and low-altitude perspectives that closely reflect the visual conditions encountered by Micro Aerial Vehicles (MAVs) during forest Search and Rescue (SAR) missions. Our baseline evaluations reveal that standard object detection models, trained on prior large-scale object detection datasets or SAR-oriented datasets, show limited performance on ForestPersons. This indicates that prior benchmarks are not well aligned with the challenges of missing person detection under the forest canopy. We offer this benchmark to support advanced person detection capabilities in real-world SAR scenarios. The dataset is publicly available at https://huggingface.co/datasets/etri/ForestPersons.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces 'ForestPersons', a new large-scale image dataset intended to support Search and Rescue (SAR) missions for missing persons in forested areas. The authors identify the limitations of existing high-altitude drone datasets, where the forest canopy obstructs the view of individuals below. As a solution, they have constructed a dataset of 96,482 images captured by simulating the perspective of Micro Aerial Vehicles (MAVs) at a low altitude of 1.5m–2.0m. This dataset includes diverse seasons (summer, fall, winter), lighting conditions, and poses of the missing person (standing, sitting, lying). A key feature is the quantification and labeling of occlusion levels caused by vegetation as a 'visibility level'.

### Strengths
- Clear Problem Definition: The paper successfully highlights the explicit limitations of existing high-altitude SAR datasets (visual obstruction by the canopy) and justifies the need for a specialized dataset focused on the 'under-canopy' environment.

- Data Diversity: The systematic effort to collect data across various seasons (notably including snow-covered winter scenes) and different poses (standing, sitting, lying) to address diverse scenarios in a forest environment is commendable.

- Quantification of Occlusion: A core strength of this dataset is that it does not avoid the key challenge of occlusion in forest environments. Instead, it defines and labels it as a 4-level 'visibility' attribute, providing a benchmark to evaluate model robustness against occlusion.

- Experimental Validation: The authors experimentally demonstrate that models trained on existing datasets (e.g., SARD, COCO) perform poorly on 'ForestPersons' (Table 2), thereby reinforcing the necessity and originality of the proposed dataset.

### Weaknesses
1. Unrealistic Simulation of MAV Flight (Domain Gap):
This dataset was not captured by an actual flying MAV, but rather "simulated" by a person holding a camera at 1.5–2.0m. Consequently, unique visual artifacts inherent to actual MAV flight are missing. Specifically, 'Motion Blur' caused by the MAV's rapid movement and vibration, and the 'Rotor Wash' phenomenon, where propeller downwash disturbs surrounding leaves and branches, are not reflected in the data. Therefore, models trained on this dataset may suffer a significant performance drop when deployed on an actual MAV due to this 'Domain Gap'.

2. Limitation of Data Modality (RGB-Only):
This paper relies exclusively on RGB (color) sensor data. However, many state-of-the-art SAR studies, including prior work mentioned by the authors (e.g., WiSARD), adopt a multimodal approach fusing RGB and Thermal imagery as a standard. In real-world forest environments, a missing person may be heavily occluded by bushes or camouflaged, making them impossible to identify with RGB alone. A thermal sensor plays a decisive role in such cases by detecting body heat. The reliance on RGB not only makes night-time detection impossible but also addresses a sub-optimal, simplified version of the real-world problem, as even daytime detection is severely limited.

3. Lack of Realism in Staged Scenarios:
This dataset was built by filming 'actors' performing 'staged missing person scenarios'. Such simulated situations may not adequately reflect the severity and atypical nature of actual distress situations. A real victim may be in a highly irregular pose due to injury or hypothermia, or may be partially buried under dirt, leaves, or debris. The poses in the sample images appear relatively distinct, suggesting a lack of realism in representing data from extreme, real-world scenarios.

4. Subjectivity of Core Annotations:
The reliability of the 'visibility level' label, one of the paper's key contributions, is questionable. The authors' own inter-annotator agreement analysis in Appendix F (Table 8) shows a Cohen's Kappa of approximately 0.45 for the 'visibility' attribute, which statistically represents only "moderate agreement". This implies that the core Ground-Truth label for visibility is highly subjective and noisy. Performance analyses based on this unreliable label (e.g., Figure 6) should be interpreted with caution.

### Questions
1. Regarding Unrealistic MAV Flight Simulation (Domain Gap)

Q1.1 (Domain Gap): This dataset was constructed using a handheld camera simulation rather than actual MAV flight. Consequently, dynamic environmental changes unique to MAV flight, such as 'motion blur' or 'rotor wash', are not included in the data. How do you assess the potential impact of this 'Domain Gap' on the performance of a model deployed on an actual MAV?

Q1.2 (Future Plans): To address this 'Domain Gap', do you have plans to collect data using an actual autonomous MAV in the future, or at least to augment the simulation data with realistic noise such as motion blur?

2. Regarding the Limitation of Data Modality (RGB-Only)

Q2.1 (Exclusion of Thermal): In forest SAR environments, thermal sensors play a decisive role in detecting victims occluded by vegetation, as demonstrated in prior studies (e.g., WiSARD) cited in your paper. Could you explain if thermal data was intentionally excluded from your dataset design, or if it was omitted due to collection difficulties?

Q2.2 (Scope Limitation): An RGB-only sensor makes night-time detection impossible and severely limits even daytime detection performance. Despite this limitation, what specific scenarios do you believe this RGB-only dataset can contribute to in real-world SAR operations?

3. Regarding the Lack of Realism in Staged Scenarios

Q3.1 (Realism of Poses): The dataset is built upon 'staged scenarios' featuring 'actors'. However, an actual victim might be in a much more irregular pose due to injury or hypothermia, or partially buried under dirt/leaves, than what is depicted in the sample images (e.g., Figure 3). What are your thoughts on the risk this discrepancy between 'staged' poses and 'actual' victim conditions poses to the model's generalization performance?

4. Regarding the Subjectivity of Core Annotations

Q4.1 (Annotation Reliability): The inter-annotator agreement for the 'visibility level', a key contribution of this paper, was only "moderate" (Cohen's Kappa $\approx$ 0.45, Appendix F). This suggests the label is highly subjective and noisy. Given this low reliability, do you believe the performance analysis based on this label (e.g., Figure 6) is valid and meaningful? We would like to hear your opinion on this.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper describes a dataset to benchmark object detection models on the task of detecting humans in a forest from a ground-level perspective. This is intended to improve the performance of below-the-canopy drones in search and rescue (SAR) operations. Most existing benchmarks employ above-the-canopy imagery, which would limit their performance, particularly during leaf-on season.
A suite of object detection models are benchmarked on the proposed dataset.

### Strengths
- Paper well written and structured, with correct experimental setup.
- Motivation for the task is made clear.

### Weaknesses
1. I appreciate the train-val-test split protocol used in this work, which is done at the sequence level. However, there are not enough details about the potential correlation between sequences. Is it possible that two sequences are captured the same day, on similar locations and with the same subjects? Overall, it would be useful to have some more information about the location of the sequences and the diversity of subjects (to make sure there’s no overfitting to a specific outfit or location type).
2. In the same line, there is no discussion about the potential generalization to other forest types. From the few photos in the paper, it would seem to be some type of temperate broadleaf forest. Although the varying conditions the videos where taken in do suggest the dataset covers a large diversity of environments, I can only imagine that this would hardly work in denser tropical forests. It would be helpful to get an indication of which biomes are covered by the dataset, in order to assess potential for geographical generalization.
3. This paper presents a dataset where the main edge is that it is capture in a different setting that other comparable datasets. As such, there is little novelty to speak of.  Novelty is typically a requirement according to the ICLR reviewer guidelines. I’m not 100% sure what this entails when it comes to datasets, but I would imagine enabling the benchmarking of so far un-benchmarkable tasks. The proposed dataset does not allow to evaluate methods on anything that is fundamentally different, although its different viewpoint and diversity will likely be helpful to train models that will be useful to practitioners. As such, it maybe worth questioning the adequacy of ICLR as a venue to publish this paper, although I do commend the authors for the quality of their work.

### Questions
I would like to read the response of the authors to the questions formulated in weaknesses 1 and 2:
- Can they provide statistics about commonalities between video sequences in terms of the location, time and subjects?
- Have the authors considered to which geographical locations they would expect the dataset to generalize to?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a dataset on under-tree canopy people detection, which is different from the over-canopy person detection datasets. Therefore, it contributes a significant new novel step forward in its domain, no dataset like this exists already. The data is big with a lot of varying background conditions. They provide a detailed benchmark with various backbones and also give an analysis on the possible weaknesses of this dataset as well  (inter-annotator dis-/ agreements, failure cases, recall etc). Also, the dataset is available on huggingface and I have downloaded and probed around in it to verify that it has the contents and labels that it claims.

### Strengths
Well written and thorough benchmarks with different levels of difficulty and settings.

### Weaknesses
The length of the clips is highly variable, between 50-450 frames. An analysis of how the number of frames available vs the person detection accuracy is needed. Does collecting more data on a scene from various angles help get better performance? What is the ideal number of frames, after which the gains are minimal?

One thing I don't feel comfortable is the pose classification. On an initial read, it feels  like they are providing actual human pose instead of what they have provided: lying down, sitting, and standing classifications.

### Questions
Is it possible to get an above canopy and a below canopy view, i.e. fly two different drones at the same time ? This could open generative applications, i.e. generating under canopy views from over-canopy views? Something like what was done in the AG-Reid.v2 dataset https://arxiv.org/pdf/2401.02634?

There has been a lot of interest in having fast FPS processing, especially on edge devices. It is not necessarily related to the quality of your dataset, but since you have already done experiments with multiple methods and backbones on different hardware, it will be interesting to see a column/section on the FPS of these various methods? Especially a FPS (or model size or compute time/memory requirement) vs accuracy and uncertainty?

Any experiments on if data attribute (pose, weather, location type,  weather etc) prediction or making the model aware of the attributes helps with improving person detection performance?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ForestPersons, a large-scale dataset for under-canopy missing person detection in Search and Rescue (SAR) missions, containing 96,482 images and 204,078 annotations collected from ground-level perspectives to simulate Micro Aerial Vehicle viewpoints. The dataset captures individuals in diverse poses (lying, sitting, standing) who are naturally partially occluded by vegetation, branches, and terrain across different seasons, weather conditions, and lighting. Individuals were positioned in different postures and naturally partially occluded by vegetation, branches, or uneven terrain, with annotations including bounding boxes, pose labels, and visibility levels. Experiments demonstrate large performance drops when applying models trained on existing SAR datasets to ForestPersons, and significant degradation from ground-level person datasets, establishing the need for domain-specific training data.

### Strengths
The experimental validation demonstrates a substantial practical problem, with existing SAR models showing catastrophic performance drops on under-canopy scenarios, providing strong empirical evidence for the dataset's necessity and filling a genuine gap in "Search and Rescue" applications that could have real-world impact for missing person detection.

### Weaknesses
-- limited technical and scientific novelty: This is mainly a domain-specific dataset contribution without methodological innovations in computer vision or machine learning. The work involves training standard object detection models on forest imagery and demonstrates expected domain transfer limitations, offering no new architectures, techniques, or fundamental insights beyond data collection for a specific application scenario.

-- narrow scope and generalizability: The dataset addresses a very specific task (under-canopy person detection for SAR) with limited broader applicability to computer vision research. While the data collection required significant effort and funding, the contribution is primarily valuable to SAR practitioners rather than advancing general object detection, occlusion handling, or robustness techniques that could benefit the wider research community.

-- simulated rather than authentic data: The dataset uses staged photography with handheld/tripod cameras positioned at 1.5-2m height to simulate MAV perspectives, rather than actual drone footage from real SAR missions, raising questions about ecological validity and whether the simulated conditions truly represent operational SAR scenarios.

### Questions
Have you evaluated modern Large Vision-Language Models like GPT-5/4o, Gemini, or Claude on your benchmark? (also open source models like MOLMO?) These models often demonstrate strong zero-shot object detection and description capabilities across diverse domains. Given that your evaluation focuses on traditional object detection models (YOLO, Faster R-CNN, etc.) mostly from 2015-2021, it's unclear whether the identified performance gaps persist with state-of-the-art VLMs that might already handle under-canopy person detection effectively without domain-specific training.

### Soundness
2

### Presentation
3

### Contribution
2
