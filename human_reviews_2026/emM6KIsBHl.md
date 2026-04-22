# OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 6

## Abstract
Smart factories use advanced technologies to optimize production and increase efficiency. To this end, the recognition of worker activity allows for accurate quantification of performance metrics, improving efficiency holistically while contributing to worker safety. OpenMarcie is, to the best of our knowledge, the biggest multimodal dataset designed for human action monitoring in manufacturing environments. It includes data from wearables sensing modalities and cameras distributed in the surroundings. The dataset is structured around two experimental settings, involving a total of 36 participants. In the first setting, twelve participants perform a bicycle assembly and disassembly task under semi-realistic conditions without a fixed protocol, promoting divergent and goal-oriented problem-solving. The second experiment involves twenty-five volunteers (24 valid data) engaged in a 3D printer assembly task, with the 3D printer manufacturer's instructions provided to guide the volunteers in acquiring procedural knowledge. This setting also includes sequential collaborative assembly, where participants assess and correct each other's progress, reflecting real-world manufacturing dynamics. OpenMarcie includes over 37 hours of egocentric and exocentric, multimodal, and multipositional data, featuring eight distinct data types and more than 200 independent information channels. The dataset is benchmarked across three human activity recognition tasks: activity classification, open vocabulary captioning, and cross-modal alignment. The dataset and associated code of OpenMarcie are publicly available at https://kaggle.com/datasets/47634878f107d35a57172f270241993d822fd67114c0f4d173b40b2dd9346a58

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents OpenMarcie, a multimodal dataset for activity recognition in industrial settings. It includes two scenarios: an ad-hoc bicycle assembly and a procedural 3D-printer assembly, captured with eight synchronized modalities (IMUs, barometer, thermal, spectrometer, RGB-LiDAR/depth, multiview RGB-D, stereo audio). It supports tasks such as human action recognition, captioning, and cross-modal alignment. Baseline models show that vision-inertial fusion consistently outperforms single modalities. The dataset emphasizes real-world variability, multimodal fusion, and ethical data handling.

### Strengths
1. Dual-scenario design: Captures both open-ended (goal-oriented) and procedural workflows, reflecting real variations in industrial activity.
2. Comprehensive sensing: Rich multimodal setup with 282 synchronized channels across ego and exo viewpoints, combining wearable, visual, and acoustic streams.
3. Thoughtful labeling: LLM-assisted annotation pipeline with multi-action verb–object–tool labels enhances expressiveness and scalability.
4. Fusion effectiveness: Particularly interesting finding that vision + IMU fusion consistently outperforms vision-only across all evaluated tasks, demonstrating the complementarity of motion and visual cues.
5.  Consistent results: Clear, reproducible baselines showing meaningful performance trends across modalities and benchmarks.
6.  Ethical data handling: Comprehensive consent, anonymization, and privacy measures (face blurring, exclusion of participant speech) are carefully documented.

### Weaknesses
1.  Controlled environment: Although participants come from over 20 countries, all recordings were conducted in a lab-like setting. This limits the dataset’s realism compared to actual industrial floors with variable lighting, background noise, and worker fatigue.
2. Audio realism: The recorded audio lacks machinery noise and natural acoustic interference, making it less representative of real industrial conditions. A noisy-environment extension or augmentation study would strengthen robustness claims.
3.  LLM-based annotation validation: The label translation pipeline using large language models is creative but would benefit from more quantitative validation, e.g., inter-annotator agreement with human-labeled subsets and an error analysis of mismatches.
4. Benchmark scope: Current benchmarks (HAR, captioning, cross-modal alignment) are strong baselines, but natural extensions such as step recognition, intent prediction, or error detection are not explored. Including even one additional task would better showcase dataset potential.
5. Documentation completeness: The paper lists many sensors and modalities but lacks a concise “data card” summarizing sampling rates, synchronization accuracy, calibration, and licensing. A structured summary table would make adoption easier.

### Questions
1. Label robustness: Could you provide quantitative evidence of the reliability of the LLM-assisted labeling process (e.g., agreement with human annotations or failure cases in translation)?
2. Synchronization and calibration: What is the precise time synchronization accuracy across ego/exo cameras and wearable sensors? Are calibration files (e.g., camera intrinsics/extrinsics) included in the release?
3. Cross-scenario generalization: Have you evaluated how models trained on one scenario (e.g., bicycle assembly) perform on the other (e.g., 3D printer assembly)?
4. Audio realism: Are there plans to include or simulate noisy industrial acoustic conditions to test robustness in realistic environments?
5. Benchmark extensions: Do you plan to add other tasks such as procedural step recognition, intent prediction, or error detection to demonstrate the dataset’s broader utility?

### Soundness
3

### Presentation
3

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
The paper introduces OpenMarcie, a large-scale, multimodal dataset designed for Human Activity Recognition (HAR) in industrial environments, which includes over 37 hours of data from 36 participants with two distinct assembly scenarios. The dataset integrates eight data types from wearable sensors and synchronized egocentric and exocentric cameras. The proposed dataset can be used for activity classification, open vocabulary captioning, and cross-modal alignment, making a contribution to HAR field.

### Strengths
1. Exceptional multimodal richness and scale: combine a wide range of synchronized data streams: egocentric/exocentric RGBD video, LiDAR, stereo sound, IMUs, thermal cameras, and spectrometers.
2. Realistic experimental scenarios: bicycle assembly and 3D printer assembly, which are specifically for an industrial environment.
3. Benchmarking and validation: apart from constructing a dataset, this paper also proposed strong baselines.
4. Use a high-quality annotation pipeline with human annotation and LLMs.

### Weaknesses
1. Limited demographic generalizability: for example, the majority of participants were male (24 of 36), right-handed (31 of 36), young (47% aged 22-24), and had an engineering background (72%). 
2. Missing real-world setting: the experiments were conducted in a lab or "test-bench" setting, not an actual real-world environment.
3. Underperformance of the audio modality: there is an obvious performance gap between audio-only modality and other single modality. In addition, adding the audio into other modalities will negatively affect the their performance. Could you please explain the reason?

### Questions
1. Could authors justify for the claim that this demographic profile is representative of the broader industrial workforce?
2. Have the authors considered experiments where realistic industrial noise is added to the data to simulate real-world conditions?
3. Could you explain why the audio modality has a negative effects on other modalities? Is caused by audio modality itself or fusion methanism?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a multimodal dataset designed for human action monitoring in manufacturing environments. The dataset focuses on two scenarios: bicycle and 3D printer assembly and disassembly.

### Strengths
The overall idea is quite interesting: to collect data relevant for manufacturing.

### Weaknesses
One weakness is the lack of diversity on the tasks (only two scenarios are included). It is not clear if the data collection approach is scalable. Given the size of the dataset and the limited set of tasks, it is unclear if this dataset offers something new that cannot be done with other existing datasets such as Ego4D. The main advantage of this dataset is the inclusion of more sensing modalities, but it is not clear this compensates the lack of diversity.

Despite that the goal is to collect data relevant for manufacturing, the data is recorded in a setting that does not seem to be a manufacturing setting. Some similar tasks are also part of Ego-Exo4D (https://docs.ego-exo4d-data.org/). In the previous link one can find examples related to repairing a bike with data captured with several devices. A discussion on the advantages of OpenMarcie over Ego-Exo4D is only done in table 2 (very briefly) and in the appendix A. Due to the similarities between OpenMarcie and Ego-Exo4D a more in depth discussion is needed.

OpenMarcie contains 37 hours of data. Table 1 indicates that 6% of Ego-Exo4D is industrial-relevant, corresponding to 72 hours, nearly twice OpenMarcie. Although Table 1 reports only percentages, the more informative metric is total recorded hours; by that measure, Ego-Exo4D is larger.

It will be good to also have an evaluation of the performance obtained on different tasks when a models is trained on Ego-Exo4D versus OpenMarcie. Given that the contribution is a new dataset, this dataset should be benchmarked against other existing datasets. 

Minor:

Figure 7 is not very informative. It will be good to add also the percentages in numbers. Otherwise, it is hard to read and one has to guess the numbers.

### Questions
Could do provide a more direct comparison between OpenMarcie and Ego-Exo4D? Include in table 1 the total number of hours of data available relevant for manufacturing in all datasets.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a multimodal dataset named OpenMarcie for industrial human activity understanding, covering two assembly scenarios (bicycle and 3D‑printer), 8 sensing modalities, synchronized ego/exo video, and wearables. It benchmarks HAR, open‑vocabulary captioning, and cross‑modal alignment with subject‑disjoint splits. An annotation pipeline mixes human labels with LLM‑based structured translation from natural‑language narrations, and the benchmarks consistently show that Inertial+Vision outperforms other modality choices. Faces are blurred and speech is removed to preserve privacy.

### Strengths
This is a good topic. Well‑scoped industrial focus with realistic goal‑driven workflows (incl. sequential collaborative assembly), bridging the gap between short scripted actions and long‑horizon procedures.

The dataset covers many modalities and placements (IMU, RGB‑LiDAR, thermal, spectrometer, stereo audio, ego/exo RGB‑D), enabling research on fusion, substitution, and privacy‑preserving sensing.

The benchmarks, models, and metrics are standard and appropriate; the late‑fusion transformer and InfoNCE‑based alignment are sensible choices. Reported numbers (e.g., HAR Macro‑F1 up to 0.882 on bicycle with Inertial+Vision) are generally consistent and align with intuition.

The writing is clear and easy to follow.

Authors state that dataset/code/splits will be released

### Weaknesses
External validity? Limited generalization capability?: data are collected in a test bench, not a running factory; acoustic environment also lacks authentic machinery noise—this likely underestimates the value of audio and may limit deployment realism.

LLM‑assisted labeling is scalable but introduces distributional priors; although consistency checks are reported, a more thorough human validation on a held‑out subset (e.g., inter‑annotator agreement vs. LLM outputs) would strengthen claims.

### Questions
How stable are the Inertial+Vision gains across different window sizes or temporal aggregation strategies (e.g., 2–5 s)?

For the LLM pipeline, what is the human time saving vs. purely manual labeling, and what is the error profile where LLMs disagree with humans?

### Soundness
3

### Presentation
3

### Contribution
3
