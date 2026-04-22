# DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Video generation models, as one form of world models, has emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. 
In autonomous driving, this vision gives rise to driving world models—generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation.
Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment.
To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset—curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers—with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality.
DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work focuses on advancing world models for autonomous driving by introducing a new benchmark, DrivingGen. Unlike previous evaluations that primarily assess visual quality, DrivingGen offers a fine-grained assessment of physical realism, safety, and trajectory consistency, establishing a unified evaluation framework to promote the development of reliable, controllable, and deployable driving world models. Through benchmarking 14 state-of-the-art models, our study reveals clear trade-offs: general-purpose models produce visually appealing results but often violate physical constraints, whereas driving-specific models capture realistic motion dynamics yet fall short in visual quality.

### Strengths
1. The motivation is clear and valuable for advancing future driving world models.
2. The authors present a thorough empirical study: they evaluate a wide range of methods.
3. Comprehensive evaluation framework: by assessing methods along multiple dimensions — distribution, quality, temporal alignment, and trajectory alignment.

### Weaknesses
1. Although the metrics are intended for evaluating world models, most of them still focus on visual quality. There are relatively few metrics that assess whether the state transformations of the world model are reasonable and accurate.
2. Counterfactual reasoning is also an important capability of a world model. It could be interesting to consider evaluating this aspect in future work.
3. The evaluation of physical reasonable is somewhat lacking. It might be useful to draw inspiration from VBench 2.0 and incorporate metrics.

### Questions
1. How to deal with cases where the generated video fails to reconstruct trajectories? SLAM-based methods, do not successfully produce a trajectory for every video sequence.
2. DrivingDojo proposes the AIF metric to measure trajectory alignment. It might be worth discussing the advantages and limitations of the trajectory metrics proposed in this paper.
3. What is the approximate time and resource cost for evaluating 400 videos? This is also an important consideration when assessing the practicality of a benchmark.
4. AVG Ranking may not be an appropriate metric. Could a composite, weighted total score be designed to more comprehensively evaluate world model performance

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on evaluating driving video generation methods. Existing benchmarks emphasize image quality metrics (e.g., FID, FVD) and often fail to assess aspects critical to autonomous driving, such as the controllability of the ego vehicle's trajectory, the reasonableness of agent trajectories, and the consistency of 3D contents. Current benchmarks also lack data diversity in terms of weather, lighting, road conditions, and geographic locations. This work samples data from multiple datasets to enhance data diversity. proposes a set of metrics for evaluating video output, and benchmarks the performance of several state-of-the-art models.

### Strengths
Establishing a more practical benchmark for driving video generation is valuable. The contributions of providing more diverse test data and comprehensive evaluation metrics are clear. To maximize its contribution, the benchmark and evaluation code should be made open-source.

### Weaknesses
No major flaws were identified in the current work. However, the authors could further improve the benchmark by including evaluations for scene content controllability. While the paper addresses video quality, temporal consistency, and ego trajectory controllability, the controllability of generated scene contents( such as agents controlled with bounding boxes or roads via maps/lane) is also important for autonomous driving applications. These control signals can be extracted from videos using detection and segmentation methods if not available in the original datasets.

Typo: 'Qulity' in Table 2.

### Questions
see weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The DrivingGen benchmark is introduced as the first comprehensive evaluation framework for generative world models in autonomous driving, targeting key gaps where existing methods overlook safety-critical factors, trajectory plausibility, temporal consistency, and motion controllability. DrivingGen utilizes a highly diverse dataset spanning varied weather, time of day, global regions, and complex maneuvers essential for robust deployment. It introduces a suite of novel, specialized metrics that jointly assess performance across four dimensions: visual realism, trajectory plausibility, temporal coherence, and control fidelity. Benchmarking 14 state-of-the-art models revealed a key finding: a trade-off exists where general models may appear visually superior but "break physics," while driving-specific models prioritize realistic motion accuracy but often lag in visual quality, thus establishing a unified framework to guide future development.

### Strengths
1. The dataset distributions are balanced and diverse across all conditions. 

2. DrivingGen has created a new, specialized set of various measurements to evaluate generated driving videos; these are designed for the complexities of driving and are therefore more effective than standard video evaluation tools.

3. The experiments are comprehensive.

### Weaknesses
1. It would be more convincing to incorporate downstream task's performance (detection, mapping, planning) into the evaluation system. But this seems difficult since the data contains only front-view data.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DrivingGen, a comprehensive benchmark designed to evaluate generative world models for autonomous driving. The authors argue that existing benchmarks are limited in scope, failing to capture the full requirements of driving-specific simulation—such as visual realism, trajectory plausibility, temporal consistency, and controllability—while also lacking diversity in weather, geography, and driving maneuvers.

### Strengths
- DrivingGen is the first benchmark to jointly evaluate visual, kinematic, and interactive aspects of generative driving world models, addressing critical gaps in prior works.
- Introduces FTD for trajectory distribution, kinematic quality scores, and agent disappearance detection using VLMs—all tailored for driving safety and realism.
- The dataset includes under-represented conditions (e.g., night, snow, sandstorms) and global geographic variety, enabling more robust and realistic evaluation.
- The authors validate metric alignment with human preferences, increasing confidence in the benchmark’s practical relevance.

### Weaknesses
- With only 400 clips, the dataset may not fully represent the long-tail of real-world driving scenarios, despite its diversity.
- The benchmark focuses on open-loop video generation and does not assess models in interactive, closed-loop simulation settings.

### Questions
- Why was the dataset limited to 400 clips? Are there plans to expand it in the future to better cover rare and safety-critical scenarios?
- How do you mitigate the impact of video artifacts on SLAM and depth estimation, especially for models with low visual fidelity?

### Soundness
3

### Presentation
3

### Contribution
3
