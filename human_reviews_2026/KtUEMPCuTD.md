# MMHU: A Massive-Scale Multimodal Benchmark for Human Behavior Understanding in Autonomous Driving

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Humans are integral components of the transportation ecosystem, and understanding their behaviors is crucial to facilitating the development of safe driving systems. Although recent progress has explored various aspects of human behavior---such as motion, trajectories, and intention---a comprehensive benchmark for evaluating human behavior understanding in autonomous driving remains unavailable. In this work, we propose \textbf{MMHU}, a large-scale benchmark for human behavior analysis featuring rich annotations, such as human motion and trajectories, text description for human motions, human intention, and critical behavior labels relevant to driving safety. Our dataset encompasses 57k human motion clips and 1.73M frames gathered from diverse sources, including established driving datasets such as Waymo, in-the-wild videos from YouTube, and self-collected data. A human-in-the-loop annotation pipeline is developed to generate rich behavior captions. We provide a thorough dataset analysis and benchmark multiple tasks—ranging from motion prediction to motion generation and human behavior question answering—thereby offering a broad evaluation suite. Our dataset will be released to promote further human-centric research in this vital area of autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work presents MMHU, a video-motion-language annotation dataset on pedestrian in traffic scenario. The proposed MMLU is not only in very large scale, but also can and have been proved to support a variety of downstream tasks including motion prediction, motion generation and reasoning.

### Strengths
1. The pipeline is sound and nearly purely automatic. The video choice, the motion segmentation, the motion capture and language annotations processes are carefully designed and are basically sound for me.
2. The resultant dataset has clear applications and sufficient size. The authors have shown that the resultant dataset is (at least one of) the largest datasets in pedestrian's motion, which can be used for a variety of downstream safety-critical tasks autonomous driving.
3. The modalities contained in the dataset is diverse, across video, abstract motion and both low-level and high-level language annotations, which can support various downstream applications on research in this area.

### Weaknesses
1. **Dataset Quality** Although each step of building MMHU is carefully designed, it is inevitable the errors can accumulate throughout the process. For example, human labeling for fine-tuning VLM labeling model cannot 100% eliminate wrong language annotations; also other SOTA models in video-motion translation etc. cannot guarantee 100% correctness. Since the dataset is in a safety-critical domain, I hope the authors to a) at least do human evaluations on the dataset quality to let readers know the rough quality of the dataset, or to b) do human verification and corrections if funding and time allows.
2. **Missing Citations** I notice that in Tab. 1, the authors compare MMHU to language datasets in general driving. In this case, it might help to further add datasets like BDD-X and Rank2Tell to make the comparison more complete. Furthermore for automatic labeling pipeline, it might help to analyze difference to WOMD-Reasoning to justify the novelty in the labeling pipeline's design.

### Questions
Please see the weaknesses for points I think could further improve the quality of the manuscript.

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
3

### Summary
This work proposed a large-scale dataset (MMHU) focusing on human behavior understanding in Autonomous Driving scenarios. The data contains three sources (Waymo, YouTube driving videos, and self-collected data). The main contributions are the large data scale and the systematic annotation pipeline. Experiments on several different tasks also show that finetuning with MMHU can augment the performance on different tasks.

### Strengths
1. The large scale of the proposed data. MMHU aggregates ~1.73M frames and ~57K human instances from three sources, containing multiple types of labels: Motion, Trajectory, VQA, Text, and Behavior. 

2. The work proposes a systematic annotation pipeline, including human detection, tracking, SMPL reconstruction, frame interpolation, and the kinematic and behavioral text generation. It builds a systematic pipeline for large-scale, scalable annotation.

3. Experiments were conducted on multiple downstream tasks, including motion prediction, intention prediction, motion generation, and behavior VQA.

### Weaknesses
1. The main issue of this work is the lack of rigorous validation of data quality. For example, the Waymo dataset also provided Ground Truth 3D human pose keypoints and Trajectory, so it would be helpful to provide the differences between MMHU's labels and Waymo's official GT. 

2. Although the author mentioned the trajectory a lot, they didn't conduct experiments on trajectory prediction. Adding an experiment in trajectory prediction, especially pose-based trajectory prediction (e.g., [1],[2]), potentially showing pose/motion labels can augment the trajectory prediction performance, which can further validate the robustness and consistency of collected motion/pose and trajectory labels. 

[1] Saadatnejad, Saeed, et al. "Social-Transmotion: Promptable Human Trajectory Prediction." The Twelfth International Conference on Learning Representations.

[2] Taketsugu, Hiromu, et al. "Physical plausibility-aware trajectory prediction via locomotion embodiment." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

3. About data generalization, have you performed experiments to assess the generalization capability of models trained only on MMHU? For example, how does a model trained only on MMHU perform on the original test sets of 3DPW or JAAD for motion or intention prediction, without any fine-tuning on those datasets? This would be a much stronger test of your dataset's value.

### Questions
See weaknesses

### Soundness
3

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
In this paper, the authors propose MMHU, a large-scale benchmark dataset for human behavior understanding in autonomous driving scenarios. The authors develop a human-in-the-loop annotation pipeline that generates rich annotations, including 3D human motion and trajectories, hierarchical text descriptions (low-level and high-level), and labels for 13 critical behaviors relevant to driving safety. Compared to prior works, MMHU provides a benchmark with comprehensive human-centric annotations at a substantially larger scale. The authors evaluate baseline methods across three tasks (motion prediction, motion generation, and behavior VQA), demonstrating that existing models show limitations and can achieve improved performance when fine-tuned on MMHU.

### Strengths
1. The authors construct a large-scale dataset for human behavior understanding in autonomous driving scenarios.
2. The authors design a meaningful annotation pipeline for data curation, especially in handling human motion and trajectory extraction from monocular videos.
3. Comprehensive evaluations reveal the limitations of existing models, and the experiments demonstrate performance improvements on established benchmarks (e.g., 3DPW for motion prediction, JAAD for intention prediction) when incorporating the MMHU data.

### Weaknesses
1.  While substantial effort is devoted to human motion and trajectory processing, the text description and QA components rely primarily on VLM templates and prompting strategies. For example, the Behavior VQA portion consists of simple binary judgment questions (e.g., "Is the person crossing the street?"), which limits the methodological novelty.
2. The Critical Behavior Recognition (Section 3.4) does not adequately address the paper's own stated question: "What aspects of human behavior are critical to autonomous driving?" The 13 identified behaviors are obtained through simple VLM summarization, lacking justification for whether these specific behaviors comprehensively capture the critical aspects relevant to driving safety or if important behaviors are missing.

2. The critical behaviors annotation workflow raises methodological concerns: a subset receives human annotation for VLM fine-tuning, and then these VLM-generated labels serve as ground truth for fine-tuning other VLMs (Table 5). This remains confused. The paper should present comprehensive comparisons among: (1) the VLM used to generate labels, (2) VLMs without fine-tuning, (3) VLMs fine-tuned on VLM-generated labels, and (4) VLMs fine-tuned on human annotations only. 

4. The evaluation scope remains confined to human-centric tasks (motion prediction, motion generation, behavior VQA) without demonstrating impacts on autonomous driving systems. It remains unclear how these improvements benefit integrated driving models (e.g., planning, decision-making, or end-to-end driving frameworks).

5. Table ordering does not follow the narrative flow of the paper, affecting readability.

### Questions
Please see the weakenesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a novel large data set for human behavior understanding. They collect large amounts of data from multiple sources (Waymo, YT, others), and label it for several relevant tasks (such as motion prediction, intent prediction, and others. They describe how the data was labeled, and use it to evaluate several strong baselines on this data.

### Strengths
- Novel large-scale data collected and to be open-sourced.
- Very relevant topic being targeted.
- Quite extensive evaluation provided.

### Weaknesses
- Poorly written paper, with many details missing.
- Many parts of the methodology and data details are not sufficiently explained.
- Explanations are, in large part, quite high-level and handwavy.
- The evaluation, while extensive, is not discussed in too much detail.

### Questions
- Figure 1 is not referenced in text, also Table 7.
- The authors say in the title that the data is massive-scale, but that doesn't seem to be the case really.
- The sources of data are not clearly described; it is not clear which YT videos were included, details around self-collected data are missing, and even Waymo data is not really clearly specified. The authors do mention some details in the Appendix, but the Appendix is referenced only once in line 194, for an unrelated topic.
- Also, the included sensor data is not well explained.
- The interpolation is not discussed sufficiently.
- The human detector is not explained at all.
- SMPL is a large part of their methodology, yet it is not really described sufficiently. Even the acronym is not properly introduced.
- Line 237, "we leverage the bounding boxes as a prior", how exactly? Unclear.
- The percentages in Figure 3b don't have to sum to 1, no? But the values given do sum up to exactly 1, which is strange.
- Line 276, "we utilize LLMs to aggregate ...", how exactly? The authors don't provide sufficient detail. Similarly, in "The VLM is then prompted ...", how exactly?
- nit: "As shown in 2", Figure 2?
- How was data split into V, H, and T? Unclear.
- Why is T data so small?
- "which should be", should be or it is? Unclear what the authors wanted to say here?
- In general, this entire paragraph is quite tough to parse.
- "a closed-ended question", what exactly is meant by this?
- In the baseline evaluation section (5.3) and below, the authors just mention the results, without discussing them in sufficient detail (or at all), and they do not provide any potential insights.
- Line 448, "and MMHU", which subset?
- Tables are not referenced in the order they are enumerated in.
- Tables 6-8 titles repeat the setup already mentioned in the text. This can be removed to retrieve an entire paragraph worth of space.
- The authors do not provide any qualitative analysis, which should help improve the evaluation section.

### Soundness
2

### Presentation
1

### Contribution
2
